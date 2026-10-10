"""Symbolic experiments (PRD3 §8): 5-fold top-1 / top-3 / macro-F1 / NLL / ECE for either model.

    python -m src.midi.experiment --data synthetic                        # configs/symbolic.yaml as is
    python -m src.midi.experiment --data synthetic --symbolic-model vector --assignment A0
    python -m src.midi.experiment --data synthetic --suite                # S1-S5 in one go
    python -m src.midi.experiment --data data/lmd/subset.csv --suite      # once M14 builds the subset

--data is `synthetic` or a CSV with columns track_id, path (.mid), label (';' between several) and
optionally fold (0-4; frozen folds). Representations are cached in data/midi/reps/. One row per run
goes to symbolic.csv (symbolic_synthetic.csv for the synthetic data) with the full config and git hash.
Scores are calibrated as in src.sorter.compare: temperature fit on the other folds, judged on this one.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import paths
from src.evaluate import N_FOLDS, SEED, chance_rates, macro_f1, topk_hits
from src.midi import represent, synthetic
from src.models.symbolic import base
from src.sorter import calibrate

REPS = paths.DATA / "midi" / "reps"
LOG_COLUMNS = ["timestamp", "run_id", "data", "model", "n_songs", "n_classes",
               "top1_mean", "top1_std", "top3_mean", "top3_std", "macro_f1_mean", "macro_f1_std",
               "nll_mean", "ece_mean", "majority_top1", "chance_top1", "chance_top3", "fit_s",
               "git_hash", "config"]

# PRD3 §8 S1-S5 as overrides of the base config.
SUITE = {
    "S1": {"model": "markov", "markov": {"sequences": ["chords"], "order": 1}},
    "S2": {"model": "markov", "markov": {"sequences": ["chords", "pitch_classes", "drums"], "order": 1}},
    "S3": {"model": "markov", "markov": {"sequences": ["chords", "pitch_classes", "drums"], "order": 2, "alpha": 1.0}},
    "S4-A0": {"model": "vector", "vector": {"assignment_model": "A0"}, "augment": {"transpose": True}},
    "S4-A2": {"model": "vector", "vector": {"assignment_model": "A2"}, "augment": {"transpose": True}},
    "S4-A3": {"model": "vector", "vector": {"assignment_model": "A3"}, "augment": {"transpose": True}},
    "S5-A2-noaug": {"model": "vector", "vector": {"assignment_model": "A2"}, "augment": {"transpose": False}},
}


def merge(cfg: dict, over: dict) -> dict:
    out = copy.deepcopy(cfg)
    for k, v in over.items():
        out[k] = merge(out.get(k, {}), v) if isinstance(v, dict) else v
    return out


# ---------------------------------------------------------------- data

def _cached(track_id: str, path: Path, rcfg: dict) -> represent.SongRep:
    key = {"mtime": path.stat().st_mtime, **rcfg}
    f = REPS / f"{track_id}.pkl"
    if f.exists():
        saved = pickle.loads(f.read_bytes())
        if saved["key"] == key:
            return saved["rep"]
    rep = represent.from_path(path, track_id, **rcfg)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps({"key": key, "rep": rep}))
    return rep


def load_data(source: str, rcfg: dict, per_genre: int = 30):
    """(reps, labels as sets of names, frozen fold per song or None)."""
    if source == "synthetic":
        data = synthetic.dataset(per_genre)
        return [represent.from_midi(pm, tid, **rcfg) for tid, _, pm in data], [{g} for _, g, _ in data], None
    df = pd.read_csv(source)
    reps, labels, folds = [], [], []
    for r in df.itertuples():
        try:
            reps.append(_cached(str(r.track_id), Path(r.path), rcfg))
        except (ValueError, OSError, EOFError) as e:      # unreadable or empty MIDI: skip, report
            print(f"  skip {r.track_id}: {e}")
            continue
        labels.append(set(str(r.label).split(";")))
        folds.append(getattr(r, "fold", None))
    return reps, labels, (np.array(folds, dtype=int) if "fold" in df else None)


def make_folds(ids: list[str]) -> np.ndarray:
    order = np.random.default_rng(SEED).permutation(len(ids))
    fold = np.zeros(len(ids), dtype=int)
    fold[np.argsort(ids, kind="stable")[order]] = np.arange(len(ids)) % N_FOLDS
    return fold


# ---------------------------------------------------------------- evaluation

def evaluate(cfg: dict, reps: list, labels: list[set[str]], fold: np.ndarray) -> tuple[dict, np.ndarray, list[str]]:
    classes, y = base.class_index(labels)
    P = len(classes)
    t0 = time.time()
    S = np.zeros((len(reps), P))
    for f in range(N_FOLDS):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        if len(te) == 0 or len(tr) == 0:
            continue
        m = base.make(cfg).fit([reps[i] for i in tr], [labels[i] for i in tr])
        cols = [classes.index(c) for c in m.classes_]
        S[np.ix_(te, np.arange(P))] = -np.inf          # a class missing from training can't be predicted
        S[np.ix_(te, cols)] = m.scores([reps[i] for i in te])
    fit_s = time.time() - t0

    sizes = np.bincount([p for ls in y for p in ls], minlength=P).astype(float)
    per = {k: [] for k in ("top1", "top3", "macro_f1", "nll", "ece")}
    for f in range(N_FOLDS):
        te, rest = fold == f, fold != f
        if not te.any():
            continue
        y_te = [y[i] for i in np.where(te)[0]]
        per["top1"].append(100 * topk_hits(S[te], y_te, 1).mean())
        per["top3"].append(100 * topk_hits(S[te], y_te, 3).mean())
        per["macro_f1"].append(macro_f1(S[te], y_te, sizes))
        T = calibrate.fit_temperature(S[rest], [y[i] for i in np.where(rest)[0]])
        per["nll"].append(calibrate.nll(S[te], y_te, T))
        per["ece"].append(calibrate.ece(calibrate.softmax(S[te], T).max(axis=1),
                                        topk_hits(S[te], y_te, 1).astype(float)))
    c1, c3 = chance_rates(y, P)
    res = {"n_songs": len(reps), "n_classes": P, "fit_s": round(fit_s, 2),
           "majority_top1": 100 * float(np.mean([int(sizes.argmax()) in ls for ls in y])),
           "chance_top1": 100 * c1, "chance_top3": 100 * c3}
    for k, v in per.items():
        res[f"{k}_mean"] = float(np.mean(v))
        res[f"{k}_std"] = float(np.std(v))
    return res, S, classes


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def log(row: dict, path: Path) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=base.CONFIG)
    ap.add_argument("--data", default="synthetic", help="'synthetic' or a labels CSV")
    ap.add_argument("--symbolic-model", choices=["markov", "vector"])
    ap.add_argument("--assignment", help="A0-A4 or a src.sorter.models name (vector mode)")
    ap.add_argument("--suite", action="store_true", help="run S1-S5 instead of the config as is")
    ap.add_argument("--per-genre", type=int, default=30, help="synthetic songs per genre")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    cfg = base.load_config(a.config, a.symbolic_model, a.assignment)
    t0 = time.time()
    reps, labels, fold = load_data(a.data, cfg.get("represent", {}), a.per_genre)
    if fold is None:
        fold = make_folds([r.track_id for r in reps])
    print(f"{len(reps)} songs, {len(set().union(*labels))} classes ({time.time() - t0:.1f}s to represent)")

    runs = {k: merge(cfg, v) for k, v in SUITE.items()} if a.suite else {cfg.get("run_id", "S?"): cfg}
    data_name = "synthetic" if a.data == "synthetic" else Path(a.data).stem
    log_path = paths.ROOT / ("symbolic_synthetic.csv" if a.data == "synthetic" else "symbolic.csv")
    for i, (run_id, c) in enumerate(runs.items()):
        res, _, _ = evaluate(c, reps, labels, fold)
        if i == 0:
            print(f"  {'S0 floor':<14} majority top-1 {res['majority_top1']:5.1f}   "
                  f"chance top-1 {res['chance_top1']:5.1f}  top-3 {res['chance_top3']:5.1f}")
        print(f"  {run_id:<14} top-1 {res['top1_mean']:5.1f} ± {res['top1_std']:4.1f}   "
              f"top-3 {res['top3_mean']:5.1f} ± {res['top3_std']:4.1f}   F1 {res['macro_f1_mean']:.3f}   "
              f"NLL {res['nll_mean']:.2f}   ECE {res['ece_mean']:.3f}   ({res['fit_s']:.1f}s)")
        if not a.no_log:
            log({"timestamp": datetime.now().isoformat(timespec="seconds"), "run_id": run_id,
                 "data": data_name, "model": c["model"], "git_hash": git_hash(),
                 "config": json.dumps(c, sort_keys=True), **res}, log_path)
    if not a.no_log:
        print(f"Logged to {log_path.name}")


if __name__ == "__main__":
    main()
