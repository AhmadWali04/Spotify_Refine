"""Pick the assignment model: every model on the same 5 folds as Phase 1, then calibrate the winner.

    python -m src.sorter.compare                         # spaces from data/decision.json
    python -m src.sorter.compare --config configs/e1b_tags_lsa.yaml
    python -m src.sorter.compare --library data/raw/library_synthetic.json

Appends one row per (space, model) to models.csv and writes data/model_choice.json, which
`python -m src.sorter.run` reads. Choice rule mirrors PRD1: the first model (in the preference
order of models.MODELS) whose top-3, top-1 and macro-F1 are all within one fold-std of the best.
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src import paths
from src.evaluate import N_FOLDS, labeled_set, load_or_make_folds, macro_f1, topk_hits
from src.sorter import calibrate, models, spaces
from src.vectors import load_space

LOG_COLUMNS = ["timestamp", "space", "model", "n_eval", "n_playlists",
               "top1_mean", "top1_std", "top3_mean", "top3_std", "macro_f1_mean",
               "nll_mean", "ece_mean", "fit_s", "chosen"]


def space_data(lib: dict, ids: list[str], X: np.ndarray, folds_path: Path):
    names, labels = labeled_set(lib)
    fold = load_or_make_folds(lib, labels, folds_path)
    row = {t: i for i, t in enumerate(ids)}
    ev = [t for t in sorted(labels) if t in row]
    return {"names": names, "labels": labels, "ids": ev, "X": X[[row[t] for t in ev]],
            "y": [labels[t] for t in ev], "fold": np.array([fold[t] for t in ev])}


def oof_scores(name: str, d: dict) -> np.ndarray:
    """Out-of-fold scores: every labeled song scored by a model that never saw it."""
    P = len(d["names"])
    S = np.zeros((len(d["ids"]), P))
    for f in range(N_FOLDS):
        tr, te = d["fold"] != f, d["fold"] == f
        if not te.any() or not tr.any():
            continue
        m = models.make(name).fit(d["X"][tr], [d["y"][i] for i in np.where(tr)[0]], P)
        S[te] = m.scores(d["X"][te])
    return S


def evaluate_model(name: str, d: dict) -> tuple[dict, np.ndarray]:
    t0 = time.time()
    S = oof_scores(name, d)
    fit_s = time.time() - t0
    sizes = np.zeros(len(d["names"]))
    for ls in d["labels"].values():
        for p in ls:
            sizes[p] += 1
    per = {k: [] for k in ("top1", "top3", "macro_f1", "nll", "ece")}
    for f in range(N_FOLDS):
        te, rest = d["fold"] == f, d["fold"] != f
        if not te.any():
            continue
        y_te = [d["y"][i] for i in np.where(te)[0]]
        per["top1"].append(100 * topk_hits(S[te], y_te, 1).mean())
        per["top3"].append(100 * topk_hits(S[te], y_te, 3).mean())
        per["macro_f1"].append(macro_f1(S[te], y_te, sizes))
        # Calibrate on the other folds' out-of-fold scores, judge on this one.
        T = calibrate.fit_temperature(S[rest], [d["y"][i] for i in np.where(rest)[0]])
        per["nll"].append(calibrate.nll(S[te], y_te, T))
        Pm = calibrate.softmax(S[te], T)
        per["ece"].append(calibrate.ece(Pm.max(axis=1), topk_hits(S[te], y_te, 1).astype(float)))
    res = {"model": name, "n_eval": len(d["ids"]), "n_playlists": len(d["names"]), "fit_s": round(fit_s, 2)}
    for k, v in per.items():
        res[f"{k}_mean"] = float(np.mean(v))
        res[f"{k}_std"] = float(np.std(v))
    return res, S


def choose(results: list[dict]) -> dict:
    """Simplest model within one fold-std of the best on top-3, top-1 and macro-F1. Macro-F1 is
    there so a model can't win by funnelling every song into the biggest playlists."""
    # A floor on the band, so a best score with ~0 std across folds doesn't make ties impossible.
    floor = {"top3": 0.5, "top1": 0.5, "macro_f1": 0.01}
    best = {m: max(results, key=lambda r: r[f"{m}_mean"]) for m in floor}
    for r in results:                                   # MODELS is in preference order
        if all(r[f"{m}_mean"] >= b[f"{m}_mean"] - max(b[f"{m}_std"], floor[m]) for m, b in best.items()):
            return r
    return best["top3"]


def calibration(S: np.ndarray, d: dict, target_precision: float) -> dict:
    T = calibrate.fit_temperature(S, d["y"])
    Pm = calibrate.softmax(S, T)
    conf = Pm.max(axis=1)
    hit1 = topk_hits(S, d["y"], 1).astype(float)
    hit3 = topk_hits(S, d["y"], 3).astype(float)
    conf_thr = calibrate.confident_threshold(conf, hit1, target_precision)
    return {
        "temperature": T,
        "confident_threshold": conf_thr if np.isfinite(conf_thr) else None,   # None: no tier is that precise
        "abstain_threshold": calibrate.abstain_threshold(conf, hit3),
        "target_precision": target_precision,
        "oof_confident_share": float(np.mean(conf >= conf_thr)),
    }


def run(lib: dict, lib_path: Path, cfgs: list[dict], model_names: list[str] | None = None,
        target_precision: float = 0.9, log: bool = True) -> dict:
    suffix = paths.library_suffix(lib_path)
    folds_path = paths.DATA / f"folds{suffix}.json"
    out = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "pulled_at": lib.get("pulled_at"), "spaces": []}
    for cfg in cfgs:
        ids, X = load_space(cfg, lib, lib_path)
        d = space_data(lib, ids, X, folds_path)
        print(f"\n== space {cfg['run_id']}: {len(d['ids'])} labeled songs, {len(d['names'])} playlists")
        results, scores = [], {}
        for name in model_names or list(models.MODELS):
            r, S = evaluate_model(name, d)
            results.append(r)
            scores[name] = S
            print(f"  {name:<16} top-1 {r['top1_mean']:5.1f} ± {r['top1_std']:4.1f}   "
                  f"top-3 {r['top3_mean']:5.1f} ± {r['top3_std']:4.1f}   F1 {r['macro_f1_mean']:.3f}   "
                  f"NLL {r['nll_mean']:.2f}   ECE {r['ece_mean']:.3f}   ({r['fit_s']:.1f}s)")
        win = choose(results)
        cal = calibration(scores[win["model"]], d, target_precision)
        print(f"  -> chosen: {win['model']}  (T={cal['temperature']:.3g}, confident >= "
              f"{cal['confident_threshold'] or float('inf'):.2f} covers {100 * cal['oof_confident_share']:.0f}% at "
              f"{100 * target_precision:.0f}% precision, abstain < {cal['abstain_threshold']:.2f})")
        out["spaces"].append({"run_id": cfg["run_id"], "config": cfg, "model": win["model"],
                              "metrics": win, "calibration": cal, "all": results})
        if log:
            log_path = paths.ROOT / f"models{suffix}.csv"
            new = not log_path.exists()
            with log_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
                if new:
                    w.writeheader()
                for r in results:
                    w.writerow({**{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()},
                                "timestamp": out["generated_at"], "space": cfg["run_id"],
                                "chosen": r["model"] == win["model"]})
    paths.write_json(paths.for_library(lib_path, "model_choice.json"), out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=str(paths.LIBRARY))
    ap.add_argument("--config", help="featurizer config (default: data/decision.json)")
    ap.add_argument("--fallback", help="second featurizer config for songs the first can't cover")
    ap.add_argument("--models", nargs="*", help=f"subset of {list(models.MODELS)}")
    ap.add_argument("--precision", type=float, default=0.9, help="top-1 precision for the 'confident' tier")
    args = ap.parse_args()
    lib_path = Path(args.library)
    lib = paths.load_library(lib_path)
    run(lib, lib_path, spaces.resolve(lib, lib_path, args.config, args.fallback), args.models, args.precision)


if __name__ == "__main__":
    main()
