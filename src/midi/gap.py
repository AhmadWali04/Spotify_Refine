"""S6: how much does transcription cost? (PRD3 §8, M17)

    python -m src.midi.gap --n 140                      # best-so-far config (S3), 140 Lakh songs
    python -m src.midi.gap --n 140 --run S4-A0          # any run id from experiment.SUITE

For N songs from the frozen Lakh subset (stratified by genre): crop a 30 s excerpt of the clean MIDI
(the length of a preview), render it to audio with FluidSynth + a GM soundfont, and transcribe that
audio with the full pipeline (src.midi.transcribe). Then, fold by fold, train on the clean full-length
songs outside the fold and score the held-out songs three ways:

    full       the clean full-length MIDI        (the S1-S5 setting)
    excerpt    the clean 30 s excerpt            (what a preview could give at best)
    transcribed the excerpt, rendered and transcribed

transcription gap = excerpt - transcribed. Also reports how close the transcribed matrices are to the
clean ones (key agreement, chord-transition and drum-grid cosine). One row per run -> symbolic_gap.csv.
Renders, excerpts and transcriptions are cached, so a rerun only does what is missing.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
from scipy.io import wavfile

from src import paths
from src.evaluate import N_FOLDS, SEED, topk_hits
from src.midi import experiment, represent, transcribe
from src.models.symbolic import base

EXCERPT_S = 30.0
SOUNDFONT = "GeneralUser-GS.sf2"
LOG_COLUMNS = ["timestamp", "run_id", "n_songs", "renderer",
               "top1_full", "top3_full", "top1_excerpt", "top3_excerpt", "top1_transcribed", "top3_transcribed",
               "gap_top1", "gap_top3", "key_agreement", "chord_trans_cos", "drum_grid_cos", "git_hash", "config"]


def s6_dir() -> Path:
    return paths.DATA / "lmd" / "s6"


def soundfont() -> Path:
    sf = paths.DATA / "soundfonts" / SOUNDFONT
    if not sf.exists():
        raise SystemExit(f"Missing {sf}. Download GeneralUser GS from "
                         "https://github.com/mrbumpy409/GeneralUser-GS (GeneralUser-GS.sf2) into data/soundfonts/.")
    return sf


def excerpt(pm: pretty_midi.PrettyMIDI, length: float = EXCERPT_S) -> pretty_midi.PrettyMIDI:
    """The window [30 s, 60 s) when the song is long enough (intros are rarely typical), else the last 30 s."""
    end = pm.get_end_time()
    t0 = 30.0 if end >= 60.0 + length / 2 else max(0.0, end - length)
    out = copy.deepcopy(pm)
    for inst in out.instruments:       # adjust_times drops notes ringing past the window; clip them instead
        for n in inst.notes:
            if n.start < t0 + length < n.end:
                n.end = t0 + length - 1e-4
    out.adjust_times([t0, t0 + length], [0.0, length])
    return out


def render(pm: pretty_midi.PrettyMIDI, path: Path, sr: int = transcribe.SR) -> Path:
    pm = copy.deepcopy(pm)
    for inst in pm.instruments:          # LMD drum tracks name kits a soundfont may lack (silent); use Standard
        if inst.is_drum:
            inst.program = 0
    y = pm.fluidsynth(fs=sr, sf2_path=str(soundfont()))
    y = y / max(np.abs(y).max(), 1e-9) * 0.9
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sr, (y * 32767).astype(np.int16))
    return path


def pick(subset: pd.DataFrame, n: int) -> pd.DataFrame:
    """About n songs, the same number per genre, seeded."""
    per = max(1, n // subset["genre"].nunique())
    return pd.concat([g.sample(min(per, len(g)), random_state=SEED)
                      for _, g in subset.groupby("genre")]).reset_index(drop=True)


def prepare(songs: pd.DataFrame, rcfg: dict) -> dict[str, tuple]:
    """track_id -> (clean excerpt rep, transcribed rep). Skips songs that fail at any step."""
    out = {}
    for i, r in enumerate(songs.itertuples()):
        tid = r.track_id
        clean_mid = s6_dir() / "clean" / f"{tid}.mid"
        try:
            if not clean_mid.exists():
                clean_mid.parent.mkdir(parents=True, exist_ok=True)
                excerpt(pretty_midi.PrettyMIDI(r.path)).write(str(clean_mid))
            wav = s6_dir() / "audio" / f"{tid}.wav"
            if not wav.exists():
                render(pretty_midi.PrettyMIDI(str(clean_mid)), wav)
            trans_mid = transcribe.transcribe(wav, track_id=f"s6_{tid}")
            out[tid] = (represent.from_path(clean_mid, tid, **rcfg).compact(),
                        represent.from_path(trans_mid, tid, **rcfg).compact())
            print(f"  [{i + 1}/{len(songs)}] {tid} ({r.genre}) ok", flush=True)
        except Exception as e:                 # a few LMD files break FluidSynth or produce silence
            print(f"  [{i + 1}/{len(songs)}] {tid}: skipped ({type(e).__name__}: {e})", flush=True)
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a.ravel() @ b.ravel() / (na * nb)) if na and nb else float("nan")


def evaluate_gap(cfg: dict, reps: list, labels: list[set[str]], fold: np.ndarray, pairs: dict[str, tuple]) -> dict:
    ids = [r.track_id for r in reps]
    hits = {k: {1: [], 3: []} for k in ("full", "excerpt", "transcribed")}
    for f in range(N_FOLDS):
        test = [i for i in np.where(fold == f)[0] if ids[i] in pairs]
        if not test:
            continue
        tr = np.where(fold != f)[0]
        m = base.make(cfg).fit([reps[i] for i in tr], [labels[i] for i in tr])
        y = [{m.classes_.index(c) for c in labels[i] if c in m.classes_} for i in test]
        views = {"full": [reps[i] for i in test],
                 "excerpt": [pairs[ids[i]][0] for i in test],
                 "transcribed": [pairs[ids[i]][1] for i in test]}
        for k, songs in views.items():
            S = m.scores(songs)
            for t in (1, 3):
                hits[k][t] += topk_hits(S, y, t).tolist()
    res = {f"top{t}_{k}": 100 * float(np.mean(hits[k][t])) for k in hits for t in (1, 3)}
    res["gap_top1"] = res["top1_excerpt"] - res["top1_transcribed"]
    res["gap_top3"] = res["top3_excerpt"] - res["top3_transcribed"]
    c, t = zip(*pairs.values())
    res["key_agreement"] = 100 * float(np.mean([a.key_name == b.key_name for a, b in zip(c, t)]))
    res["chord_trans_cos"] = float(np.nanmean([_cos(a.matrices["chord_trans"], b.matrices["chord_trans"]) for a, b in zip(c, t)]))
    res["drum_grid_cos"] = float(np.nanmean([_cos(a.matrices["drum_grid"], b.matrices["drum_grid"]) for a, b in zip(c, t)]))
    res["n_songs"] = len(pairs)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=140, help="songs to render and transcribe (spread over genres)")
    ap.add_argument("--run", default="S3", help="a run id from src.midi.experiment.SUITE")
    ap.add_argument("--subset", type=Path, default=paths.DATA / "lmd" / "subset.csv")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", module="pretty_midi")

    cfg = experiment.merge(base.load_config(), experiment.SUITE[a.run])
    rcfg = cfg.get("represent", {})
    subset = pd.read_csv(a.subset)
    songs = pick(subset, a.n)
    print(f"S6: {len(songs)} songs ({songs['genre'].nunique()} genres), config {a.run}")
    pairs = prepare(songs, rcfg)
    reps, labels, fold = experiment.load_data(str(a.subset), rcfg)
    res = evaluate_gap(cfg, reps, labels, fold, pairs)
    print(f"\n  {'':12} top-1   top-3   ({res['n_songs']} held-out songs)")
    for k in ("full", "excerpt", "transcribed"):
        print(f"  {k:12} {res[f'top1_{k}']:5.1f}   {res[f'top3_{k}']:5.1f}")
    print(f"  gap (excerpt - transcribed): top-1 {res['gap_top1']:+.1f}, top-3 {res['gap_top3']:+.1f} points")
    print(f"  transcribed vs clean: key agrees {res['key_agreement']:.0f}%, chord-transition cosine "
          f"{res['chord_trans_cos']:.2f}, drum-grid cosine {res['drum_grid_cos']:.2f}")
    log = paths.ROOT / "symbolic_gap.csv"
    new = not log.exists()
    with log.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"), "run_id": f"S6-{a.run}",
                    "renderer": f"fluidsynth:{SOUNDFONT}", "git_hash": experiment.git_hash(),
                    "config": json.dumps(cfg, sort_keys=True), **res})
    print(f"Logged to {log.name}")


if __name__ == "__main__":
    main()
