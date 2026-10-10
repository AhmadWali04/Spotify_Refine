"""Lakh MIDI (LMD-matched) + tagtraum genre labels -> a clean, balanced, frozen subset (PRD3 M14).

    python -m src.data.lakh download                 # ~1.4 GB archive + labels into data/lmd/ (resumable)
    python -m src.data.lakh extract                  # only the best-matching MIDI per labeled song
    python -m src.data.lakh build --cap 400          # clean, balance, freeze folds -> data/lmd/subset.csv
    python -m src.data.lakh cache                    # re-cache representations after represent.VERSION changes
    python -m src.midi.experiment --data data/lmd/subset.csv --suite

Labels: tagtraum CD2 by default (majority genre, ~9.4k labeled LMD songs) or CD2C (consensus only,
~6.2k, cleaner). --multi adds CD2's minority genre as a second label. LMD has several MIDI versions per
MSD track; only the highest match score is kept. A file is dropped if it won't parse, is under 30 s, has
no tempo events (no beat information: pretty_midi would invent 120 bpm), or has no pitched notes. Each
genre is capped at --cap songs (random, seeded) and genres with fewer than --min-genre usable songs are
dropped. Folds are stratified by genre. The subset is frozen: build refuses to overwrite without --force.
Building also fills the representation cache (data/midi/reps/) that the experiment harness reads.
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mido
import numpy as np
import pandas as pd

from src import paths
from src.evaluate import N_FOLDS, SEED

BASE = "http://hog.ee.columbia.edu/craffel/lmd/"
URLS = {
    "lmd_matched.tar.gz": BASE + "lmd_matched.tar.gz",
    "match_scores.json": BASE + "match_scores.json",
    "msd_tagtraum_cd2.cls.zip": "https://www.tagtraum.com/genres/msd_tagtraum_cd2.cls.zip",
    "msd_tagtraum_cd2c.cls.zip": "https://www.tagtraum.com/genres/msd_tagtraum_cd2c.cls.zip",
}
MIN_SECONDS = 30.0


def lmd_dir() -> Path:
    return paths.DATA / "lmd"


# ---------------------------------------------------------------- download

def download(dest: Path | None = None, skip_archive: bool = False) -> None:
    """Resumable: a partial file continues from where it stopped (HTTP Range)."""
    dest = dest or lmd_dir()
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        if skip_archive and name.endswith(".tar.gz"):
            continue
        out = dest / name
        part = out.with_name(out.name + ".part")
        if out.exists():
            print(f"  have {name}")
            continue
        done = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        with urllib.request.urlopen(req, timeout=60) as r, part.open("ab" if done else "wb") as f:
            if done and r.status != 206:              # server ignored the range: start over
                f.seek(0)
                f.truncate()
                done = 0
            total = done + int(r.headers.get("Content-Length", 0))
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total > 50 << 20 and done % (100 << 20) < (1 << 20):
                    print(f"  {name}: {done >> 20} / {total >> 20} MB", flush=True)
        part.rename(out)
        if name.endswith(".zip"):
            with zipfile.ZipFile(out) as z:
                z.extractall(dest)
        print(f"  got {name}")


# ---------------------------------------------------------------- labels + matches

def load_labels(dest: Path | None = None, label_set: str = "cd2", multi: bool = False) -> dict[str, list[str]]:
    """MSD track id -> [majority genre] (+ minority genre with multi, CD2 only)."""
    out = {}
    for line in (dest or lmd_dir()).joinpath(f"msd_tagtraum_{label_set}.cls").open():
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        out[f[0]] = [g for g in f[1:3 if multi else 2] if g]
    return out


def best_matches(dest: Path | None = None) -> dict[str, tuple[str, float]]:
    """MSD track id -> (md5 of its highest-scoring MIDI, score)."""
    scores = json.loads((dest or lmd_dir()).joinpath("match_scores.json").read_text())
    return {msd: max(m.items(), key=lambda kv: kv[1]) for msd, m in scores.items() if m}


def member_name(msd: str, md5: str) -> str:
    return f"lmd_matched/{msd[2]}/{msd[3]}/{msd[4]}/{msd}/{md5}.mid"


def extract(dest: Path | None = None, label_set: str = "cd2") -> int:
    """One streaming pass over the archive, writing data/lmd/midi/<msd>.mid for every labeled song."""
    dest = dest or lmd_dir()
    labels = load_labels(dest, label_set)
    wanted = {member_name(msd, md5): msd for msd, (md5, _) in best_matches(dest).items() if msd in labels}
    out = dest / "midi"
    out.mkdir(exist_ok=True)
    n = 0
    with tarfile.open(dest / "lmd_matched.tar.gz", mode="r|gz") as tf:
        for m in tf:
            msd = wanted.get(m.name)
            if msd is None or not m.isfile():
                continue
            (out / f"{msd}.mid").write_bytes(tf.extractfile(m).read())
            n += 1
            if n % 1000 == 0:
                print(f"  extracted {n} / {len(wanted)}", flush=True)
    print(f"Extracted {n} MIDI files to {out}")
    return n


# ---------------------------------------------------------------- cleaning

def check(args: tuple[str, str, dict]) -> dict:
    """Worker: why a file is unusable, or its duration once its representation is cached."""
    import warnings
    from src.midi import experiment            # imported here so worker processes stay light to start
    warnings.filterwarnings("ignore", module="pretty_midi")   # LMD is full of slightly invalid files
    msd, path, rcfg = args
    try:
        mf = mido.MidiFile(path)
        if not any(msg.type == "set_tempo" for tr in mf.tracks for msg in tr):
            return {"track_id": msd, "reason": "no tempo events"}
        if mf.length < MIN_SECONDS:
            return {"track_id": msd, "reason": f"shorter than {MIN_SECONDS:.0f} s"}
        rep = experiment.cached_rep(msd, Path(path), rcfg)
        if rep.scalars["note_density"] == 0:
            return {"track_id": msd, "reason": "no pitched notes"}
        return {"track_id": msd, "reason": "", "duration_s": round(mf.length, 1),
                "n_beats": int(rep.summary["n_beats"][0])}
    except Exception as e:                      # corrupt files take many shapes in LMD
        return {"track_id": msd, "reason": f"unreadable: {type(e).__name__}"}


def stratified_folds(labels: list[str], seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(labels), dtype=int)
    for g in sorted(set(labels)):
        idx = np.flatnonzero(np.array(labels) == g)
        fold[rng.permutation(idx)] = np.arange(len(idx)) % N_FOLDS
    return fold


def build(dest: Path | None = None, label_set: str = "cd2", multi: bool = False, cap: int = 400,
          min_genre: int = 60, rcfg: dict | None = None, workers: int | None = None,
          force: bool = False) -> pd.DataFrame:
    dest = dest or lmd_dir()
    out_csv = dest / "subset.csv"
    if out_csv.exists() and not force:
        raise SystemExit(f"{out_csv} is frozen; pass --force to rebuild it (folds will change).")
    if rcfg is None:
        from src.models.symbolic.base import load_config
        rcfg = load_config().get("represent", {})
    labels = load_labels(dest, label_set, multi)
    matches = best_matches(dest)
    midi = dest / "midi"
    cands = sorted(msd for msd in labels if msd in matches and (midi / f"{msd}.mid").exists())
    if not cands:
        raise SystemExit(f"No MIDI files in {midi}; run `python -m src.data.lakh extract` first.")

    # Shuffle within genre so a cap takes a random sample, then check only as many as needed.
    rng = np.random.default_rng(SEED)
    by_genre: dict[str, list[str]] = {}
    for msd in rng.permutation(cands):
        by_genre.setdefault(labels[msd][0], []).append(str(msd))
    todo = [m for g in by_genre.values() for m in g[: int(cap * 1.5) + 20]]   # headroom for rejects
    print(f"Checking {len(todo)} of {len(cands)} labeled files ({len(by_genre)} genres)...")
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 1)) as ex:
        results = {r["track_id"]: r for r in ex.map(check, [(m, str(midi / f"{m}.mid"), rcfg) for m in todo],
                                                   chunksize=8)}

    rows, rejects = [], []
    for g, ms in by_genre.items():
        good = [results[m] for m in ms if m in results and not results[m]["reason"]]
        rejects += [results[m] for m in ms if m in results and results[m]["reason"]]
        if len(good) < min_genre:
            print(f"  drop {g}: {len(good)} usable songs (< {min_genre})")
            continue
        for r in good[:cap]:
            rows.append({**r, "path": str(midi / f"{r['track_id']}.mid"), "genre": g,
                         "label": ";".join(labels[r["track_id"]]), "match_score": round(matches[r["track_id"]][1], 4)})
    if not rows:
        raise SystemExit(f"No genre has {min_genre} usable songs; lower --min-genre.")
    df = pd.DataFrame(rows).drop(columns="reason")
    df["fold"] = stratified_folds(df["genre"].tolist())
    df = df[["track_id", "path", "label", "genre", "fold", "match_score", "duration_s", "n_beats"]]
    df.to_csv(out_csv, index=False)
    pd.DataFrame(rejects).to_csv(dest / "rejects.csv", index=False)
    print(f"\nWrote {out_csv.name}: {len(df)} songs, {df['genre'].nunique()} genres, {N_FOLDS} folds")
    print(df["genre"].value_counts().to_string())
    print(f"Rejected {len(rejects)}: " + ", ".join(f"{k} {v}" for k, v in Counter(r['reason'] for r in rejects).most_common()))
    return df


def warm_cache(dest: Path | None = None, workers: int | None = None) -> None:
    """Rebuild the representation cache for the frozen subset in parallel (after represent.VERSION changes)."""
    from src.models.symbolic.base import load_config
    rcfg = load_config().get("represent", {})
    df = pd.read_csv((dest or lmd_dir()) / "subset.csv")
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 1)) as ex:
        bad = [r for r in ex.map(check, [(t, p, rcfg) for t, p in zip(df["track_id"], df["path"])], chunksize=8)
               if r["reason"]]
    print(f"Cached {len(df) - len(bad)} representations" + (f"; {len(bad)} now fail: {bad[:3]}" if bad else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--labels-only", action="store_true", help="skip the 1.4 GB archive")
    e = sub.add_parser("extract")
    c = sub.add_parser("cache", help="rebuild cached representations of the frozen subset")
    c.add_argument("--workers", type=int)
    b = sub.add_parser("build")
    for p in (e, b):
        p.add_argument("--labels", choices=["cd2", "cd2c"], default="cd2")
    b.add_argument("--multi", action="store_true", help="add CD2's minority genre as a second label")
    b.add_argument("--cap", type=int, default=400, help="max songs per genre")
    b.add_argument("--min-genre", type=int, default=60, help="drop genres with fewer usable songs")
    b.add_argument("--workers", type=int)
    b.add_argument("--force", action="store_true", help="overwrite the frozen subset")
    a = ap.parse_args()
    if a.cmd == "download":
        download(skip_archive=a.labels_only)
    elif a.cmd == "cache":
        warm_cache(workers=a.workers)
    elif a.cmd == "extract":
        extract(label_set=a.labels)
    else:
        build(label_set=a.labels, multi=a.multi, cap=a.cap, min_genre=a.min_genre, workers=a.workers, force=a.force)


if __name__ == "__main__":
    main()
