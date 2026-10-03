"""Project paths and small shared helpers."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
VECTORS = DATA / "vectors"
CACHE = DATA / "cache"           # per-track audio features: cache/essentia/<id>.npy, cache/clap/<id>.npy
AUDIO = RAW / "audio"            # iTunes 30-second previews: <track_id>.m4a
MODELS = DATA / "models"         # downloaded Essentia model graphs
APPLIED = DATA / "applied"       # one changelog per write to Spotify (used for undo)
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
LIBRARY = RAW / "library.json"
TAGS = RAW / "tags.parquet"
ITUNES = RAW / "itunes.parquet"
FOLDS = DATA / "folds.json"
EXPERIMENTS = ROOT / "experiments.csv"

for _d in (RAW, VECTORS, REPORTS, CACHE, AUDIO, MODELS, APPLIED):
    _d.mkdir(parents=True, exist_ok=True)


def load_library(path: Path = LIBRARY) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.fetch.spotify` first "
            "(or `python -m src.fetch.synthetic` to test the loop without Spotify)."
        )
    return json.loads(path.read_text())


def library_suffix(lib_path: Path) -> str:
    """'' for the real library, '_<stem>' for synthetic / alternate ones, so their outputs never mix."""
    return "" if Path(lib_path).resolve() == LIBRARY.resolve() else f"_{Path(lib_path).stem}"


def for_library(lib_path: Path, name: str) -> Path:
    """data/<name> for the real library, data/<stem>_<suffix>.<ext> otherwise."""
    p = DATA / name
    return p.with_name(f"{p.stem}{library_suffix(lib_path)}{p.suffix}")


def tags_path(lib_path: Path) -> Path:
    return TAGS.with_name(f"tags{library_suffix(lib_path)}.parquet")


def write_json(path: Path, obj) -> None:
    """Atomic write so a crash mid-save never leaves a half-written review or suggestions file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
