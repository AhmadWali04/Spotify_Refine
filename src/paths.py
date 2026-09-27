"""Project paths and small shared helpers."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
VECTORS = DATA / "vectors"
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
LIBRARY = RAW / "library.json"
TAGS = RAW / "tags.parquet"
FOLDS = DATA / "folds.json"
EXPERIMENTS = ROOT / "experiments.csv"

for _d in (RAW, VECTORS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)


def load_library(path: Path = LIBRARY) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.fetch.spotify` first "
            "(or `python -m src.fetch.synthetic` to test the loop without Spotify)."
        )
    return json.loads(path.read_text())
