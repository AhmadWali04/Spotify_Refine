"""Cached featurization shared by the experiment loop and the sorter.

Vectors live in data/vectors/<run_id>.npy + .index.csv and are reused while the
config and library match.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import paths
from src.linalg import block_weight, zscore


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def config_for_run(run_id: str) -> Path:
    """Find the config file whose run_id matches (experiments.csv only stores the run id)."""
    for p in sorted(paths.CONFIGS.glob("*.yaml")):
        if load_config(p).get("run_id") == run_id:
            return p
    raise FileNotFoundError(f"No config in {paths.CONFIGS} has run_id {run_id!r}")


def cached_featurize(cfg: dict, lib: dict, lib_path: Path, force: bool = False):
    """Returns (covered_ids, X_raw, featurize_minutes, config_hash)."""
    from src.featurize import FEATURIZERS

    run_id = cfg["run_id"]
    key = json.dumps({"method": cfg["method"], "params": cfg.get("params", {}),
                      "library": str(lib_path.name), "pulled_at": lib.get("pulled_at")},
                     sort_keys=True)
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    suffix = paths.library_suffix(lib_path)
    npy = paths.VECTORS / f"{run_id}{suffix}.npy"
    idx = paths.VECTORS / f"{run_id}{suffix}.index.csv"
    meta = paths.VECTORS / f"{run_id}{suffix}.meta.json"

    if not force and npy.exists() and meta.exists() and json.loads(meta.read_text())["hash"] == h:
        m = json.loads(meta.read_text())
        print(f"Using cached vectors {npy.name}")
        return pd.read_csv(idx)["track_id"].tolist(), np.load(npy), m["featurize_min"], h

    fn = FEATURIZERS[cfg["method"]]
    track_ids = list(lib["tracks"])
    t0 = time.time()
    ids, X = fn(lib, track_ids, cfg.get("params", {}))
    minutes = (time.time() - t0) / 60
    np.save(npy, X)
    pd.DataFrame({"track_id": ids}).to_csv(idx, index=False)
    meta.write_text(json.dumps({"hash": h, "featurize_min": minutes, "config": cfg}, indent=1))
    return ids, X, minutes, h


def build_X(X: np.ndarray, cfg: dict) -> np.ndarray:
    if cfg.get("standardize", False):
        X = zscore(X)
        if cfg.get("block_weight", False):
            X = block_weight(X)
    return X


def load_space(cfg: dict, lib: dict, lib_path: Path, force: bool = False):
    """Featurize (cached) and build X exactly as the experiment loop does. Returns (ids, X)."""
    ids, X_raw, _, _ = cached_featurize(cfg, lib, lib_path, force)
    return list(ids), build_X(np.asarray(X_raw, dtype=np.float64), cfg)
