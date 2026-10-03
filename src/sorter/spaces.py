"""Which vector space(s) the sorter uses.

By default: the featurizer chosen by `python -m src.decide` (data/decision.json), plus its
fallback for songs it can't vectorize. `--config` overrides it. Songs are scored in the first
space that covers them, with a model trained in that same space.
"""
from __future__ import annotations

from pathlib import Path

from src import paths
from src.vectors import load_config

DEMO_CONFIG = paths.CONFIGS / "demo_synthetic.yaml"


def resolve(lib: dict, lib_path: Path, config: str | None = None, fallback: str | None = None) -> list[dict]:
    """Returns the ordered list of configs (primary first)."""
    if config:
        cfgs = [load_config(config)] + ([load_config(fallback)] if fallback else [])
    else:
        dpath = paths.for_library(lib_path, "decision.json")
        if dpath.exists():
            import json
            d = json.loads(dpath.read_text())
            cfgs = [load_config(paths.ROOT / d["config"])]
            if d.get("fallback"):
                cfgs.append(load_config(paths.ROOT / d["fallback"]["config"]))
        elif lib.get("synthetic_truth"):
            cfgs = [load_config(DEMO_CONFIG)]
        else:
            raise FileNotFoundError(
                f"{dpath.name} not found. Finish the Phase 1 experiments and run `python -m src.decide`, "
                "or pass --config configs/<method>.yaml to pick a featurizer by hand.")
    if lib.get("synthetic_truth") is None and any(c["method"] == "synthetic" for c in cfgs):
        raise ValueError("The synthetic demo featurizer can't be used on a real library.")
    return cfgs
