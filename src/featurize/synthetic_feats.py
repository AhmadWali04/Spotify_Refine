"""Demo vectors for the synthetic library: each song sits near the direction of its hidden group.

Only meaningful for data/raw/library_synthetic.json (made by `python -m src.fetch.synthetic`).
It lets the whole product, review UI included, run before any API keys are set.
"""
from __future__ import annotations

import numpy as np


def featurize(lib: dict, track_ids: list[str], params: dict):
    truth = lib.get("synthetic_truth")
    if not truth:
        raise ValueError("The 'synthetic' featurizer only works on a synthetic library.")
    rng = np.random.default_rng(params.get("seed", 0))
    groups = sorted({g for t in track_ids for g in truth[t]})
    dims = params.get("dims", 32)
    dirs = {g: rng.standard_normal(dims) * params.get("separation", 2.2) for g in groups}
    X = rng.standard_normal((len(track_ids), dims))
    for i, t in enumerate(track_ids):
        for g in truth[t]:
            X[i] += dirs[g] / len(truth[t])
    return list(track_ids), X.astype(np.float32)
