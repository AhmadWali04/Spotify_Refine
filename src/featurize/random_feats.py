"""E0: random Gaussian vectors. Sets the chance-level floor."""
from __future__ import annotations

import numpy as np


def featurize(lib: dict, track_ids: list[str], params: dict):
    rng = np.random.default_rng(params.get("seed", 0))
    return list(track_ids), rng.standard_normal((len(track_ids), params.get("dims", 64)))
