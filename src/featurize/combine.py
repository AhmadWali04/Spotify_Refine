"""E4 (concatenation), E5 (concatenation + PCA) and E6 (interaction features).

params.blocks is a list of {method, params} sub-featurizers. Only songs covered by every
block are kept. Each block is z-scored and scaled by 1/sqrt(block size) so no source
dominates the cosine geometry just by having more dimensions.
"""
from __future__ import annotations

import numpy as np

from src.linalg import block_weight, pca, zscore


def _blocks(lib: dict, track_ids: list[str], params: dict):
    from src.featurize import FEATURIZERS

    specs = params["blocks"]
    outs = [FEATURIZERS[b["method"]](lib, track_ids, b.get("params", {})) for b in specs]
    common = set(track_ids)
    for ids, _ in outs:
        common &= set(ids)
    keep = [t for t in track_ids if t in common]
    mats = []
    for (ids, X), spec in zip(outs, specs):
        row = {t: i for i, t in enumerate(ids)}
        mats.append((spec, np.asarray(X, dtype=np.float64)[[row[t] for t in keep]]))
    return keep, mats


def _weighted(X: np.ndarray, weight: float = 1.0) -> np.ndarray:
    return weight * block_weight(zscore(X))


def featurize_concat(lib: dict, track_ids: list[str], params: dict):
    keep, mats = _blocks(lib, track_ids, params)
    X = np.hstack([_weighted(M, spec.get("weight", 1.0)) for spec, M in mats])
    return keep, X.astype(np.float32)


def featurize_concat_pca(lib: dict, track_ids: list[str], params: dict):
    keep, X = featurize_concat(lib, track_ids, params)
    Z, _, evr = pca(np.asarray(X, dtype=np.float64), params.get("k", 50))
    print(f"PCA: top {Z.shape[1]} components explain {100 * evr.sum():.1f}% of variance")
    return keep, Z.astype(np.float32)


def featurize_interactions(lib: dict, track_ids: list[str], params: dict):
    """E4 plus mood x top-genre products from the Essentia block (each product is a rank-1 term)."""
    from src.featurize.essentia_feats import NAMED_AXES

    keep, mats = _blocks(lib, track_ids, params)
    base = np.hstack([_weighted(M, spec.get("weight", 1.0)) for spec, M in mats])
    ess = [M for spec, M in mats if spec["method"] == "essentia"]
    if not ess:
        raise ValueError("interactions need an essentia block in params.blocks")
    E = ess[0]
    n_moods = params.get("n_moods", 5)
    n_genres = params.get("n_genres", 10)
    moods = E[:, :n_moods]
    genres = E[:, len(NAMED_AXES):]
    top = np.argsort(-genres.mean(axis=0))[:n_genres]
    inter = (moods[:, :, None] * genres[:, None, top]).reshape(len(keep), -1)
    return keep, np.hstack([base, _weighted(inter)]).astype(np.float32)
