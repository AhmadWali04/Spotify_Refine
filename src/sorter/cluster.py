"""Group leftover songs (no good playlist match) into candidate new playlists.

Spherical k-means (cosine geometry, same as the rest of the project), k chosen by silhouette.
Clusters smaller than min_size are dissolved into "misc" (label -1).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score

from src.linalg import l2_normalize


def kmeans(X: np.ndarray, k: int, n_init: int = 5, iters: int = 60, seed: int = 0):
    """Spherical k-means with k-means++ seeding. Returns (labels, unit-length centers)."""
    Xn = l2_normalize(X)
    rng = np.random.default_rng(seed)
    best = (None, None, -np.inf)
    for _ in range(n_init):
        C = [Xn[rng.integers(len(Xn))]]
        for _ in range(1, k):
            d = 1 - np.max(Xn @ np.array(C).T, axis=1)          # cosine distance to nearest center
            p = np.maximum(d, 0) ** 2
            C.append(Xn[rng.choice(len(Xn), p=p / p.sum()) if p.sum() > 0 else rng.integers(len(Xn))])
        C = np.array(C)
        for _ in range(iters):
            lab = np.argmax(Xn @ C.T, axis=1)
            newC = np.array([Xn[lab == j].sum(axis=0) if (lab == j).any() else C[j] for j in range(k)])
            newC = l2_normalize(newC)
            if np.allclose(newC, C):
                break
            C = newC
        lab = np.argmax(Xn @ C.T, axis=1)
        obj = float(np.sum(Xn * C[lab]))                         # total cosine similarity to centers
        if obj > best[2]:
            best = (lab, C, obj)
    return best[0], best[1]


def auto_cluster(X: np.ndarray, k_max: int = 12, min_size: int = 8, seed: int = 0):
    """Returns (labels with -1 = misc, chosen k, silhouette)."""
    n = len(X)
    if n < 2 * min_size:
        return np.full(n, -1), 0, float("nan")
    best = (np.zeros(n, dtype=int), 1, -1.0)
    for k in range(2, min(k_max, n // min_size) + 1):
        lab, _ = kmeans(X, k, seed=seed)
        if len(set(lab)) < 2:
            continue
        s = float(silhouette_score(X, lab, metric="cosine"))
        if s > best[2]:
            best = (lab, k, s)
    lab, k, s = best
    lab = lab.copy()
    sizes = np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.array([])
    for j, size in enumerate(sizes):
        if size < min_size:
            lab[lab == j] = -1
    # Renumber 0..m-1, biggest cluster first.
    keep = [j for j in np.argsort(-np.bincount(lab[lab >= 0])) if (lab == j).any()] if (lab >= 0).any() else []
    remap = {old: new for new, old in enumerate(keep)}
    return np.array([remap.get(v, -1) for v in lab]), len(keep), s
