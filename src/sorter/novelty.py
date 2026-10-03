"""Which unsorted songs are a new kind of music rather than a fit for an existing playlist?

Anchored spherical k-means: the playlist centroids are fixed centers, and K extra free centers
are fit to the unsorted songs. A song that ends up closer to a free center than to every
playlist centroid is a candidate "new" song.

Spare free centers will always find *something*, often the edge of an existing playlist. So each
free center must earn its place: its "gain" (how much closer its songs are to it than to their best
playlist, on average) has to beat the gains that spurious centers reach on songs we know are not
new. That null is measured by running the same procedure on held-out sorted songs, with centroids
from the other half, so the margin adapts to how tight your playlists actually are.

On the synthetic demo this catches ~9 in 10 brand-new-genre songs, against ~1/4 for a
nearest-neighbor similarity threshold.
"""
from __future__ import annotations

import numpy as np

from src.linalg import l2_normalize
from src.sorter.cluster import kmeans


def _anchored(U: np.ndarray, A: np.ndarray, k: int, min_size: int, iters: int, seed: int):
    """Returns (assignment: -1 = a playlist, j >= 0 = free center j; gain per free center)."""
    _, C = kmeans(U, min(k + len(A), len(U)), n_init=2, seed=seed)
    F = C[np.argsort(np.max(C @ A.T, axis=1))[:k]]       # start from the centers farthest from playlists
    for _ in range(iters):
        asg = np.argmax(U @ np.vstack([A, F]).T, axis=1) - len(A)
        alive = np.bincount(asg[asg >= 0], minlength=len(F)) >= min_size
        if not alive.any():
            return np.full(len(U), -1), np.zeros(0)
        newF = l2_normalize(np.array([U[asg == j].sum(axis=0) for j in np.where(alive)[0]]))
        if len(newF) == len(F) and np.allclose(newF, F):
            break
        F = newF
    asg = np.argmax(U @ np.vstack([A, F]).T, axis=1) - len(A)
    best_anchor = np.max(U @ A.T, axis=1)
    gain = np.array([np.mean(U[asg == j] @ F[j] - best_anchor[asg == j]) if (asg == j).any() else -np.inf
                     for j in range(len(F))])
    return np.where(asg >= 0, asg, -1), gain


def null_margin(X_labeled: np.ndarray, y: list[set], k: int, min_size: int, seed: int = 0) -> float:
    """Largest gain a free center reaches on held-out sorted songs (which are, by definition, not new)."""
    rng = np.random.default_rng(seed)
    half = rng.random(len(X_labeled)) < 0.5
    P = 1 + max(p for ls in y for p in ls)
    Y = np.zeros((len(y), P))
    for i, ls in enumerate(y):
        Y[i, list(ls)] = 1
    cnt = Y[half].sum(axis=0)
    A = l2_normalize((Y[half].T @ X_labeled[half])[cnt > 0] / cnt[cnt > 0][:, None])
    U = l2_normalize(X_labeled[~half])
    if len(U) < 2 * min_size:
        return 0.0
    _, gain = _anchored(U, A, k, min_size, 30, seed)
    return float(np.max(gain[np.isfinite(gain)], initial=0.0))


def anchored_new(X_unsorted: np.ndarray, centroids: np.ndarray, k: int | None = None,
                 min_size: int = 8, iters: int = 30, seed: int = 0, margin: float = 0.0) -> np.ndarray:
    """Boolean mask over X_unsorted: True where a free center whose gain beats `margin` claims the song."""
    n = len(X_unsorted)
    if n < 2 * min_size or len(centroids) == 0:
        return np.zeros(n, dtype=bool)
    k = k or default_k(n)
    asg, gain = _anchored(l2_normalize(X_unsorted), l2_normalize(centroids), k, min_size, iters, seed)
    keep = np.where(gain > margin)[0]
    return np.isin(asg, keep)


def default_k(n: int) -> int:
    return int(np.clip(n // 40, 2, 15))
