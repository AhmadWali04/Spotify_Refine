"""Turn raw model scores into probabilities and decide how much to trust them.

  - Temperature softmax: p = softmax(S / T), with T fit to minimize held-out negative log-likelihood.
    A song in several playlists is "right" if any of them is predicted, so its likelihood is the
    total probability on its playlists.
  - Confident threshold: the lowest top-1 probability at which held-out top-1 precision still
    reaches the target (these songs get a "confident" badge and can be bulk-accepted).
  - Abstain threshold: below it, held-out top-3 accuracy drops under 50%; those songs go to
    the leftover pool to be clustered into new playlists.

Whether a song is a new kind of music altogether is decided separately (src/sorter/novelty.py).
"""
from __future__ import annotations

import numpy as np


def softmax(S: np.ndarray, T: float) -> np.ndarray:
    Z = S / T
    finite = np.isfinite(Z)
    m = np.where(finite, Z, -np.inf).max(axis=1, keepdims=True)
    E = np.where(finite, np.exp(Z - m), 0.0)
    return E / np.maximum(E.sum(axis=1, keepdims=True), 1e-300)


def nll(S: np.ndarray, y: list[set], T: float) -> float:
    Pm = softmax(S, T)
    mass = np.array([Pm[i, list(ls)].sum() for i, ls in enumerate(y)])
    return float(-np.mean(np.log(np.maximum(mass, 1e-12))))


def fit_temperature(S: np.ndarray, y: list[set]) -> float:
    """1-D grid search over T, scaled to the spread of the scores (log-likelihoods, cosines and
    vote counts all live on different scales)."""
    finite = np.where(np.isfinite(S), S, np.nan)
    scale = float(np.nanmedian(np.nanstd(finite, axis=1))) or 1.0
    grid = scale * np.logspace(-2.5, 1.5, 81)
    losses = [nll(S, y, T) for T in grid]
    return float(grid[int(np.argmin(losses))])


def ece(conf: np.ndarray, hit: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the top-1 probability."""
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - hit[m].mean())
    return float(e)


def confident_threshold(conf: np.ndarray, hit1: np.ndarray, target: float = 0.9, min_n: int = 10) -> float:
    order = np.argsort(-conf)
    prec = np.cumsum(hit1[order]) / np.arange(1, len(order) + 1)
    ok = np.where((prec >= target) & (np.arange(1, len(order) + 1) >= min_n))[0]
    return float(conf[order][ok[-1]]) if len(ok) else float("inf")


def abstain_threshold(conf: np.ndarray, hit3: np.ndarray, target: float = 0.5, bins: int = 10) -> float:
    """Upper edge of the contiguous run of lowest-confidence bins whose top-3 accuracy < target."""
    if len(conf) < bins * 5:
        return 0.0
    edges = np.quantile(conf, np.linspace(0, 1, bins + 1))
    thr = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi)
        if m.any() and hit3[m].mean() < target:
            thr = float(hi)
        else:
            break
    return thr

