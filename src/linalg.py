"""Hand-written linear algebra used across featurizers and evaluation.

scikit-learn is only used in tests to check these against a reference.
"""
from __future__ import annotations

import numpy as np


def zscore(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Center each column and scale to unit variance. Constant columns become 0."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return (X - mu) / np.where(sd < eps, 1.0, sd)


def block_weight(X: np.ndarray) -> np.ndarray:
    """Scale a z-scored block by 1/sqrt(d) so each block contributes equal total variance."""
    return X / np.sqrt(X.shape[1])


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via SVD of the centered data.

    Returns (scores n x k, components k x d, explained_variance_ratio of length k).
    Component signs are fixed so the largest-|value| entry of each is positive,
    making results deterministic.
    """
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0])
    signs = np.sign(Vt[np.arange(Vt.shape[0]), np.abs(Vt).argmax(axis=1)])
    signs[signs == 0] = 1
    U, Vt = U * signs, Vt * signs[:, None]
    var = S ** 2
    return U[:, :k] * S[:k], Vt[:k], var[:k] / var.sum()


def truncated_svd(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LSA: X ~ U_k S_k V_k^T without centering. Returns (U_k S_k, V_k^T, singular values)."""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(k, len(S))
    return U[:, :k] * S[:k], Vt[:k], S[:k]
