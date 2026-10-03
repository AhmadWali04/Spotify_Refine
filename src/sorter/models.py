"""Assignment models. Each one maps song vectors to a score per playlist (higher = better fit).

    model.fit(X, y, P)   # y[i] is the set of playlist indices song i is in (multi-label)
    model.scores(X)      # -> (n, P) array; -inf means "cannot be this playlist"

All written by hand in NumPy (SciPy only for the sparse graph), like src/linalg.py:
  centroid         cosine to the playlist mean (the Phase 1 scorer)
  lda              shared covariance, linear boundaries, playlist-size priors
  mahalanobis      per-playlist covariance (shrunk toward the shared one): tight vs. broad playlists
  knn              votes of the k most similar sorted songs
  label_spreading  diffuses labels over a kNN similarity graph that includes the unsorted songs
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.linalg import l2_normalize


def _onehot(y: list[set], P: int) -> np.ndarray:
    Y = np.zeros((len(y), P))
    for i, ls in enumerate(y):
        Y[i, list(ls)] = 1
    return Y


def _expand(X: np.ndarray, y: list[set]):
    """Multi-label -> one row per (song, playlist) pair, for the generative models."""
    rows = [(i, p) for i, ls in enumerate(y) for p in ls]
    idx = np.array([r[0] for r in rows])
    lab = np.array([r[1] for r in rows])
    return X[idx], lab


class _PCA:
    """Project onto the top-k principal directions of the training songs (fit once, reuse on test)."""

    def __init__(self, k: int):
        self.k = k

    def fit(self, X):
        self.mean = X.mean(axis=0)
        if X.shape[1] <= self.k:
            self.V = None
            return self
        _, _, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        self.V = Vt[: self.k].T
        return self

    def __call__(self, X):
        Xc = X - self.mean
        return Xc if self.V is None else Xc @ self.V


class Centroid:
    name = "centroid"

    def fit(self, X, y, P):
        Y = _onehot(y, P)
        counts = Y.sum(axis=0)
        self.mu = (Y.T @ X) / np.maximum(counts, 1)[:, None]
        self.empty = counts == 0
        return self

    def scores(self, X):
        S = l2_normalize(X) @ l2_normalize(self.mu).T
        S[:, self.empty] = -np.inf
        return S


class KNN:
    def __init__(self, k: int = 10):
        self.k = k
        self.name = f"knn{k}"

    def fit(self, X, y, P):
        self.Xn, self.Y = l2_normalize(X), _onehot(y, P)
        return self

    def scores(self, X):
        sim = l2_normalize(X) @ self.Xn.T
        k = min(self.k, sim.shape[1])
        nn = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        w = np.take_along_axis(sim, nn, axis=1)
        # Votes, with similarity as a small tie-breaker (same rule as the Phase 1 scorer).
        S = np.einsum("nk,nkp->np", 1 + 1e-3 * w, self.Y[nn])
        return S


def _shrink(S: np.ndarray, a: float) -> np.ndarray:
    """(1-a) S + a (tr(S)/d) I: pulls a noisy covariance toward a sphere so it stays invertible."""
    d = S.shape[0]
    return (1 - a) * S + a * (np.trace(S) / d) * np.eye(d)


class LDA:
    def __init__(self, shrinkage: float = 0.3, max_dims: int = 100):
        self.a, self.pca = shrinkage, _PCA(max_dims)
        self.name = f"lda-s{shrinkage}"

    def fit(self, X, y, P):
        Z = self.pca.fit(X)(X)
        Ze, lab = _expand(Z, y)
        counts = np.bincount(lab, minlength=P).astype(float)
        mu = np.zeros((P, Z.shape[1]))
        np.add.at(mu, lab, Ze)
        mu /= np.maximum(counts, 1)[:, None]
        R = Ze - mu[lab]
        Sigma = _shrink(R.T @ R / max(len(Ze) - P, 1), self.a)
        self.W = np.linalg.solve(Sigma, mu.T)                       # Sigma^-1 mu_p, d x P
        prior = np.log(np.maximum(counts, 1) / counts.sum())
        self.b = -0.5 * np.sum(mu.T * self.W, axis=0) + prior
        self.b[counts == 0] = -np.inf
        return self

    def scores(self, X):
        return self.pca(X) @ self.W + self.b


class Mahalanobis:
    def __init__(self, beta: float = 0.5, shrinkage: float = 0.3, max_dims: int = 32):
        self.beta, self.a, self.pca = beta, shrinkage, _PCA(max_dims)
        self.name = f"mahalanobis-b{beta}"

    def fit(self, X, y, P):
        Z = self.pca.fit(X)(X)
        Ze, lab = _expand(Z, y)
        d = Z.shape[1]
        self.mu = np.zeros((P, d))
        covs, counts = [], np.bincount(lab, minlength=P)
        for p in range(P):
            Zp = Ze[lab == p]
            if len(Zp):
                self.mu[p] = Zp.mean(axis=0)
            Rp = Zp - self.mu[p]
            covs.append(Rp.T @ Rp / max(len(Zp) - 1, 1))
        pooled = _shrink(sum(c * n for c, n in zip(covs, counts)) / max(counts.sum(), 1), self.a)
        self.L, self.logdet = [], np.zeros(P)
        for p in range(P):
            Sp = (1 - self.beta) * covs[p] + self.beta * pooled if counts[p] > d // 4 else pooled
            L = np.linalg.cholesky(Sp + 1e-9 * np.eye(d))
            self.L.append(L)
            self.logdet[p] = 2 * np.log(np.diag(L)).sum()
        self.empty = counts == 0
        return self

    def scores(self, X):
        Z = self.pca(X)
        S = np.empty((len(Z), len(self.L)))
        for p, L in enumerate(self.L):
            R = np.linalg.solve(L, (Z - self.mu[p]).T)              # whitened residual
            S[:, p] = -0.5 * (np.sum(R * R, axis=0) + self.logdet[p])  # Gaussian log-likelihood
        S[:, self.empty] = -np.inf
        return S


class LabelSpreading:
    """Zhou et al. (2004): F <- alpha S F + (1 - alpha) Y on a symmetric-normalized kNN graph.

    Transductive: the test songs (and any extra unlabeled songs) are nodes in the graph, so
    clusters of unsorted songs can pull labels along chains of similar songs.
    """

    def __init__(self, k: int = 10, alpha: float = 0.9, iters: int = 40):
        self.k, self.alpha, self.iters = k, alpha, iters
        self.name = f"label_spreading-k{k}"
        self.extra = None             # optional (m, d) unlabeled songs added to the graph

    def fit(self, X, y, P):
        self.Xn, self.Y = l2_normalize(X), _onehot(y, P)
        self.Y /= np.maximum(self.Y.sum(axis=1, keepdims=True), 1)
        return self

    def _graph(self, Z: np.ndarray, chunk: int = 2048) -> sp.csr_matrix:
        n, k = len(Z), min(self.k, len(Z) - 1)
        rows, cols, vals = [], [], []
        for s in range(0, n, chunk):
            sim = Z[s:s + chunk] @ Z.T
            sim[np.arange(len(sim)), np.arange(s, s + len(sim))] = -np.inf   # no self-loops
            nn = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            rows.append(np.repeat(np.arange(s, s + len(sim)), k))
            cols.append(nn.ravel())
            vals.append(np.maximum(np.take_along_axis(sim, nn, axis=1).ravel(), 0))
        W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        W = W.maximum(W.T)
        d = np.asarray(W.sum(axis=1)).ravel()
        Dm = sp.diags(1 / np.sqrt(np.maximum(d, 1e-12)))
        return Dm @ W @ Dm

    def scores(self, X):
        parts = [self.Xn, l2_normalize(X)] + ([l2_normalize(self.extra)] if self.extra is not None else [])
        Z = np.vstack(parts)
        S = self._graph(Z)
        Y0 = np.zeros((len(Z), self.Y.shape[1]))
        Y0[: len(self.Xn)] = self.Y
        F = Y0.copy()
        for _ in range(self.iters):
            F = self.alpha * (S @ F) + (1 - self.alpha) * Y0
        out = F[len(self.Xn): len(self.Xn) + len(X)]
        return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


# Ordered by preference: the comparison picks the first model within one std of the best.
# Neighbour-vote models (kNN, label spreading) come last on purpose. They lean toward big
# playlists, and the unsorted backlog is rarely spread over playlists in the same proportions
# as your sorted songs, so a small playlist with a big backlog gets swamped. Cross-validation
# on sorted songs can't see that (on the synthetic demo kNN-10 scores 97% in CV but 78% on the
# backlog, while centroid and LDA hold at 93%), so they only win when clearly better.
MODELS = {
    "centroid": Centroid,
    "lda": lambda: LDA(0.3),
    "mahalanobis": lambda: Mahalanobis(0.5),
    "knn10": lambda: KNN(10),
    "knn25": lambda: KNN(25),
    "label_spreading": lambda: LabelSpreading(10, 0.9),
}


def make(name: str):
    return MODELS[name]()
