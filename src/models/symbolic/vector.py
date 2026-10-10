"""Vector mode (PRD3 §6.2): flatten the symbolic blocks and hand them to a Phase 2 assignment model.

  1. Flatten each block (transition matrices row-normalized first; see represent.feature_blocks).
  2. Z-score each column with training statistics, scale each block by 1/sqrt(block size).
  3. Optional PCA to pca_k components (chord_trans alone is 625 dims).
  4. Fit the assignment model: A0 centroid, A1 LDA, A2 Mahalanobis, A3 kNN-10, A4 label spreading
     (or any name in src.sorter.models.MODELS).

With transpose on, every training song is also added in the 11 other keys (key-relative blocks
shifted), which makes the model robust to key-detection mistakes.
"""
from __future__ import annotations

import numpy as np

from src.midi.represent import BLOCKS, SongRep, feature_blocks
from src.models.symbolic.base import class_index
from src.sorter import models as assign
from src.sorter.calibrate import softmax

ASSIGNMENT = {"A0": "centroid", "A1": "lda", "A2": "mahalanobis", "A3": "knn10", "A4": "label_spreading"}


class VectorModel:
    name = "vector"

    def __init__(self, assignment_model: str = "A2", blocks=BLOCKS, transpose: bool = False,
                 pca_k: int | None = None):
        self.assignment = ASSIGNMENT.get(assignment_model, assignment_model)
        unknown = set(blocks) - set(BLOCKS)
        if unknown:
            raise ValueError(f"unknown blocks {sorted(unknown)}; choose from {BLOCKS}")
        self.blocks = list(blocks)
        self.transpose = transpose
        self.pca_k = pca_k

    def raw(self, songs: list[SongRep], shift: int = 0) -> np.ndarray:
        rows = [feature_blocks(s, shift) for s in songs]
        return np.array([np.concatenate([r[b] for b in self.blocks]) for r in rows])

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd * self.block_scale
        return self.pca(Z) if self.pca else Z

    def fit(self, songs: list[SongRep], labels: list[set[str]]) -> "VectorModel":
        self.classes_, y = class_index(labels)
        X = self.raw(songs)
        if self.transpose:
            X = np.vstack([X] + [self.raw(songs, k) for k in range(1, 12)])
            y = y * 12
        sizes = {b: len(v) for b, v in feature_blocks(songs[0]).items()}
        self.block_scale = np.concatenate([np.full(sizes[b], 1 / np.sqrt(sizes[b])) for b in self.blocks])
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd < 1e-8, 1.0, sd)
        self.pca = None
        Z = self._transform(X)
        if self.pca_k:
            self.pca = assign._PCA(self.pca_k).fit(Z)
            Z = self.pca(Z)
        self.model = assign.make(self.assignment).fit(Z, y, len(self.classes_))
        return self

    def scores(self, songs: list[SongRep]) -> np.ndarray:
        return self.model.scores(self._transform(self.raw(songs)))

    def predict_proba(self, songs: list[SongRep]) -> np.ndarray:
        return softmax(self.scores(songs), 1.0)
