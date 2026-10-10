"""Markov mode (PRD3 §6.1): one smoothed transition matrix per playlist and sequence type.

    P_{p,s}(j | i) = (n_{p,s}(i -> j) + alpha) / (sum_k n_{p,s}(i -> k) + K_s alpha)
    l_p(song)      = sum_s w_s * mean_t log P_{p,s}(x_{t+1} | x_t)  +  log pi_p

Sequence types: chords (25 key-relative chords per beat), pitch_classes (12, melody onsets) and
drums (8 kick/snare/hat states per 16th). Order 2 uses the previous two symbols as the context.
A song with no transitions of some type (e.g. no drums) is scored on the types it has.
"""
from __future__ import annotations

import numpy as np

from src.midi import chords
from src.midi.represent import ALPHABETS, DRUM_ROWS, SongRep, transition_counts
from src.models.symbolic.base import class_index
from src.sorter.calibrate import softmax

DEGREE_NAMES = ["1", "b2", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7"]


def symbol_name(seq_type: str, x: int) -> str:
    if seq_type == "chords":
        return chords.roman(x)
    if seq_type == "pitch_classes":
        return DEGREE_NAMES[x]
    return "+".join(n for b, n in enumerate(DRUM_ROWS) if x >> b & 1) or "rest"


class MarkovModel:
    name = "markov"

    def __init__(self, sequences=("chords",), order: int = 1, alpha: float = 0.5,
                 weights: dict | None = None, prior: str = "size"):
        self.sequences = list(sequences)
        self.order = order
        self.alpha = alpha
        self.weights = {s: (weights or {}).get(s, 1.0) for s in self.sequences}
        self.prior = prior

    def _counts(self, song: SongRep, s: str) -> np.ndarray:
        return transition_counts(song.sequences[s], ALPHABETS[s], self.order)

    def _log_probs(self, C: np.ndarray, K: int) -> np.ndarray:
        return np.log((C + self.alpha) / (C.sum(axis=-1, keepdims=True) + K * self.alpha))

    def fit(self, songs: list[SongRep], labels: list[set[str]]) -> "MarkovModel":
        self.classes_, y = class_index(labels)
        P = len(self.classes_)
        self.logP, self.logBG = {}, {}
        for s in self.sequences:
            K = ALPHABETS[s]
            C = np.zeros((P, K ** self.order, K))
            for song, ls in zip(songs, y):
                c = self._counts(song, s)
                for p in ls:
                    C[p] += c
            self.logP[s] = self._log_probs(C, K)
            self.logBG[s] = self._log_probs(sum(self._counts(song, s) for song in songs), K)
        sizes = np.bincount([p for ls in y for p in ls], minlength=P).astype(float)
        self.log_prior = np.log(sizes / sizes.sum()) if self.prior == "size" else np.full(P, -np.log(P))
        return self

    def _transitions(self, song: SongRep, s: str) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(song.sequences[s], dtype=int)
        n = len(x) - self.order
        if n <= 0:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        ctx = np.zeros(n, dtype=int)
        for j in range(self.order):
            ctx = ctx * ALPHABETS[s] + x[j: j + n]
        return ctx, x[self.order:]

    def scores(self, songs: list[SongRep]) -> np.ndarray:
        S = np.tile(self.log_prior, (len(songs), 1))
        for i, song in enumerate(songs):
            for s in self.sequences:
                ctx, nxt = self._transitions(song, s)
                if len(ctx):
                    S[i] += self.weights[s] * self.logP[s][:, ctx, nxt].mean(axis=1)
        return S

    def predict_proba(self, songs: list[SongRep]) -> np.ndarray:
        return softmax(self.scores(songs), 1.0)

    def explain(self, song: SongRep, playlist: str, k: int = 5) -> list[dict]:
        """The k transitions in this song most characteristic of the playlist:
        largest log P_playlist - log P_background (ratio > 1 means more common in the playlist)."""
        p = self.classes_.index(playlist)
        found = {}
        for s in self.sequences:
            ctx, nxt = self._transitions(song, s)
            K = ALPHABETS[s]
            for c, n in zip(ctx, nxt):
                key = (s, int(c), int(n))
                if key not in found:
                    found[key] = {"sequence": s, "count": 0,
                                  "from": " ".join(symbol_name(s, (c // K ** j) % K) for j in reversed(range(self.order))),
                                  "to": symbol_name(s, n),
                                  "log_ratio": float(self.logP[s][p, c, n] - self.logBG[s][c, n])}
                found[key]["count"] += 1
        top = sorted(found.values(), key=lambda r: -r["log_ratio"])[:k]
        for r in top:
            r["ratio"] = float(np.exp(r["log_ratio"]))
        return top
