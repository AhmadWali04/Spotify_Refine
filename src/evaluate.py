"""Folds, centroids and metrics. Identical for every run so vectors are compared fairly.

Scorers (per PRD):
  - nearest centroid, cosine similarity (primary)
  - kNN, k = 10, cosine similarity (second view)
A song in several playlists counts as correct if any of its playlists is predicted.
"""
from __future__ import annotations

import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, silhouette_score

from src.linalg import l2_normalize

N_FOLDS = 5
SEED = 42
MIN_PLAYLIST_SIZE = 10
KNN_K = 10


# ---------------------------------------------------------------- labeled set + folds

def labeled_set(lib: dict, min_size: int = MIN_PLAYLIST_SIZE):
    """Returns (playlist_names, labels) where labels maps track_id -> set of playlist indices.

    Only playlists with >= min_size songs are scored. Duplicate names are disambiguated.
    """
    kept = [p for p in lib["playlists"] if len(p["track_ids"]) >= min_size]
    names, seen = [], {}
    for p in kept:
        n = p["name"]
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    labels: dict[str, set[int]] = {}
    for j, p in enumerate(kept):
        for t in p["track_ids"]:
            labels.setdefault(t, set()).add(j)
    return names, labels


def load_or_make_folds(lib: dict, labels: dict, path: Path) -> dict[str, int]:
    """Fold id per labeled track, fixed once per library pull and reused by every run."""
    ids_hash = hashlib.sha1(" ".join(sorted(labels)).encode()).hexdigest()[:12]
    key = {"pulled_at": lib.get("pulled_at"), "seed": SEED, "n_folds": N_FOLDS,
           "min_size": MIN_PLAYLIST_SIZE, "n_labeled": len(labels), "ids": ids_hash}
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["key"] == key:
            return saved["fold"]
        print(f"Library or fold settings changed; regenerating {path.name}")
    ids = sorted(labels)
    perm = np.random.default_rng(SEED).permutation(len(ids))
    fold = {ids[i]: int(r % N_FOLDS) for r, i in enumerate(perm)}
    path.write_text(json.dumps({"key": key, "fold": fold}))
    return fold


# ---------------------------------------------------------------- scorers

def centroid_scores(X_tr, y_tr, X_te, n_playlists: int) -> np.ndarray:
    """Cosine similarity of each test song to each playlist centroid mu_p (train songs only).

    y_tr is a list of label sets; a multi-playlist song contributes to every centroid it is in.
    Playlists with no training songs get -inf.
    """
    d = X_tr.shape[1]
    sums = np.zeros((n_playlists, d))
    counts = np.zeros(n_playlists)
    for x, ls in zip(X_tr, y_tr):
        for p in ls:
            sums[p] += x
            counts[p] += 1
    mu = sums / np.maximum(counts, 1)[:, None]
    S = l2_normalize(X_te) @ l2_normalize(mu).T
    S[:, counts == 0] = -np.inf
    return S


def knn_scores(X_tr, y_tr, X_te, n_playlists: int, k: int = KNN_K) -> np.ndarray:
    """Vote count among the k most cosine-similar training songs; similarity sum breaks ties."""
    sim = l2_normalize(X_te) @ l2_normalize(X_tr).T
    k = min(k, X_tr.shape[0])
    nn = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    S = np.zeros((X_te.shape[0], n_playlists))
    for i, row in enumerate(nn):
        for j in row:
            for p in y_tr[j]:
                S[i, p] += 1 + 1e-3 * sim[i, j]
    return S


def topk_hits(S: np.ndarray, y_te: list[set], k: int) -> np.ndarray:
    top = np.argsort(-S, axis=1)[:, :k]
    return np.array([bool(set(row) & ls) for row, ls in zip(top, y_te)])


def macro_f1(S: np.ndarray, y_te: list[set], sizes: np.ndarray) -> float:
    """Top-1 macro-F1. For multi-playlist songs the 'true' label is the prediction if it is
    one of theirs, otherwise their smallest playlist (the most specific one)."""
    pred = S.argmax(axis=1)
    true = [p if p in ls else min(ls, key=lambda q: sizes[q]) for p, ls in zip(pred, y_te)]
    return float(f1_score(true, pred, average="macro", zero_division=0))


def chance_rates(y: list[set], n_playlists: int) -> tuple[float, float]:
    """Expected top-1 / top-3 accuracy when playlists are ranked uniformly at random."""
    P = n_playlists
    t1 = np.mean([len(ls) / P for ls in y])
    t3 = np.mean([1 - comb(P - len(ls), min(3, P)) / comb(P, min(3, P)) for ls in y])
    return float(t1), float(t3)


# ---------------------------------------------------------------- main entry

def evaluate(ids: list[str], X: np.ndarray, lib: dict, folds_path: Path) -> dict:
    names, labels = labeled_set(lib)
    P = len(names)
    if P < 2:
        raise ValueError(f"Need at least 2 playlists with >= {MIN_PLAYLIST_SIZE} songs, found {P}.")
    fold = load_or_make_folds(lib, labels, folds_path)

    row_of = {t: i for i, t in enumerate(ids)}
    ev = [t for t in sorted(labels) if t in row_of]      # labeled AND vectorized
    Xe = X[[row_of[t] for t in ev]]
    ye = [labels[t] for t in ev]
    fe = np.array([fold[t] for t in ev])
    sizes = np.zeros(P)
    for ls in labels.values():
        for p in ls:
            sizes[p] += 1

    per_fold = {m: [] for m in ("top1", "top3", "knn_top1", "knn_top3", "macro_f1")}
    for f in range(N_FOLDS):
        tr, te = fe != f, fe == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        y_tr = [ye[i] for i in np.where(tr)[0]]
        y_te = [ye[i] for i in np.where(te)[0]]
        Sc = centroid_scores(Xe[tr], y_tr, Xe[te], P)
        Sk = knn_scores(Xe[tr], y_tr, Xe[te], P)
        per_fold["top1"].append(topk_hits(Sc, y_te, 1).mean())
        per_fold["top3"].append(topk_hits(Sc, y_te, 3).mean())
        per_fold["knn_top1"].append(topk_hits(Sk, y_te, 1).mean())
        per_fold["knn_top3"].append(topk_hits(Sk, y_te, 3).mean())
        per_fold["macro_f1"].append(macro_f1(Sc, y_te, sizes))

    # Silhouette by (primary = smallest) playlist label, cosine distance, subsampled for speed.
    primary = np.array([min(ls, key=lambda q: sizes[q]) for ls in ye])
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(ev), size=min(4000, len(ev)), replace=False)
    sil = (float(silhouette_score(Xe[sub], primary[sub], metric="cosine"))
           if len(set(primary[sub])) > 1 else float("nan"))

    liked = [x["id"] for x in lib["liked"]]
    chance1, chance3 = chance_rates(ye, P)
    out = {
        "n_playlists": P,
        "n_labeled": len(labels),
        "n_eval": len(ev),
        "coverage_liked": 100 * np.mean([t in row_of for t in liked]) if liked else float("nan"),
        "coverage_labeled": 100 * len(ev) / len(labels),
        "chance_top1": 100 * chance1,
        "chance_top3": 100 * chance3,
        # Always guessing the biggest playlist; kNN vote counts drift toward this on weak vectors.
        "majority_top1": 100 * float(np.mean([int(sizes.argmax()) in ls for ls in ye])),
        "silhouette": sil,
        "per_fold": {k: [float(v) for v in vs] for k, vs in per_fold.items()},
        "eval_ids": ev,
        "primary_label": primary,
        "playlist_names": names,
    }
    for k, vs in per_fold.items():
        scale = 1 if k == "macro_f1" else 100
        out[f"{k}_mean"] = scale * float(np.mean(vs))
        out[f"{k}_std"] = scale * float(np.std(vs))
    return out
