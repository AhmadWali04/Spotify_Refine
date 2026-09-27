"""E1 (tag TF-IDF) and E1b (tags + LSA), built from data/raw/tags.parquet."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.linalg import l2_normalize, truncated_svd
from src.paths import TAGS

# Tags that describe the listener, not the song.
STOP_TAGS = {
    "seen live", "favorites", "favourite", "favorite", "favourites", "my favorite",
    "love", "loved", "awesome", "beautiful", "amazing", "good", "best", "cool",
    "spotify", "albums i own", "under 2000 listeners", "check out", "fav", "favs",
}


def tag_matrix(track_ids: list[str], params: dict):
    """Song x tag weight matrix. Returns (covered_ids, M, vocab)."""
    if not TAGS.exists():
        raise FileNotFoundError(f"{TAGS} not found. Run `python -m src.fetch.lastfm` first.")
    df = pd.read_parquet(TAGS)
    df = df[df["tag"].notna() & df["track_id"].isin(set(track_ids))]
    if not params.get("use_artist_fallback", True):
        df = df[df["source"] == "track"]
    df = df[~df["tag"].isin(STOP_TAGS) & (df["weight"] > params.get("min_weight", 0))]
    df = df[df["tag"].str.len() > 1]

    # Vocabulary: tags on at least min_df songs, capped at max_features by document frequency.
    df_count = df.groupby("tag")["track_id"].nunique()
    df_count = df_count[df_count >= params.get("min_df", 5)]
    vocab = df_count.sort_values(ascending=False).index[: params.get("max_features", 2000)]
    df = df[df["tag"].isin(vocab)]

    covered = [t for t in track_ids if t in set(df["track_id"])]
    row = {t: i for i, t in enumerate(covered)}
    col = {g: j for j, g in enumerate(vocab)}
    M = np.zeros((len(covered), len(vocab)), dtype=np.float32)
    M[df["track_id"].map(row).to_numpy(), df["tag"].map(col).to_numpy()] = df["weight"].to_numpy() / 100.0
    return covered, M, list(vocab)


def tfidf(M: np.ndarray) -> np.ndarray:
    """Weighted tag counts -> TF-IDF with smooth idf, rows L2-normalized."""
    n = M.shape[0]
    df = (M > 0).sum(axis=0)
    idf = np.log((1 + n) / (1 + df)) + 1
    return l2_normalize(M * idf).astype(np.float32)


def featurize_tfidf(lib: dict, track_ids: list[str], params: dict):
    covered, M, _ = tag_matrix(track_ids, params)
    return covered, tfidf(M)


def featurize_lsa(lib: dict, track_ids: list[str], params: dict):
    covered, M, _ = tag_matrix(track_ids, params)
    Z, _, S = truncated_svd(tfidf(M), params.get("k", 100))
    return covered, Z.astype(np.float32)
