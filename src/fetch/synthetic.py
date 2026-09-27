"""Fake library for testing the loop (M2) before Spotify login works.

    python -m src.fetch.synthetic   ->  data/raw/library_synthetic.json

Then run:  python run_experiment.py --config configs/e0_random.yaml --library data/raw/library_synthetic.json
"""
from __future__ import annotations

import json

import numpy as np

from src.paths import RAW


def make(n_songs: int = 3000, n_playlists: int = 25, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ids = [f"t{i:05d}" for i in range(n_songs)]
    tracks = {t: {"name": f"Song {t}", "artists": [f"Artist {rng.integers(400)}"],
                  "album": None, "release_date": None, "duration_ms": 200000, "isrc": None}
              for t in ids}
    # Zipf-ish playlist sizes; ~60% of songs sorted, ~10% of those in two playlists.
    sizes = np.maximum(3, (400 / np.arange(1, n_playlists + 1) ** 0.8).astype(int))
    pool = rng.permutation(n_songs)[: int(0.6 * n_songs)]
    playlists = []
    start = 0
    for p, s in enumerate(sizes):
        members = list(pool[start:start + s]) if start < len(pool) else []
        start += s
        extra = rng.choice(pool, size=max(1, s // 10), replace=False)
        members = list(dict.fromkeys(members + list(extra)))
        playlists.append({"id": f"p{p}", "name": f"Playlist {p}", "owner": "me",
                          "track_ids": [ids[i] for i in members]})
    return {"pulled_at": "synthetic", "user_id": "me", "tracks": tracks,
            "liked": [{"id": t, "added_at": None} for t in ids], "playlists": playlists}


if __name__ == "__main__":
    out = RAW / "library_synthetic.json"
    out.write_text(json.dumps(make(), indent=1))
    print(f"Saved {out}")
