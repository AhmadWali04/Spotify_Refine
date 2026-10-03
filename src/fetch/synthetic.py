"""Fake library for testing the loop (M2) and demoing the full product before Spotify login works.

    python -m src.fetch.synthetic   ->  data/raw/library_synthetic.json + data/raw/tags_library_synthetic.parquet

Then run:  python run_experiment.py --config configs/e0_random.yaml --library data/raw/library_synthetic.json
      or:  python -m src.sorter.run --config configs/demo_synthetic.yaml --library data/raw/library_synthetic.json

Every song has a hidden group ("synthetic_truth"): sorted songs belong to their playlists' groups,
unsorted liked songs belong either to an existing playlist's group or to one of a few brand-new
groups, which the sorter should discover as new-playlist clusters.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.paths import RAW

VIBES = [
    ("Late Night Drive", ["synthwave", "night", "electronic", "retro"]),
    ("Gym Bangers", ["hip-hop", "workout", "trap", "energetic"]),
    ("Sunday Coffee", ["acoustic", "chill", "folk", "mellow"]),
    ("Indie Sing-Alongs", ["indie", "indie rock", "anthemic", "guitar"]),
    ("Deep Focus", ["ambient", "instrumental", "minimal", "study"]),
    ("Heartbreak Hotel", ["sad", "ballad", "singer-songwriter", "piano"]),
    ("Summer BBQ", ["funk", "soul", "feel good", "groovy"]),
    ("90s Kid", ["90s", "alternative", "grunge", "nostalgia"]),
    ("House Party", ["house", "dance", "club", "electronic"]),
    ("Rainy Day Jazz", ["jazz", "smooth jazz", "saxophone", "chill"]),
    ("Road Trip Rock", ["classic rock", "rock", "70s", "guitar"]),
    ("Bedroom Pop", ["bedroom pop", "lo-fi", "dreamy", "indie pop"]),
    ("Throwback R&B", ["rnb", "2000s", "soul", "smooth"]),
    ("Metal Mondays", ["metal", "heavy", "aggressive", "thrash"]),
    ("Latin Heat", ["reggaeton", "latin", "dance", "spanish"]),
    ("K-Pop Faves", ["k-pop", "pop", "korean", "dance"]),
    ("Country Roads", ["country", "americana", "storytelling", "guitar"]),
    ("Movie Scores", ["soundtrack", "orchestral", "epic", "instrumental"]),
    ("Punk Energy", ["punk", "pop punk", "fast", "energetic"]),
    ("Lo-Fi Beats", ["lo-fi", "hip-hop", "chill", "beats"]),
    ("Afrobeats", ["afrobeats", "afro", "dance", "groovy"]),
    ("Shoegaze Haze", ["shoegaze", "dreamy", "noise", "atmospheric"]),
    ("Classical Calm", ["classical", "piano", "romantic", "instrumental"]),
    ("Disco Fever", ["disco", "70s", "dance", "funk"]),
    ("Emo Phase", ["emo", "2000s", "rock", "angst"]),
]
HIDDEN = [
    ("bossa nova", ["bossa nova", "brazilian", "mellow", "guitar"]),
    ("drum and bass", ["drum and bass", "jungle", "fast", "electronic"]),
    ("city pop", ["city pop", "japanese", "80s", "groovy"]),
]
GENERIC = ["seen live", "favorites", "pop", "rock", "electronic", "chill", "dance"]
WORDS = ["Neon", "Velvet", "Echo", "Golden", "Paper", "Silver", "Midnight", "Wild", "Ocean", "Glass",
         "Honey", "Static", "Lunar", "Cherry", "Electric", "Hollow", "Crystal", "Summer", "Broken", "Violet"]
NOUNS = ["Hearts", "Lights", "Tides", "Ghosts", "Rivers", "Machines", "Wolves", "Skies", "Dreams", "Roads",
         "Fires", "Waves", "Stars", "Gardens", "Signals", "Mirrors", "Kids", "Engines", "Shadows", "Birds"]


def make(n_songs: int = 3000, n_playlists: int = 25, seed: int = 0, n_hidden: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    n_playlists = min(n_playlists, len(VIBES))
    n_hidden = min(n_hidden, len(HIDDEN))
    ids = [f"t{i:05d}" for i in range(n_songs)]

    # Zipf-ish playlist sizes; ~60% of songs sorted, ~10% of those in two playlists.
    sizes = np.maximum(3, (400 / np.arange(1, n_playlists + 1) ** 0.8).astype(int))
    pool = rng.permutation(n_songs)[: int(0.6 * n_songs)]
    playlists, truth = [], {t: set() for t in ids}
    start = 0
    for p, s in enumerate(sizes):
        members = list(pool[start:start + s]) if start < len(pool) else []
        start += s
        extra = rng.choice(pool, size=max(1, s // 10), replace=False)
        members = list(dict.fromkeys(members + list(extra)))
        playlists.append({"id": f"p{p}", "name": VIBES[p][0], "owner": "me", "collaborative": False,
                          "track_ids": [ids[i] for i in members]})
        for i in members:
            truth[ids[i]].add(f"g{p}")

    # Unsorted songs: 60% belong with an existing playlist, 40% to a brand-new group.
    weights = sizes / sizes.sum()
    for t in ids:
        if truth[t]:
            continue
        if n_hidden and rng.random() < 0.4:
            truth[t].add(f"h{rng.integers(n_hidden)}")
        else:
            truth[t].add(f"g{rng.choice(n_playlists, p=weights)}")

    # Each group has its own artists, so names carry some signal in the demo UI too.
    artists = {g: [f"{rng.choice(WORDS)} {rng.choice(NOUNS)}" for _ in range(12)]
               for g in [f"g{p}" for p in range(n_playlists)] + [f"h{h}" for h in range(n_hidden)]}
    tracks = {}
    for t in ids:
        g = sorted(truth[t])[0]
        tracks[t] = {"name": f"{rng.choice(WORDS)} {rng.choice(NOUNS)}", "artists": [str(rng.choice(artists[g]))],
                     "album": f"{rng.choice(WORDS)} {rng.choice(NOUNS)}", "duration_ms": int(rng.integers(150, 300)) * 1000,
                     "release_date": f"{rng.integers(1970, 2026)}", "isrc": None}
    return {"pulled_at": "synthetic", "user_id": "me", "tracks": tracks,
            "liked": [{"id": t, "added_at": None} for t in ids], "playlists": playlists,
            "synthetic_truth": {t: sorted(g) for t, g in truth.items()}}


def make_tags(lib: dict, seed: int = 0) -> pd.DataFrame:
    """Last.fm-shaped tags: each song gets 2-4 of its group's tags plus some generic noise."""
    rng = np.random.default_rng(seed)
    vocab = {f"g{p}": v[1] for p, v in enumerate(VIBES)} | {f"h{h}": v[1] for h, v in enumerate(HIDDEN)}
    rows = []
    for t, groups in lib["synthetic_truth"].items():
        for g in groups:
            for tag in rng.choice(vocab[g], size=int(rng.integers(2, 5)), replace=False):
                rows.append({"track_id": t, "tag": str(tag), "weight": int(rng.integers(30, 100)), "source": "track"})
        for tag in rng.choice(GENERIC, size=2, replace=False):
            rows.append({"track_id": t, "tag": str(tag), "weight": int(rng.integers(5, 60)), "source": "track"})
    df = pd.DataFrame(rows)
    return df.groupby(["track_id", "tag", "source"], as_index=False)["weight"].max()


if __name__ == "__main__":
    lib = make()
    out = RAW / "library_synthetic.json"
    out.write_text(json.dumps(lib, indent=1))
    tags_out = RAW / "tags_library_synthetic.parquet"
    make_tags(lib).to_parquet(tags_out, index=False)
    print(f"Saved {out} and {tags_out}")
