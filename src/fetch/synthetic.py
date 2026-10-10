"""Fake library for testing the loop (M2) and demoing the full product before Spotify login works.

    python -m src.fetch.synthetic   ->  data/raw/library_synthetic.json, tags_library_synthetic.parquet
                                        and history_library_synthetic.parquet (listening history)

Then run:  python run_experiment.py --config configs/e0_random.yaml --library data/raw/library_synthetic.json
      or:  python -m src.sorter.run --config configs/demo_synthetic.yaml --library data/raw/library_synthetic.json

Every song has a hidden group ("synthetic_truth"): sorted songs belong to their playlists' groups,
unsorted liked songs belong either to an existing playlist's group or to one of a few brand-new
groups, which the sorter should discover as new-playlist clusters.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

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
            "liked": _liked_times(ids, seed), "playlists": playlists,
            "synthetic_truth": {t: sorted(g) for t, g in truth.items()}}


def _liked_times(ids: list[str], seed: int) -> list[dict]:
    """Plausible "liked at" times for the gallery: five years, busier evenings and weekends.
    Own generator, so the songs and playlists above stay identical for a given seed."""
    rng = np.random.default_rng(seed + 1)
    end = datetime(2026, 9, 30, tzinfo=timezone.utc)
    days = (end - datetime(2021, 10, 1, tzinfo=timezone.utc)).days
    hours = np.array([1, .6, .4, .3, .2, .3, .6, 1.2, 2, 2, 1.8, 2, 2.6, 2.4, 2, 2, 2.4, 3, 3.6, 4.2, 4.6, 4.4, 3.4, 2])
    out = []
    for t in ids:
        d = end - timedelta(days=int(days * rng.random() ** 0.7))        # skew toward recent
        if d.weekday() < 5 and rng.random() < 0.3:
            d += timedelta(days=5 - d.weekday())                        # nudge some to the weekend
        d = d.replace(hour=int(rng.choice(24, p=hours / hours.sum())), minute=int(rng.integers(60)))
        out.append({"id": t, "added_at": min(d, end).strftime("%Y-%m-%dT%H:%M:%SZ")})
    return sorted(out, key=lambda x: x["added_at"], reverse=True)       # newest first, like Spotify


def make_history(lib: dict, seed: int = 0, start: str = "2022-01-01", end: str = "2026-09-30") -> pd.DataFrame:
    """Streaming-history-shaped plays for the listening charts. Each hidden group has its own era:
    a slow random swell over the years, so two periods really do sound different."""
    rng = np.random.default_rng(seed + 2)
    days = pd.date_range(start, end, freq="D", tz="UTC")
    groups = sorted({g for gs in lib["synthetic_truth"].values() for g in gs})
    by_group: dict[str, list[str]] = {g: [] for g in groups}
    for t, gs in lib["synthetic_truth"].items():
        by_group[gs[0]].append(t)
    # Group weight over time: base size x a swell with a random period (1.5-4 years) and phase.
    x = np.arange(len(days))[:, None] / 365.0
    base = np.array([len(by_group[g]) for g in groups], float) ** 0.8
    period, phase = rng.uniform(1.5, 4, len(groups)), rng.uniform(0, 2 * np.pi, len(groups))
    w = base * np.exp(1.6 * np.sin(2 * np.pi * x / period + phase))
    w /= w.sum(axis=1, keepdims=True)
    # Plays per day: more on weekends and in winter, a slow upward trend, and some quiet days.
    dow = np.where(days.dayofweek >= 5, 1.35, 1.0)
    season = 1 + 0.2 * np.cos(2 * np.pi * (days.dayofyear - 15) / 365)
    lam = 22 * dow * season * (0.7 + 0.5 * x[:, 0] / x[-1, 0]) * (rng.random(len(days)) > 0.06)
    counts = rng.poisson(lam)
    hours = np.array([.6, .3, .2, .1, .1, .2, .6, 1.4, 2, 1.8, 1.6, 1.8, 2.2, 2, 1.8, 1.9, 2.2, 2.8, 3, 3.2, 3.4, 3.2, 2.4, 1.4])
    rows_day, rows_group = [], []
    for d, c in enumerate(counts):
        if c:
            rows_day.append(np.full(c, d))
            rows_group.append(rng.choice(len(groups), size=c, p=w[d]))
    day_i, grp_i = np.concatenate(rows_day), np.concatenate(rows_group)
    track_ids = np.empty(len(day_i), dtype=object)
    for g, name in enumerate(groups):
        idx = np.flatnonzero(grp_i == g)
        pool = by_group[name]
        pop = 1 / np.arange(1, len(pool) + 1) ** 0.9                 # a few favourites get most plays
        track_ids[idx] = np.array(pool, dtype=object)[rng.choice(len(pool), size=len(idx), p=pop / pop.sum())]
    secs = rng.choice(24, size=len(day_i), p=hours / hours.sum()) * 3600 + rng.integers(0, 3600, len(day_i))
    ts = days[day_i] + pd.to_timedelta(secs, unit="s")
    tr = lib["tracks"]
    dur = np.array([tr[t]["duration_ms"] for t in track_ids])
    skipped = rng.random(len(dur)) < 0.22
    ms = np.where(skipped, (dur * rng.uniform(0.02, 0.5, len(dur))).astype(int), dur)
    df = pd.DataFrame({"ts": ts, "ms": ms, "track": [tr[t]["name"] for t in track_ids],
                       "artist": [tr[t]["artists"][0] for t in track_ids],
                       "album": [tr[t]["album"] for t in track_ids], "track_id": track_ids})
    for c in ("track", "artist", "album", "track_id"):
        df[c] = df[c].astype("string")
    return df.sort_values("ts").reset_index(drop=True)


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
    hist_out = RAW / "history_library_synthetic.parquet"
    make_history(lib).to_parquet(hist_out, index=False)
    print(f"Saved {out}, {tags_out} and {hist_out}")
