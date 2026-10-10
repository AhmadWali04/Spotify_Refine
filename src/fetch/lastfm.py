"""Crowd tags from Last.fm for E1 / E1b. Resumable: already-fetched tracks are skipped.

    python -m src.fetch.lastfm            # all tracks in library.json
    python -m src.fetch.lastfm --limit 50 # quick smoke test

Writes data/raw/tags.parquet with one row per (track_id, tag):
    track_id, tag, weight (Last.fm count 0-100), source ("track" | "artist" | "none")
Tracks with no tags at all get a single row with tag=None, source="none", so they
are not re-fetched every run.

If a streaming history has been imported (src.fetch.history), it then tags the artists you play most
but have no tagged songs by, for the listening charts' genres: data/raw/artist_tags.parquet
(artist, tag, weight). Same resumability.
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import pylast
from dotenv import load_dotenv

from src import paths
from src.paths import ROOT, TAGS, load_library

MIN_TRACK_TAGS = 3        # fewer than this -> fall back to artist tags
MAX_TAGS = 50
FLUSH_EVERY = 100
HISTORY_ARTISTS = 500     # most-played artists to tag from the streaming history


def network() -> pylast.LastFMNetwork:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("LASTFM_API_KEY")
    if not key:
        raise RuntimeError("Set LASTFM_API_KEY in .env (free key: https://www.last.fm/api/account/create)")
    return pylast.LastFMNetwork(api_key=key, api_secret=os.environ.get("LASTFM_API_SECRET"))


def _top_tags(obj) -> list[tuple[str, int]]:
    for attempt in range(3):
        try:
            return [(t.item.get_name().strip().lower(), int(t.weight))
                    for t in obj.get_top_tags(limit=MAX_TAGS)]
        except pylast.WSError:           # unknown track/artist
            return []
        except (pylast.NetworkError, pylast.MalformedResponseError):
            time.sleep(2 ** attempt)
    return []


def fetch_one(net, artist: str, title: str, artist_cache: dict) -> tuple[list, str]:
    tags = _top_tags(net.get_track(artist, title))
    if len(tags) >= MIN_TRACK_TAGS:
        return tags, "track"
    if artist not in artist_cache:
        artist_cache[artist] = _top_tags(net.get_artist(artist))
    if artist_cache[artist]:
        return artist_cache[artist], "artist"
    return tags, ("track" if tags else "none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    lib = load_library()
    net = network()
    old = pd.read_parquet(TAGS) if TAGS.exists() else pd.DataFrame(
        columns=["track_id", "tag", "weight", "source"])
    done = set(old["track_id"])
    todo = [t for t in lib["tracks"] if t not in done][: args.limit]
    print(f"{len(done)} cached, {len(todo)} to fetch")

    rows, artist_cache, t0 = [], {}, time.time()

    def flush():
        nonlocal old, rows
        if rows:
            old = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
            old.to_parquet(TAGS, index=False)
            rows = []

    for i, tid in enumerate(todo, 1):
        tr = lib["tracks"][tid]
        artist = tr["artists"][0] if tr["artists"] else ""
        tags, source = fetch_one(net, artist, tr["name"], artist_cache)
        if tags:
            rows += [{"track_id": tid, "tag": g, "weight": w, "source": source} for g, w in tags]
        else:
            rows.append({"track_id": tid, "tag": None, "weight": 0, "source": "none"})
        time.sleep(0.2)  # stay well under Last.fm's ~5 req/s
        if i % FLUSH_EVERY == 0:
            flush()
            print(f"  {i}/{len(todo)}  ({(time.time() - t0) / 60:.1f} min)")
    flush()

    per = old.groupby("track_id")["source"].first().value_counts()
    print(f"Saved {TAGS}\nTag source per track:\n{per.to_string()}")
    tag_history_artists(net, lib, old, artist_cache)


def tag_history_artists(net, lib: dict, track_tags: pd.DataFrame, artist_cache: dict) -> None:
    """Artist tags for the most-played artists in the streaming history that no tagged song covers."""
    hist_path, out = paths.history_path(paths.LIBRARY), paths.artist_tags_path(paths.LIBRARY)
    if not hist_path.exists():
        return
    plays = pd.read_parquet(hist_path, columns=["artist", "ms"])
    top = plays.groupby("artist")["ms"].sum().sort_values(ascending=False).head(HISTORY_ARTISTS).index
    tagged = track_tags.loc[track_tags["tag"].notna(), "track_id"].unique()
    covered = {(lib["tracks"].get(t, {}).get("artists") or [None])[0] for t in tagged}
    old = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["artist", "tag", "weight"])
    covered |= set(old["artist"])
    todo = [a for a in top if a and a not in covered]
    print(f"\nArtists from your listening history: {len(todo)} to tag")
    rows = []
    for i, a in enumerate(todo, 1):
        if a not in artist_cache:
            artist_cache[a] = _top_tags(net.get_artist(a))
            time.sleep(0.2)
        rows += [{"artist": a, "tag": g, "weight": w} for g, w in artist_cache[a]] or \
                [{"artist": a, "tag": None, "weight": 0}]
        if i % FLUSH_EVERY == 0 or i == len(todo):
            old = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
            old.to_parquet(out, index=False)
            rows = []
            print(f"  {i}/{len(todo)}")
    if todo:
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
