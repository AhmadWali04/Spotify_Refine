"""M1: one-time, read-only pull of liked songs and playlist memberships.

Usage:
    python -m src.fetch.spotify                 # only playlists you own
    python -m src.fetch.spotify --include-followed

Writes data/raw/library.json:
    {
      "pulled_at": ...,
      "tracks":    {track_id: {name, artists, album, duration_ms, release_date, isrc}},
      "liked":     [{"id": track_id, "added_at": ...}, ...],
      "playlists": [{"id", "name", "owner", "track_ids": [...]}, ...]
    }
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

from src.paths import LIBRARY, ROOT

SCOPES = "user-library-read playlist-read-private playlist-read-collaborative"
# Phase 2 write-back (src/sorter/apply.py). Spotipy re-prompts for consent if the cached token lacks these.
WRITE_SCOPES = SCOPES + " playlist-modify-private playlist-modify-public user-library-modify"


def client(scopes: str = SCOPES) -> spotipy.Spotify:
    load_dotenv(ROOT / ".env")
    redirect = os.environ.get("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
    if "localhost" in redirect:
        raise ValueError("Spotify requires 127.0.0.1 in the redirect URI, not localhost.")
    if not os.environ.get("SPOTIPY_CLIENT_ID") or not os.environ.get("SPOTIPY_CLIENT_SECRET"):
        raise RuntimeError("Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in .env (see README, Setup).")
    auth = SpotifyOAuth(
        scope=scopes,
        redirect_uri=redirect,
        cache_path=str(ROOT / ".spotify_cache"),
        open_browser=True,
    )
    return spotipy.Spotify(auth_manager=auth, requests_timeout=30, retries=5)


def _paged(sp: spotipy.Spotify, first: dict):
    page = first
    while page:
        yield from page["items"]
        page = sp.next(page) if page.get("next") else None


def _track_record(t: dict) -> dict:
    return {
        "name": t["name"],
        "artists": [a["name"] for a in t.get("artists", [])],
        "album": (t.get("album") or {}).get("name"),
        "release_date": (t.get("album") or {}).get("release_date"),
        "duration_ms": t.get("duration_ms"),
        "isrc": (t.get("external_ids") or {}).get("isrc"),
    }


def _playlist_items(sp: spotipy.Spotify, playlist_id: str):
    """Items of one playlist via /items (Feb 2026 rename); falls back to the old /tracks path."""
    try:
        first = sp._get(f"playlists/{playlist_id}/items", limit=100, additional_types="track")
    except spotipy.SpotifyException as e:
        if e.http_status not in (403, 404):
            raise
        first = sp.playlist_items(playlist_id, limit=100, additional_types=("track",))
    for item in _paged(sp, first):
        # Newer responses nest the object under "item"; older ones under "track".
        yield item.get("track") or item.get("item")


def pull(include_followed: bool = False) -> dict:
    sp = client()
    me = sp.current_user()
    print(f"Logged in as {me.get('display_name') or me['id']}")

    tracks: dict[str, dict] = {}
    liked = []
    for item in _paged(sp, sp.current_user_saved_tracks(limit=50)):
        t = item["track"]
        if not t or not t.get("id"):  # local files / unavailable tracks
            continue
        tracks[t["id"]] = _track_record(t)
        liked.append({"id": t["id"], "added_at": item.get("added_at")})
    print(f"Liked songs: {len(liked)}")

    playlists = []
    for p in _paged(sp, sp.current_user_playlists(limit=50)):
        owner = p["owner"]["id"]
        # Since Feb 2026, contents are only returned for playlists you own or collaborate on.
        if owner != me["id"] and not p.get("collaborative") and not include_followed:
            continue
        ids = []
        try:
            for t in _playlist_items(sp, p["id"]):
                if not t or t.get("type", "track") != "track" or not t.get("id"):
                    continue
                tracks.setdefault(t["id"], _track_record(t))
                ids.append(t["id"])
        except spotipy.SpotifyException as e:
            print(f"  ! skipped '{p['name']}': HTTP {e.http_status}")
            continue
        playlists.append({"id": p["id"], "name": p["name"], "owner": owner,
                          "collaborative": bool(p.get("collaborative")),
                          "track_ids": list(dict.fromkeys(ids))})

    return {
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "user_id": me["id"],
        "tracks": tracks,
        "liked": liked,
        "playlists": playlists,
    }


def summarize(lib: dict, min_size: int = 10) -> None:
    liked = {x["id"] for x in lib["liked"]}
    sizes = sorted(((len(p["track_ids"]), p["name"]) for p in lib["playlists"]), reverse=True)
    in_any = set().union(*(p["track_ids"] for p in lib["playlists"])) if lib["playlists"] else set()

    print(f"\n{'Size':>6}  {'Liked':>6}  Playlist")
    for p in sorted(lib["playlists"], key=lambda p: -len(p["track_ids"])):
        n = len(p["track_ids"])
        n_liked = len(liked & set(p["track_ids"]))
        flag = "" if n >= min_size else "   (excluded: < %d songs)" % min_size
        print(f"{n:>6}  {n_liked:>6}  {p['name']}{flag}")

    scored = [n for n, _ in sizes if n >= min_size]
    print(f"\nPlaylists: {len(sizes)} total, {len(scored)} with >= {min_size} songs")
    print(f"Liked songs in >= 1 playlist: {len(liked & in_any)} / {len(liked)}")
    print(f"Unique tracks (liked + playlists): {len(lib['tracks'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-followed", action="store_true",
                    help="also pull playlists you follow but don't own")
    ap.add_argument("--summary-only", action="store_true",
                    help="print sizes from the cached library.json without calling Spotify")
    args = ap.parse_args()

    if args.summary_only:
        lib = json.loads(LIBRARY.read_text())
    else:
        lib = pull(args.include_followed)
        LIBRARY.write_text(json.dumps(lib, indent=1))
        print(f"Saved {LIBRARY}")
    summarize(lib)


if __name__ == "__main__":
    main()
