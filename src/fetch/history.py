"""Import your Spotify streaming history (the privacy export) for the listening charts.

The Web API only returns your last 50 plays, so hours-over-time needs the export. Request it at
https://www.spotify.com/account/privacy/ :
  - "Extended streaming history": every play since the account was made (takes up to 30 days).
    Files: Streaming_History_Audio_*.json (older exports: endsong_*.json).
  - "Account data": the last year only (takes ~5 days). Files: StreamingHistory_music_*.json.

    python -m src.fetch.history ~/Downloads/my_spotify_data.zip           # merges with what's there
    python -m src.fetch.history ~/Downloads/Spotify\\ Extended\\ Streaming\\ History --replace

Accepts .zip files, folders and .json files, in any mix. Writes data/raw/history.parquet with one
row per play: ts (UTC, when the play ended), ms, track, artist, album, track_id (None if unknown).
Podcasts, audiobooks and video are left out.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from src import paths

COLUMNS = ["ts", "ms", "track", "artist", "album", "track_id"]
# Extended history, its older name, and the one-year "account data" music files.
WANTED = re.compile(r"^(streaming_history_audio|endsong|streaminghistory_music|streaminghistory\d)", re.I)


def _rows_extended(items: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(items)
    if "master_metadata_track_name" not in df:
        return pd.DataFrame(columns=COLUMNS)
    uri = df.get("spotify_track_uri", pd.Series(None, index=df.index)).astype("string")
    return pd.DataFrame({
        "ts": pd.to_datetime(df["ts"], utc=True, errors="coerce"),
        "ms": pd.to_numeric(df["ms_played"], errors="coerce"),
        "track": df["master_metadata_track_name"],
        "artist": df.get("master_metadata_album_artist_name"),
        "album": df.get("master_metadata_album_album_name"),
        "track_id": uri.str.extract(r"^spotify:track:(\w+)$")[0],
    })


def _rows_account(items: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(items)
    if "trackName" not in df:                               # a podcast file
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame({
        "ts": pd.to_datetime(df["endTime"], utc=True, errors="coerce"),   # "YYYY-MM-DD HH:MM", UTC
        "ms": pd.to_numeric(df["msPlayed"], errors="coerce"),
        "track": df["trackName"], "artist": df.get("artistName"), "album": None, "track_id": None,
    })


def parse_json(raw: bytes | str) -> pd.DataFrame:
    items = json.loads(raw)
    if not isinstance(items, list) or not items:
        return pd.DataFrame(columns=COLUMNS)
    return _rows_extended(items) if "ts" in items[0] else _rows_account(items)


def _sources(p: Path):
    """(name, bytes) of every history JSON inside a zip, a folder or a single file."""
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file():
                yield from _sources(f)
    elif p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if name.lower().endswith(".json") and WANTED.match(Path(name).name):
                    yield name, z.read(name)
    elif p.suffix.lower() == ".json" and WANTED.match(p.name):
        yield p.name, p.read_bytes()


def read_files(files) -> tuple[pd.DataFrame, list[str]]:
    """files: paths, or (name, bytes) pairs (uploads). Returns the plays and the file names used."""
    frames, used = [], []
    for f in files:
        if isinstance(f, tuple):
            name, data = f
            srcs = _sources_from_upload(name, data)
        else:
            srcs = _sources(Path(f).expanduser())
        for name, data in srcs:
            frames.append(parse_json(data))
            used.append(Path(name).name)
    if not frames:
        return pd.DataFrame(columns=COLUMNS), used
    return clean(pd.concat(frames, ignore_index=True)), used


def _sources_from_upload(name: str, data: bytes):
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for n in z.namelist():
                if n.lower().endswith(".json") and WANTED.match(Path(n).name):
                    yield n, z.read(n)
    elif name.lower().endswith(".json") and WANTED.match(Path(name).name):
        yield name, data


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["ts", "track"])
    df = df[df["ms"] > 0].copy()
    df["ms"] = df["ms"].astype("int64")
    for c in ("track", "artist", "album", "track_id"):
        df[c] = df[c].astype("string")
    # The two export formats overlap in the last year: same play, ts to the second vs. the minute.
    key = [df["ts"].dt.floor("min"), df["artist"].str.lower(), df["track"].str.lower()]
    df = df.assign(_k=pd.MultiIndex.from_arrays(key)).sort_values(["ts", "album"], na_position="last")
    df = df[~df["_k"].duplicated()].drop(columns="_k")
    return df[COLUMNS].reset_index(drop=True)


def link_tracks(df: pd.DataFrame, lib: dict) -> pd.DataFrame:
    """Fill track_id (and album) for plays without a URI by artist + title in the library."""
    index = {}
    for tid, t in lib["tracks"].items():
        if t.get("artists"):
            index.setdefault((t["artists"][0].lower(), t["name"].lower()), (tid, t.get("album")))
    miss = df["track_id"].isna()
    if miss.any():
        hits = pd.Series([index.get((str(a).lower(), str(n).lower()))
                          for a, n in zip(df.loc[miss, "artist"], df.loc[miss, "track"])], index=df.index[miss])
        df.loc[miss, "track_id"] = hits.map(lambda h: h[0] if h else None)
        df["album"] = df["album"].fillna(hits.map(lambda h: h[1] if h else None))
    return df


def save(new: pd.DataFrame, lib_path: Path, replace: bool = False) -> pd.DataFrame:
    out = paths.history_path(lib_path)
    if out.exists() and not replace:
        new = clean(pd.concat([pd.read_parquet(out), new], ignore_index=True))
    try:
        new = link_tracks(new, paths.load_library(lib_path))
    except FileNotFoundError:
        pass
    new.to_parquet(out, index=False)
    return new


def summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No plays."
    linked = df["track_id"].notna().mean()
    return (f"{len(df):,} plays, {df['ms'].sum() / 3.6e6:,.0f} hours, "
            f"{df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d}, {linked:.0%} matched to a Spotify track")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="export .zip, folder, or .json files")
    ap.add_argument("--replace", action="store_true", help="drop the history already imported")
    ap.add_argument("--library", default=str(paths.LIBRARY))
    args = ap.parse_args()
    df, used = read_files(args.files)
    if not used:
        raise SystemExit("No streaming-history JSON found. Look for Streaming_History_Audio_*.json "
                         "or StreamingHistory_music_*.json in the export.")
    print(f"Read {len(used)} file(s): {summary(df)}")
    df = save(df, Path(args.library), args.replace)
    print(f"Saved {paths.history_path(Path(args.library))}: {summary(df)}")
    print("Next: `python -m src.fetch.lastfm` tags the artists you play but haven't liked, for the genre charts.")


if __name__ == "__main__":
    main()
