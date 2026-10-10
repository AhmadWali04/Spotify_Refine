"""Data behind the listening charts: hours over time, genre share, the listening calendar, taste radars
and the artist chord diagram.

Plays come from the imported streaming history (src.fetch.history). Genres are Last.fm tags: a song's
own tags, else its artist's (pooled over your library's songs by them, or from src.fetch.lastfm's artist
pass). Each play counts toward one genre, its song's strongest tag, so genre hours add up to the total.
The radars and the chord diagram also work without a history, from your liked songs.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from src import paths
from src.featurize.tags import STOP_TAGS

DIMS = ("artist", "album", "track", "genre", "playlist")
BUCKETS = {"week": "W", "month": "M", "year": "Y"}     # weeks start on Monday
ALL = "__all__"                                         # series key for all listening
MAX_SERIES = 6
TOP_ITEMS = 300
TOP_GENRES = 8                                          # categorical palette slots
CHORD_ARTISTS = 24
MIN_LINK = 0.15                                         # genre overlap below this draws no ribbon
LINKS_PER_ARTIST = 4                                    # a ribbon survives if it's among either artist's strongest

# Tags that are not a genre (on top of the listener tags the sorter drops).
NON_GENRE = STOP_TAGS | {
    "female vocalists", "male vocalists", "female vocalist", "male vocalist", "female", "male",
    "british", "american", "english", "uk", "usa", "canadian", "australian", "swedish", "german", "french",
    "all", "music", "songs", "favorite songs", "favourite songs", "my music", "covers", "cover", "live",
}
DECADE = re.compile(r"^(\d{2}|\d{4})s?$")


def _tz(name: str | None) -> ZoneInfo:
    try:
        return ZoneInfo(name or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def _day(s: str | None, default: pd.Timestamp) -> pd.Timestamp:
    try:
        return pd.Timestamp(s).normalize() if s else default
    except ValueError:
        raise ValueError(f"Not a date: {s!r} (use YYYY-MM-DD).")


def _genre_tags(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["tag"].notna() & (df["weight"] >= 10)]
    return df[~df["tag"].isin(NON_GENRE) & ~df["tag"].str.match(DECADE) & (df["tag"].str.len() > 1)]


def _top_shares(weights: Counter, k: int = 5) -> list[tuple[str, float]]:
    top = weights.most_common(k)
    total = sum(w for _, w in top) or 1
    return [(g, w / total) for g, w in top]


class Listening:
    def __init__(self, lib: dict, lib_path: Path):
        self.lib = lib
        tracks = lib["tracks"]

        # ---- genres per track (best first) and per artist (normalized shares)
        self.track_genres: dict[str, list[str]] = {}
        artist_w: dict[str, Counter] = {}
        artist_n: Counter = Counter()
        tp = paths.tags_path(lib_path)
        if tp.exists():
            df = _genre_tags(pd.read_parquet(tp)).sort_values(["track_id", "weight"], ascending=[True, False])
            for tid, g in df.groupby("track_id", sort=False):
                self.track_genres[tid] = list(g["tag"].head(5))
                a = (tracks.get(tid, {}).get("artists") or [None])[0]
                if a:
                    artist_w.setdefault(a, Counter()).update(dict(zip(g["tag"].head(5), g["weight"].head(5))))
                    artist_n[a] += 1
        self.artist_genres = {a: _top_shares(c) for a, c in artist_w.items()}
        ap = paths.artist_tags_path(lib_path)
        if ap.exists():
            df = _genre_tags(pd.read_parquet(ap)).sort_values(["artist", "weight"], ascending=[True, False])
            for a, g in df.groupby("artist", sort=False):
                if a not in self.artist_genres:
                    self.artist_genres[a] = _top_shares(Counter(dict(zip(g["tag"].head(5), g["weight"].head(5)))))

        # ---- liked songs, for the radars and the chord diagram without a history
        liked = [x["id"] for x in lib["liked"]]
        self.liked = pd.DataFrame({
            "artist": [(tracks.get(t, {}).get("artists") or [None])[0] for t in liked],
            "genre": [self._genre(t, (tracks.get(t, {}).get("artists") or [None])[0]) for t in liked],
        })

        # ---- plays
        self.plays = None
        hp = paths.history_path(lib_path)
        if hp.exists():
            p = pd.read_parquet(hp)
            if len(p):
                self._set_plays(p)
        self._local: dict[str, pd.Series] = {}

    def _genre(self, track_id, artist) -> str | None:
        g = self.track_genres.get(track_id)
        if g:
            return g[0]
        ag = self.artist_genres.get(artist)
        return ag[0][0] if ag else None

    def _set_plays(self, p: pd.DataFrame) -> None:
        p = p.reset_index(drop=True)
        tid = p["track_id"].astype(object).where(p["track_id"].notna(), None)
        artist = p["artist"].fillna("Unknown artist").astype(str)
        by_artist = {a: (g[0][0] if g else None) for a, g in self.artist_genres.items()}
        genre = tid.map(lambda t: (self.track_genres.get(t) or [None])[0] if t else None)
        genre = genre.fillna(artist.map(by_artist))
        self.plays = pd.DataFrame({
            "ts": p["ts"], "hours": p["ms"] / 3.6e6, "artist": artist,
            "album": (p["album"].fillna("Unknown album").astype(str) + " · " + artist),
            "track": (p["track"].astype(str) + " · " + artist),
            "genre": genre, "track_id": tid,
        })
        # Playlist membership, one row per (play, playlist): a play counts toward every playlist it's in.
        member: dict[str, list[str]] = {}
        for pl in self.lib["playlists"]:
            for t in set(pl["track_ids"]):
                member.setdefault(t, []).append(pl["name"])
        self.playlist = tid.map(lambda t: member.get(t) if t else None).explode().dropna()

    # ------------------------------------------------------------ helpers

    @property
    def has_history(self) -> bool:
        return self.plays is not None

    def _require(self) -> pd.DataFrame:
        if self.plays is None:
            raise FileNotFoundError("No listening history yet. Import your Spotify data export first.")
        return self.plays

    def local(self, tz: str | None) -> pd.Series:
        """Play end times as naive local times in the viewer's time zone."""
        key = str(_tz(tz))
        if key not in self._local:
            self._local[key] = self._require()["ts"].dt.tz_convert(key).dt.tz_localize(None)
        return self._local[key]

    def _mask(self, dim: str, item: str) -> pd.Series:
        p = self._require()
        if item == ALL:
            return pd.Series(True, index=p.index)
        if dim == "playlist":
            m = pd.Series(False, index=p.index)
            m[self.playlist.index[self.playlist == item].unique()] = True
            return m
        if dim not in DIMS:
            raise ValueError(f"dim must be one of {', '.join(DIMS)}")
        return p[dim] == item

    def _ranked(self, dim: str) -> pd.Series:
        """Hours per item of a dimension, largest first."""
        p = self._require()
        if dim == "playlist":
            s = p["hours"].reindex(self.playlist.index).groupby(self.playlist.values).sum()
        elif dim in DIMS:
            s = p.groupby(dim)["hours"].sum()
        else:
            raise ValueError(f"dim must be one of {', '.join(DIMS)}")
        return s.sort_values(ascending=False)

    def genre_rank(self) -> list[str]:
        """Genres in overall order: the categorical colors are assigned by this, so they never move."""
        if self.has_history:
            return list(self._ranked("genre").index[:TOP_GENRES * 3])
        return [g for g, _ in Counter(self.liked["genre"].dropna()).most_common(TOP_GENRES * 3)]

    # ------------------------------------------------------------ endpoints

    def meta(self, tz: str | None = None) -> dict:
        out = {"has_history": self.has_history, "genre_rank": self.genre_rank(), "liked": len(self.liked),
               "liked_tagged": round(float(self.liked["genre"].notna().mean()), 3) if len(self.liked) else 0}
        if not self.has_history:
            return out
        p, loc = self.plays, self.local(tz)
        total = float(p["hours"].sum())
        out.update({
            "plays": len(p), "hours": round(total, 1),
            "first": loc.min().date().isoformat(), "last": loc.max().date().isoformat(),
            "years": sorted(int(y) for y in loc.dt.year.unique()),
            "tagged": round(float(p.loc[p["genre"].notna(), "hours"].sum() / total), 3) if total else 0,
            "items": {d: [[k, round(float(v), 2)] for k, v in self._ranked(d).head(TOP_ITEMS).items()] for d in DIMS},
        })
        return out

    def search(self, dim: str, q: str, limit: int = 20) -> list:
        s = self._ranked(dim)
        q = q.strip().lower()
        if q:
            s = s[s.index.str.lower().str.contains(q, regex=False)]
        return [[k, round(float(v), 2)] for k, v in s.head(limit).items()]

    def series(self, dim: str, bucket: str, items: list[str], tz: str | None = None) -> dict:
        p, loc = self._require(), self.local(tz)
        if bucket not in BUCKETS:
            raise ValueError("bucket must be week, month or year")
        freq = BUCKETS[bucket]
        if not items:
            items = list(self._ranked(dim).index[:3])
        items = items[:MAX_SERIES]
        per = loc.dt.to_period(freq)
        full = pd.period_range(per.min(), per.max(), freq=freq)
        out = []
        for it in items:
            m = self._mask(dim, it)
            s = p.loc[m, "hours"].groupby(per[m]).sum().reindex(full, fill_value=0.0)
            out.append({"key": it, "hours": [round(float(v), 3) for v in s.values]})
        return {"dim": dim, "bucket": bucket, "periods": [x.start_time.date().isoformat() for x in full], "series": out}

    def share(self, start: str | None, end: str | None, tz: str | None = None) -> dict:
        p, loc = self._require(), self.local(tz)
        a = _day(start, loc.min().normalize())
        b = _day(end, loc.max().normalize()) + pd.Timedelta(days=1)
        q = p[(loc >= a) & (loc < b)]
        g = q.groupby("genre")["hours"].sum().sort_values(ascending=False)
        top_artists = (q.dropna(subset=["genre"]).groupby(["genre", "artist"])["hours"].sum()
                       .sort_values(ascending=False).groupby(level=0).head(3))
        artists: dict[str, list[str]] = {}
        for (genre, artist), _ in top_artists.items():
            artists.setdefault(genre, []).append(artist)
        return {"start": a.date().isoformat(), "end": (b - pd.Timedelta(days=1)).date().isoformat(),
                "total": round(float(q["hours"].sum()), 2),
                "untagged": round(float(q.loc[q["genre"].isna(), "hours"].sum()), 2),
                "genres": [[k, round(float(v), 3), artists.get(k, [])] for k, v in g.head(100).items()]}

    def calendar(self, year: int, dim: str | None, item: str | None, tz: str | None = None) -> dict:
        p, loc = self._require(), self.local(tz)
        m = (loc.dt.year == year)
        if dim and item:
            m &= self._mask(dim, item)
        q, day = p[m], loc[m].dt.date
        daily = q["hours"].groupby(day).sum()
        top_col = "track" if dim in ("artist", "album", "track") and item else "artist"
        top = q.groupby([day, q[top_col]])["hours"].sum()
        best = top.groupby(level=0).idxmax() if len(top) else pd.Series(dtype=object)
        return {"year": year, "dim": dim, "item": item, "top_kind": top_col,
                "days": [[d.isoformat(), round(float(h), 3), best[d][1]] for d, h in daily.items()],
                "total": round(float(daily.sum()), 2)}

    def _shares(self, frame: pd.DataFrame, weight: str | None) -> pd.Series:
        f = frame.dropna(subset=["genre"])
        s = f.groupby("genre")[weight].sum() if weight else f["genre"].value_counts()
        return s / s.sum() if s.sum() else s

    def radar(self, a: tuple, b: tuple, tz: str | None = None) -> dict:
        """Genre share of listening in two periods, on the genres that lead either one."""
        p, loc = self._require(), self.local(tz)
        webs = []
        for start, end in (a, b):
            lo = _day(start, loc.min().normalize())
            hi = _day(end, loc.max().normalize()) + pd.Timedelta(days=1)
            q = p[(loc >= lo) & (loc < hi)]
            webs.append({"start": lo.date().isoformat(), "end": (hi - pd.Timedelta(days=1)).date().isoformat(),
                         "hours": round(float(q["hours"].sum()), 1), "s": self._shares(q, "hours")})
        return self._webs(webs)

    def overall(self) -> dict:
        """All-time taste: what you play (history hours) next to what you keep (liked songs)."""
        webs = []
        if self.has_history:
            webs.append({"label": "What you play", "unit": "hours", "hours": round(float(self.plays["hours"].sum()), 1),
                         "s": self._shares(self.plays, "hours")})
        webs.append({"label": "What you like", "unit": "songs", "songs": len(self.liked), "s": self._shares(self.liked, None)})
        return self._webs(webs)

    @staticmethod
    def _webs(webs: list[dict]) -> dict:
        score = Counter()
        for w in webs:
            score.update(w["s"].to_dict())
        axes = [g for g, _ in score.most_common(TOP_GENRES)]
        for w in webs:
            s = w.pop("s")
            w["values"] = [round(float(s.get(g, 0.0)), 4) for g in axes]
            w["covered"] = round(float(sum(w["values"])), 3)       # share of the web the axes account for
        return {"axes": axes, "webs": webs}

    def chord(self, n: int = CHORD_ARTISTS) -> dict:
        """Your top artists, linked by how much their genre mixes overlap."""
        if self.has_history:
            rank, unit = self._ranked("artist"), "hours"
        else:
            rank, unit = self.liked["artist"].dropna().value_counts().astype(float), "songs"
        artists = [a for a in rank.index if self.artist_genres.get(a)][:n]
        prof = {a: dict(self.artist_genres[a]) for a in artists}
        links = []
        for i, x in enumerate(artists):
            for j in range(i + 1, len(artists)):
                y = artists[j]
                shared = sorted(((g, min(prof[x][g], prof[y][g])) for g in prof[x].keys() & prof[y].keys()),
                                key=lambda t: -t[1])
                w = sum(v for _, v in shared)
                if w >= MIN_LINK:
                    links.append({"s": i, "t": j, "w": round(w, 3), "genre": shared[0][0],
                                  "shared": [[g, round(v, 3)] for g, v in shared]})
        # Keep each artist's strongest links only: generic shared tags would otherwise join everyone.
        best: dict[int, list[float]] = {}
        for l in links:
            for k in (l["s"], l["t"]):
                best.setdefault(k, []).append(l["w"])
        cut = {k: sorted(v, reverse=True)[:LINKS_PER_ARTIST][-1] for k, v in best.items()}
        links = [l for l in links if l["w"] >= cut[l["s"]] or l["w"] >= cut[l["t"]]]
        linked = {k for l in links for k in (l["s"], l["t"])}
        return {"unit": unit, "links": links, "lonely": [artists[i] for i in range(len(artists)) if i not in linked],
                "artists": [{"name": a, "value": round(float(rank[a]), 2),
                             "genres": [[g, round(v, 3)] for g, v in self.artist_genres[a]]} for a in artists]}
