"""Human-readable "vibes" for a group of songs (a playlist or a leftover cluster).

Uses whatever is on disk: Last.fm tags (distinctive tags by lift over the whole library),
Essentia named axes (mood / danceability / valence / arousal as z-scores vs. the library,
plus the top Discogs style), top artists and the dominant decade.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.featurize.tags import STOP_TAGS

AXIS_LABELS = {
    "mood_happy": ("happy", "not happy"), "mood_sad": ("sad", "not sad"),
    "mood_aggressive": ("aggressive", "gentle"), "mood_relaxed": ("relaxed", "tense"),
    "mood_party": ("party", "low-key"), "danceability": ("danceable", "not danceable"),
    "valence": ("upbeat", "dark"), "arousal": ("high energy", "calm"),
}


class Describer:
    def __init__(self, lib: dict, tags_file: Path | None = None, use_essentia: bool = True):
        self.lib = lib
        self.tag_sets: dict[str, set[str]] = {}
        if tags_file is not None and Path(tags_file).exists():
            df = pd.read_parquet(tags_file)
            df = df[df["tag"].notna() & ~df["tag"].isin(STOP_TAGS) & (df["weight"] >= 10)]
            self.tag_sets = df.groupby("track_id")["tag"].agg(set).to_dict()
        n = max(len(self.tag_sets), 1)
        self.global_freq = Counter(g for s in self.tag_sets.values() for g in s)
        self.global_freq = {g: c / n for g, c in self.global_freq.items()}

        self.ess, self.ess_names = {}, []
        if use_essentia:
            from src.featurize import essentia_feats
            ids, X = essentia_feats.cached_features(list(lib["tracks"]))
            if ids:
                self.ess_names = essentia_feats.feature_names()
                self.ess = dict(zip(ids, X))
                A = X[:, : len(essentia_feats.NAMED_AXES)]
                self.ax_mu, self.ax_sd = A.mean(axis=0), np.maximum(A.std(axis=0), 1e-6)

    def _tags(self, ids: list[str], top: int = 5) -> list[dict]:
        have = [self.tag_sets[t] for t in ids if t in self.tag_sets]
        if len(have) < 3:
            return []
        freq = Counter(g for s in have for g in s)
        out = []
        for g, c in freq.items():
            f = c / len(have)
            if f < 0.15:
                continue
            lift = f / max(self.global_freq.get(g, f), 1e-9)
            out.append({"tag": g, "share": round(f, 2), "lift": round(lift, 2), "score": f * np.log(max(lift, 1.0001))})
        out.sort(key=lambda r: -r["score"])
        return [{k: v for k, v in r.items() if k != "score"} for r in out[:top]]

    def _axes(self, ids: list[str]) -> tuple[list[dict], str | None]:
        from src.featurize.essentia_feats import NAMED_AXES
        rows = [self.ess[t] for t in ids if t in self.ess]
        if not rows or len(rows) < 0.5 * len(ids):
            return [], None
        X = np.stack(rows)
        z = (X[:, : len(NAMED_AXES)].mean(axis=0) - self.ax_mu) / self.ax_sd * np.sqrt(min(len(rows), 25)) / 5
        axes = []
        for j in np.argsort(-np.abs(z))[:3]:
            if abs(z[j]) < 0.3:
                continue
            hi, lo = AXIS_LABELS[NAMED_AXES[j]]
            axes.append({"axis": NAMED_AXES[j], "label": hi if z[j] > 0 else lo, "z": round(float(z[j]), 2)})
        g = X[:, len(NAMED_AXES):].mean(axis=0)
        genre = self.ess_names[len(NAMED_AXES) + int(g.argmax())].removeprefix("genre:") if g.size else None
        return axes, genre

    def describe(self, ids: list[str]) -> dict:
        tracks = self.lib["tracks"]
        artists = Counter(a for t in ids for a in tracks.get(t, {}).get("artists", [])[:1])
        years = [int(str(tracks[t].get("release_date") or "")[:4]) for t in ids
                 if str(tracks.get(t, {}).get("release_date") or "")[:4].isdigit()]
        decade = None
        if years:
            dec, c = Counter(y // 10 * 10 for y in years).most_common(1)[0]
            if c >= 0.6 * len(years):
                decade = f"{dec}s"
        axes, genre = self._axes(ids)
        return {"size": len(ids), "tags": self._tags(ids), "axes": axes, "genre": genre,
                "artists": [a for a, _ in artists.most_common(3)], "decade": decade}


def _title(s: str) -> str:
    """'city pop' -> 'City Pop', but '80s' stays '80s' and 'rnb' / 'DnB' keep their own case."""
    return " ".join(w[:1].upper() + w[1:] for w in s.split())


def suggest_name(desc: dict, fallback: str) -> str:
    tags = [t["tag"] for t in desc["tags"]]
    if tags:
        return " · ".join(_title(t) for t in tags[:2])
    if desc.get("genre"):
        return desc["genre"].split("---")[-1]
    if desc.get("axes"):
        return " · ".join(a["label"].title() for a in desc["axes"][:2])
    if desc.get("artists"):
        return f"Like {desc['artists'][0]}"
    return fallback
