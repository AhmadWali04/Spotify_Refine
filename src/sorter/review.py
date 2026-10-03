"""Your review decisions (data/review.json) and the change plan they add up to.

A decision per song:
    {"action": "add",  "playlist_ids": [...]}   put it in one or more existing playlists
    {"action": "new",  "cluster_id": "c3"}      put it in a new playlist (a leftover cluster)
    {"action": "skip"}                          leave it where it is
Per cluster: {"name": ..., "approved": bool, "playlist_id": <set once created on Spotify>}

Nothing touches Spotify until the plan is applied (src/sorter/apply.py). Applied decisions are
stamped with "applied_at" and never re-applied.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src import paths

ACTIONS = {"add", "new", "skip"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Review:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}
        self.data.setdefault("decisions", {})
        self.data.setdefault("clusters", {})

    @classmethod
    def for_library(cls, lib_path: Path) -> "Review":
        return cls(paths.for_library(lib_path, "review.json"))

    @property
    def decisions(self) -> dict:
        return self.data["decisions"]

    @property
    def clusters(self) -> dict:
        return self.data["clusters"]

    def save(self) -> None:
        paths.write_json(self.path, self.data)

    def decide(self, track_id: str, action: str, playlist_ids: list[str] | None = None,
               cluster_id: str | None = None) -> dict:
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {sorted(ACTIONS)}")
        old = self.decisions.get(track_id)
        if old and old.get("applied_at"):
            raise ValueError("This song's change was already applied to Spotify; undo it there first.")
        if action == "add" and not playlist_ids:
            raise ValueError("'add' needs at least one playlist")
        if action == "new" and not cluster_id:
            raise ValueError("'new' needs a cluster_id")
        d = {"action": action, "decided_at": _now()}
        if action == "add":
            d["playlist_ids"] = list(dict.fromkeys(playlist_ids))
        if action == "new":
            d["cluster_id"] = cluster_id
        self.decisions[track_id] = d
        return d

    def undo(self, track_id: str) -> None:
        d = self.decisions.get(track_id)
        if d and d.get("applied_at"):
            raise ValueError("Already applied to Spotify; use `python -m src.sorter.apply --undo` instead.")
        self.decisions.pop(track_id, None)

    def set_cluster(self, cluster_id: str, name: str | None = None, approved: bool | None = None) -> dict:
        c = self.clusters.setdefault(cluster_id, {})
        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Playlist name can't be empty")
            if c.get("playlist_id") and name != c.get("name"):
                raise ValueError("This playlist already exists on Spotify; rename it there.")
            c["name"] = name[:100]
        if approved is not None:
            c["approved"] = bool(approved)
        return c


def cluster_members(sugg: dict, review: Review, cluster_id: str) -> list[str]:
    """Songs headed for a new playlist: cluster songs you haven't sent elsewhere, plus songs
    you explicitly moved into it."""
    base = next((c["track_ids"] for c in sugg["clusters"] if c["id"] == cluster_id), [])
    out = []
    for t in base:
        d = review.decisions.get(t)
        if d is None or (d["action"] == "new" and d["cluster_id"] == cluster_id):
            out.append(t)
    out += [t for t, d in review.decisions.items()
            if d["action"] == "new" and d.get("cluster_id") == cluster_id and t not in base]
    return out


def build_plan(lib: dict, sugg: dict, review: Review) -> dict:
    """The exact Spotify changes your decisions imply, minus anything already done."""
    existing = {p["id"]: set(p["track_ids"]) for p in lib["playlists"]}
    pl_names = {p["id"]: p["name"] for p in lib["playlists"]}
    adds: dict[str, list[str]] = {}
    for t, d in review.decisions.items():
        if d["action"] != "add" or d.get("applied_at"):
            continue
        for pid in d["playlist_ids"]:
            if t not in existing.get(pid, set()):
                adds.setdefault(pid, []).append(t)

    creates = []
    for c in sugg["clusters"]:
        meta = review.clusters.get(c["id"], {})
        if not meta.get("approved"):
            continue
        members = [t for t in cluster_members(sugg, review, c["id"])
                   if not review.decisions.get(t, {}).get("applied_at")]
        if not members:
            continue
        creates.append({"cluster_id": c["id"], "name": meta.get("name") or c["name"],
                        "playlist_id": meta.get("playlist_id"),     # set: playlist exists, just add
                        "track_ids": members})

    return {
        "add": [{"playlist_id": pid, "name": pl_names.get(pid, pid), "track_ids": ts} for pid, ts in adds.items()],
        "create": creates,
        "n_songs": sum(len(v) for v in adds.values()) + sum(len(c["track_ids"]) for c in creates),
    }
