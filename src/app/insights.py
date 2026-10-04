"""Data behind the taste gallery: everything is computed from what's already on disk
(library, tags, the sorter's vector space and suggestions), so it costs no API calls.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src import paths
from src.featurize.tags import STOP_TAGS
from src.linalg import pca

MAX_POINTS = 5000
TOP_ARTISTS = 40
TOP_PLAYLISTS = 12
TOP_TAGS = 14


def _year(date: str | None) -> int | None:
    try:
        return int(str(date)[:4])
    except (TypeError, ValueError):
        return None


def _tag_sets(lib_path: Path) -> dict[str, list[str]]:
    """Each track's tags (best first), without the stop tags the sorter also ignores."""
    f = paths.tags_path(lib_path)
    if not f.exists():
        return {}
    df = pd.read_parquet(f)
    df = df[df["tag"].notna() & ~df["tag"].isin(STOP_TAGS) & (df["weight"] >= 10)]
    df = df.sort_values(["track_id", "weight"], ascending=[True, False])
    return df.groupby("track_id")["tag"].agg(list).to_dict()


def _space(sugg: dict, lib: dict, lib_path: Path):
    """The sorter's primary vector space, or None if it can't be rebuilt."""
    from src.vectors import config_for_run, load_config, load_space
    if not sugg.get("spaces"):
        return None
    try:
        cfg = load_config(config_for_run(sugg["spaces"][0]["run_id"]))
        return load_space(cfg, lib, lib_path)
    except (FileNotFoundError, KeyError, ValueError):
        return None


def galaxy(lib: dict, sp, run_id: str, groups: list[dict]) -> dict | None:
    """2-D PCA of every liked or playlisted song in the sorter's space."""
    if sp is None:
        return None
    ids, X = sp
    keep_set = {x["id"] for x in lib["liked"]} | {t for p in lib["playlists"] for t in p["track_ids"]}
    keep = [i for i, t in enumerate(ids) if t in keep_set]
    if len(keep) < 3:
        return None
    if len(keep) > MAX_POINTS:
        keep = sorted(np.random.default_rng(0).choice(keep, MAX_POINTS, replace=False))
    Z, _, ratio = pca(X[keep], 2)
    Z = Z / np.maximum(np.abs(Z).max(axis=0), 1e-9)
    member: dict[str, list[int]] = {}
    for g, grp in enumerate(groups):
        for t in grp["track_ids"]:
            member.setdefault(t, []).append(g)
    pts = []
    for (x, y), i in zip(Z, keep):
        t = ids[i]
        tr = lib["tracks"].get(t, {})
        pts.append([round(float(x), 4), round(float(y), 4), tr.get("name"),
                    ", ".join(tr.get("artists", [])[:2]), member.get(t, [])])
    return {"points": pts, "explained": [round(float(r), 3) for r in ratio], "space": run_id}


def kinship(sp, top: list[dict]) -> dict | None:
    """Cosine similarity between playlist centroids, ordered so similar playlists sit together."""
    if sp is None or len(top) < 2:
        return None
    ids, X = sp
    row = {t: i for i, t in enumerate(ids)}
    names, cents = [], []
    for p in top:
        idx = [row[t] for t in p["track_ids"] if t in row]
        if len(idx) >= 3:
            names.append(p["name"])
            cents.append(X[idx].mean(axis=0))
    if len(cents) < 2:
        return None
    C = np.array(cents)
    C = C - C.mean(axis=0)                                  # relative to the average playlist
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    S = C @ C.T
    order = np.argsort(pca(C, 1)[0][:, 0]) if len(C) > 2 else np.arange(len(C))
    S = S[np.ix_(order, order)]
    return {"names": [names[i] for i in order], "sim": np.round(S, 3).tolist()}


def build(lib: dict, sugg: dict, lib_path: Path) -> dict:
    liked = lib["liked"]
    liked_ids = [x["id"] for x in liked]
    tracks = lib["tracks"]
    tags = _tag_sets(lib_path)

    # ---- headline numbers
    artists = Counter(a for t in liked_ids for a in tracks.get(t, {}).get("artists", [])[:1])
    years = [y for t in liked_ids if (y := _year(tracks.get(t, {}).get("release_date")))]
    decades = Counter(10 * (y // 10) for y in years)
    summary = {
        "liked": len(liked_ids), "artists": len(artists), "playlists": len(lib["playlists"]),
        "unsorted": sugg.get("counts", {}).get("unsorted"),
        "top_decade": max(decades, key=decades.get) if decades else None,
        "median_year": int(np.median(years)) if years else None,
        "variety": round(len(artists) / max(len(liked_ids), 1), 3),   # unique first artists per liked song
        "hours": round(sum(tracks.get(t, {}).get("duration_ms") or 0 for t in liked_ids) / 3.6e6, 1),
    }

    # ---- eras: release year vs. when it was liked
    eras = [[_year(tracks.get(x["id"], {}).get("release_date")), x.get("added_at"),
             tracks.get(x["id"], {}).get("name"), ", ".join(tracks.get(x["id"], {}).get("artists", [])[:2])]
            for x in liked]
    eras = [e for e in eras if e[0]]

    # ---- artist orbit
    first_tag = {}
    for t in liked_ids:
        a = (tracks.get(t, {}).get("artists") or [None])[0]
        if a and tags.get(t):
            first_tag.setdefault(a, Counter()).update(tags[t][:3])
    orbit = [{"artist": a, "songs": n, "tag": first_tag[a].most_common(1)[0][0] if a in first_tag else None}
             for a, n in artists.most_common(TOP_ARTISTS)]

    # ---- groups that can be highlighted: playlists by size, then the sorter's new-playlist clusters
    pls = sorted(lib["playlists"], key=lambda p: -len(p["track_ids"]))
    groups = [{"name": p["name"], "kind": "playlist", "track_ids": p["track_ids"]} for p in pls]
    groups += [{"name": c["name"], "kind": "new", "track_ids": c["track_ids"]} for c in sugg.get("clusters", [])]
    unsorted = [s["track_id"] for s in sugg.get("songs", [])]
    if unsorted:
        groups.append({"name": "Not in a playlist", "kind": "unsorted", "track_ids": unsorted})

    # ---- genre DNA: share of each top playlist's songs carrying each top tag
    dna = None
    top_pl = pls[:TOP_PLAYLISTS]
    if tags:
        tag_count = Counter(g for t in set(liked_ids) | {t for p in top_pl for t in p["track_ids"]}
                            for g in tags.get(t, [])[:5])
        top_tags = [g for g, _ in tag_count.most_common(TOP_TAGS)]
        rows = []
        for p in top_pl:
            have = [set(tags[t][:5]) for t in p["track_ids"] if t in tags]
            if have:
                rows.append({"playlist": p["name"], "n": len(have),
                             "share": [round(sum(g in s for s in have) / len(have), 3) for g in top_tags]})
        dna = {"tags": top_tags, "rows": rows} if rows and top_tags else None

    sp = _space(sugg, lib, lib_path)
    run_id = sugg["spaces"][0]["run_id"] if sugg.get("spaces") else None
    return {
        "summary": summary,
        "groups": [{"name": g["name"], "kind": g["kind"], "size": len(g["track_ids"])} for g in groups],
        "galaxy": galaxy(lib, sp, run_id, groups),
        "eras": eras,
        "orbit": orbit,
        "dna": dna,
        "kinship": kinship(sp, top_pl),
        "demo": bool(lib.get("synthetic_truth")),
    }
