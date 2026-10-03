"""Build suggestions for every liked song that isn't in a playlist yet.

    python -m src.sorter.run                       # uses data/decision.json + data/model_choice.json
    python -m src.sorter.run --config configs/e1b_tags_lsa.yaml
    python -m src.sorter.run --library data/raw/library_synthetic.json   # demo, no keys needed
    python -m src.sorter.run --recompare           # re-run the model comparison first

Each unsorted liked song gets its top-3 playlists with calibrated probabilities and a tier:
    confident  top-1 probability above the threshold that held 90% precision on sorted songs
    suggested  a reasonable guess; review it
    leftover   low confidence, or a new kind of music (src/sorter/novelty.py) -> clustered into new playlists
    no_data    no featurizer could vectorize it (e.g. no tags and no audio)

Writes data/suggestions.json, which the review app reads.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import paths
from src.evaluate import MIN_PLAYLIST_SIZE
from src.sorter import calibrate, cluster, compare, models, novelty, spaces
from src.sorter.describe import Describer, suggest_name
from src.vectors import load_space


def scored_playlists(lib: dict, min_size: int = MIN_PLAYLIST_SIZE) -> list[dict]:
    """Same playlists, same order as evaluate.labeled_set, so model indices map back to ids."""
    return [p for p in lib["playlists"] if len(p["track_ids"]) >= min_size]


def _choice_is_current(choice: dict, lib: dict, cfgs: list[dict]) -> bool:
    return (choice.get("pulled_at") == lib.get("pulled_at")
            and [s["config"] for s in choice["spaces"]] == cfgs)


def load_choice(lib: dict, lib_path: Path, cfgs: list[dict], recompare: bool) -> dict:
    path = paths.for_library(lib_path, "model_choice.json")
    if not recompare and path.exists():
        choice = json.loads(path.read_text())
        if _choice_is_current(choice, lib, cfgs):
            return choice
        print("Model choice is stale (library or featurizer changed); re-running the comparison.")
    return compare.run(lib, lib_path, cfgs)


def _previews() -> dict[str, str]:
    if not paths.ITUNES.exists():
        return {}
    df = pd.read_parquet(paths.ITUNES)
    df = df[df["status"] == "ok"]
    return dict(zip(df["track_id"], df["preview_url"]))


def build(lib: dict, lib_path: Path, cfgs: list[dict], recompare: bool = False,
          min_cluster: int = 8) -> dict:
    choice = load_choice(lib, lib_path, cfgs, recompare)
    targets_pl = scored_playlists(lib)
    P = len(targets_pl)
    sorted_ids = {t for p in lib["playlists"] for t in p["track_ids"]}
    labels: dict[str, set[int]] = {}
    for j, p in enumerate(targets_pl):
        for t in p["track_ids"]:
            labels.setdefault(t, set()).add(j)
    unsorted = [x["id"] for x in lib["liked"] if x["id"] not in sorted_ids]
    remaining = list(dict.fromkeys(unsorted))
    print(f"\n{len(unsorted)} liked songs are not in any playlist; {P} playlists can receive songs.")

    songs: dict[str, dict] = {}
    leftover_by_space: list[tuple[str, list[str], np.ndarray]] = []
    for sp in choice["spaces"]:
        cfg, cal = sp["config"], sp["calibration"]
        ids, X = load_space(cfg, lib, lib_path)
        row = {t: i for i, t in enumerate(ids)}
        lab_ids = [t for t in sorted(labels) if t in row]
        Xl = X[[row[t] for t in lab_ids]]
        yl = [labels[t] for t in lab_ids]
        tgt = [t for t in remaining if t in row]
        remaining = [t for t in remaining if t not in row]
        if not tgt:
            continue
        Xt = X[[row[t] for t in tgt]]
        m = models.make(sp["model"]).fit(Xl, yl, P)
        Pm = calibrate.softmax(m.scores(Xt), cal["temperature"])
        k_free = novelty.default_k(len(tgt))
        margin = novelty.null_margin(Xl, yl, k_free, min_cluster)
        new = novelty.anchored_new(Xt, models.Centroid().fit(Xl, yl, P).mu, k_free, min_cluster, margin=margin)
        conf_thr = cal["confident_threshold"] if cal["confident_threshold"] is not None else np.inf
        left = []
        for i, t in enumerate(tgt):
            top = np.argsort(-Pm[i])[:3]
            conf = float(Pm[i, top[0]])
            novel = bool(new[i])
            if novel or conf < cal["abstain_threshold"]:
                tier = "leftover"
                left.append(i)
            elif conf >= conf_thr:
                tier = "confident"
            else:
                tier = "suggested"
            songs[t] = {"track_id": t, "tier": tier, "confidence": round(conf, 4), "novel": novel,
                        "space": sp["run_id"], "cluster": None,
                        "suggestions": [{"playlist_id": targets_pl[j]["id"], "p": round(float(Pm[i, j]), 4)}
                                        for j in top]}
        leftover_by_space.append((sp["run_id"], [tgt[i] for i in left], Xt[left]))

    for t in remaining:
        songs[t] = {"track_id": t, "tier": "no_data", "confidence": None, "novel": None,
                    "space": None, "cluster": None, "suggestions": []}

    describer = Describer(lib, paths.tags_path(lib_path))
    clusters = []
    for space_id, ids, Xs in leftover_by_space:
        lab, k, sil = cluster.auto_cluster(Xs, min_size=min_cluster)
        print(f"Leftovers in {space_id}: {len(ids)} songs -> {k} clusters (silhouette {sil:.2f})")
        for j in range(k):
            members = [t for t, c in zip(ids, lab) if c == j]
            cid = f"c{len(clusters)}"
            desc = describer.describe(members)
            clusters.append({"id": cid, "space": space_id, "track_ids": members, "vibe": desc,
                             "name": suggest_name(desc, f"New mix {len(clusters) + 1}")})
            for t in members:
                songs[t]["cluster"] = cid

    previews = _previews()
    for t, s in songs.items():
        tr = lib["tracks"].get(t, {})
        s.update({"name": tr.get("name"), "artists": tr.get("artists", []), "album": tr.get("album"),
                  "release_date": tr.get("release_date"), "preview_url": previews.get(t)})

    playlists = []
    scored = {p["id"] for p in targets_pl}
    for p in lib["playlists"]:
        playlists.append({"id": p["id"], "name": p["name"], "size": len(p["track_ids"]),
                          "scored": p["id"] in scored,
                          "writable": p.get("owner") == lib.get("user_id") or bool(p.get("collaborative")),
                          "vibe": describer.describe(p["track_ids"])})

    tiers = pd.Series([s["tier"] for s in songs.values()]).value_counts().to_dict()
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pulled_at": lib.get("pulled_at"), "library": lib_path.name,
        "demo": bool(lib.get("synthetic_truth")),
        "spaces": [{"run_id": s["run_id"], "model": s["model"], "top1": s["metrics"]["top1_mean"],
                    "top3": s["metrics"]["top3_mean"], "calibration": s["calibration"]} for s in choice["spaces"]],
        "counts": {"unsorted": len(unsorted), **tiers},
        "playlists": playlists, "clusters": clusters,
        "songs": sorted(songs.values(), key=lambda s: -(s["confidence"] or -1)),
    }
    print("Tiers: " + ", ".join(f"{k} {v}" for k, v in tiers.items()))
    if lib.get("synthetic_truth"):
        demo_report(lib, out, targets_pl)
    return out


def demo_report(lib: dict, out: dict, targets_pl: list[dict]) -> None:
    """On the synthetic library we know the answer: check the product end to end."""
    truth = lib["synthetic_truth"]
    gid = {f"g{int(p['id'][1:])}": p["id"] for p in targets_pl}
    belongs, hidden = [], []
    for s in out["songs"]:
        groups = truth[s["track_id"]]
        if any(g.startswith("h") for g in groups):
            hidden.append(s["tier"] == "leftover")
        elif s["suggestions"]:
            right = {gid[g] for g in groups if g in gid}
            if right:
                belongs.append((s["suggestions"][0]["playlist_id"] in right,
                                any(x["playlist_id"] in right for x in s["suggestions"]), s["tier"]))
    if belongs:
        b = np.array([(a, c) for a, c, _ in belongs])
        conf = [a for a, _, tier in belongs if tier == "confident"]
        print(f"[demo] songs that belong in an existing playlist: top-1 {100 * b[:, 0].mean():.0f}%, "
              f"top-3 {100 * b[:, 1].mean():.0f}%, confident-tier precision "
              f"{100 * np.mean(conf) if conf else float('nan'):.0f}% ({len(conf)} songs)")
    if hidden:
        print(f"[demo] songs from brand-new groups sent to leftovers: {100 * np.mean(hidden):.0f}%")
    for c in out["clusters"]:
        gs = pd.Series([truth[t][0] for t in c["track_ids"]]).value_counts()
        print(f"[demo] cluster '{c['name']}' ({len(c['track_ids'])} songs): "
              f"{100 * gs.iloc[0] / gs.sum():.0f}% from group {gs.index[0]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=str(paths.LIBRARY))
    ap.add_argument("--config", help="featurizer config (default: data/decision.json)")
    ap.add_argument("--fallback", help="second featurizer config for songs the first can't cover")
    ap.add_argument("--recompare", action="store_true", help="re-run the model comparison first")
    ap.add_argument("--min-cluster", type=int, default=8, help="smallest new-playlist cluster")
    args = ap.parse_args()
    lib_path = Path(args.library)
    lib = paths.load_library(lib_path)
    cfgs = spaces.resolve(lib, lib_path, args.config, args.fallback)
    out = build(lib, lib_path, cfgs, args.recompare, args.min_cluster)
    path = paths.for_library(lib_path, "suggestions.json")
    paths.write_json(path, out)
    print(f"Saved {path.relative_to(paths.ROOT)}. Review with: python -m src.app"
          + ("" if lib_path.resolve() == paths.LIBRARY.resolve() else f" --library {args.library}"))


if __name__ == "__main__":
    main()
