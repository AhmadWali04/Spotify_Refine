"""Write your reviewed changes to Spotify, with a changelog for undo.

    python -m src.sorter.apply                 # dry run: print the plan, change nothing
    python -m src.sorter.apply --yes           # apply it
    python -m src.sorter.apply --undo data/applied/2026-10-02T21-04-11.json

Only adds songs and creates new private playlists; it never removes or reorders anything you
already have. Uses the Feb 2026 endpoints: POST /me/playlists, POST /playlists/{id}/items,
DELETE /playlists/{id}/items, DELETE /me/library (unfollow, for undo).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src import paths
from src.sorter.review import Review, build_plan

CHUNK = 100                       # Spotify's max items per add / remove request
DESCRIPTION = "Made with Spotify Refine"


def uri(track_id: str) -> str:
    return f"spotify:track:{track_id}"


def write_client():
    from src.fetch.spotify import WRITE_SCOPES, client
    return client(WRITE_SCOPES)


def _add(sp, playlist_id: str, uris: list[str]) -> None:
    for s in range(0, len(uris), CHUNK):
        sp._post(f"playlists/{playlist_id}/items", payload={"uris": uris[s:s + CHUNK]})


def apply_plan(sp, plan: dict, review: Review, changelog_dir: Path = paths.APPLIED) -> dict:
    """Runs the plan, stamping each decision as it lands so a crash halfway is safe to re-run."""
    stamp = datetime.now().isoformat(timespec="seconds")
    log = {"applied_at": stamp, "created": [], "added": [], "errors": []}
    log_path = changelog_dir / f"{stamp.replace(':', '-')}.json"

    def mark(track_ids):
        for t in track_ids:
            if t in review.decisions:
                review.decisions[t]["applied_at"] = stamp
            else:   # a cluster song you never touched individually
                review.decisions[t] = {"action": "auto", "applied_at": stamp}
        review.save()
        paths.write_json(log_path, log)

    for c in plan["create"]:
        try:
            pid = c.get("playlist_id")
            if not pid:
                pid = sp._post("me/playlists", payload={"name": c["name"], "public": False,
                                                        "description": DESCRIPTION})["id"]
                review.set_cluster(c["cluster_id"])["playlist_id"] = pid
                log["created"].append({"cluster_id": c["cluster_id"], "playlist_id": pid, "name": c["name"]})
            uris = [uri(t) for t in c["track_ids"]]
            _add(sp, pid, uris)
            log["added"].append({"playlist_id": pid, "name": c["name"], "uris": uris})
            mark(c["track_ids"])
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            log["errors"].append({"what": f"create '{c['name']}'", "error": str(e)})

    for a in plan["add"]:
        try:
            uris = [uri(t) for t in a["track_ids"]]
            _add(sp, a["playlist_id"], uris)
            log["added"].append({"playlist_id": a["playlist_id"], "name": a["name"], "uris": uris})
            mark(a["track_ids"])
        except Exception as e:  # noqa: BLE001
            log["errors"].append({"what": f"add to '{a['name']}'", "error": str(e)})

    review.data.setdefault("applied", []).append(log_path.name)
    review.save()
    paths.write_json(log_path, log)
    log["path"] = str(log_path)
    return log


def undo(sp, log_path: Path, review: Review | None = None) -> dict:
    """Remove exactly what one apply added, and unfollow the playlists it created."""
    log = json.loads(Path(log_path).read_text())
    created = {c["playlist_id"] for c in log["created"]}
    done = {"removed": 0, "unfollowed": 0, "errors": []}
    for a in log["added"]:
        if a["playlist_id"] in created:
            continue                                   # the whole playlist goes below
        try:
            for s in range(0, len(a["uris"]), CHUNK):
                sp._delete(f"playlists/{a['playlist_id']}/items",
                           payload={"items": [{"uri": u} for u in a["uris"][s:s + CHUNK]]})
            done["removed"] += len(a["uris"])
        except Exception as e:  # noqa: BLE001
            done["errors"].append(f"{a['name']}: {e}")
    for pid in created:
        try:
            sp._delete("me/library", uris=f"spotify:playlist:{pid}")
            done["unfollowed"] += 1
        except Exception as e:  # noqa: BLE001
            done["errors"].append(f"unfollow {pid}: {e}")
    if review is not None:
        stamp = log["applied_at"]
        for t, d in list(review.decisions.items()):
            if d.get("applied_at") == stamp:
                if d["action"] == "auto":
                    del review.decisions[t]
                else:
                    d.pop("applied_at")
        for c in review.clusters.values():
            if c.get("playlist_id") in created:
                c.pop("playlist_id")
        review.save()
    return done


def summarize(plan: dict) -> str:
    lines = []
    for c in plan["create"]:
        verb = "add to new playlist" if c.get("playlist_id") else "create playlist"
        lines.append(f"  + {verb} '{c['name']}' with {len(c['track_ids'])} songs")
    for a in plan["add"]:
        lines.append(f"  + add {len(a['track_ids'])} songs to '{a['name']}'")
    return "\n".join(lines) or "  (nothing to do: review some songs first with `python -m src.app`)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=str(paths.LIBRARY))
    ap.add_argument("--yes", action="store_true", help="actually write to Spotify")
    ap.add_argument("--undo", metavar="CHANGELOG", help="revert one apply using its changelog file")
    args = ap.parse_args()
    lib_path = Path(args.library)
    review = Review.for_library(lib_path)

    if args.undo:
        print(undo(write_client(), Path(args.undo), review))
        return
    lib = paths.load_library(lib_path)
    if lib.get("synthetic_truth"):
        raise SystemExit("This is the synthetic demo library; there is nothing on Spotify to change.")
    sugg = json.loads(paths.for_library(lib_path, "suggestions.json").read_text())
    plan = build_plan(lib, sugg, review)
    print(f"Plan ({plan['n_songs']} songs):\n{summarize(plan)}")
    if not args.yes:
        print("\nDry run. Re-run with --yes to apply.")
        return
    log = apply_plan(write_client(), plan, review)
    print(f"\nDone. Changelog: {log['path']}")
    for e in log["errors"]:
        print(f"  ! {e['what']}: {e['error']}")
    print("Re-pull the library (python -m src.fetch.spotify) and re-run the sorter so it learns from these.")


if __name__ == "__main__":
    main()
