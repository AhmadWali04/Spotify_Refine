"""Flask API behind the review page. Binds to 127.0.0.1 only: it's a personal tool."""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src import paths
from src.sorter import apply as applier
from src.sorter.review import Review, build_plan, cluster_members

STATIC = Path(__file__).parent / "static"


def create_app(lib_path: Path, write_client=None) -> Flask:
    """write_client: factory for a write-scoped Spotify client (tests pass a fake)."""
    app = Flask(__name__, static_folder=None)
    state: dict = {}

    def load():
        state["lib"] = paths.load_library(lib_path)
        sp = paths.for_library(lib_path, "suggestions.json")
        if not sp.exists():
            raise FileNotFoundError(f"{sp.name} not found. Run `python -m src.sorter.run"
                                    + ("" if lib_path.resolve() == paths.LIBRARY.resolve()
                                       else f" --library {lib_path}") + "` first.")
        state["sugg"] = json.loads(sp.read_text())
        state["review"] = Review.for_library(lib_path)

    load()

    def ok(**kw):
        return jsonify({"ok": True, **kw})

    @app.errorhandler(ValueError)
    @app.errorhandler(KeyError)
    @app.errorhandler(FileNotFoundError)
    @app.errorhandler(RuntimeError)
    def bad(e):
        return jsonify({"ok": False, "error": str(e).strip("'\"")}), 400

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name):
        return send_from_directory(STATIC, name)

    @app.get("/api/state")
    def get_state():
        sugg, review = state["sugg"], state["review"]
        clusters = []
        for c in sugg["clusters"]:
            meta = review.clusters.get(c["id"], {})
            clusters.append({**c, "name": meta.get("name") or c["name"], "suggested_name": c["name"],
                             "approved": bool(meta.get("approved")), "playlist_id": meta.get("playlist_id"),
                             "members": cluster_members(sugg, review, c["id"])})
        return jsonify({
            "meta": {k: sugg[k] for k in ("generated_at", "pulled_at", "library", "demo", "spaces", "counts")},
            "playlists": sugg["playlists"], "clusters": clusters, "songs": sugg["songs"],
            "decisions": review.decisions,
        })

    @app.post("/api/reload")
    def reload():
        load()
        return ok()

    @app.post("/api/decide")
    def decide():
        b = request.get_json(force=True)
        d = state["review"].decide(b["track_id"], b["action"], b.get("playlist_ids"), b.get("cluster_id"))
        state["review"].save()
        return ok(decision=d)

    @app.post("/api/undo")
    def undo():
        state["review"].undo(request.get_json(force=True)["track_id"])
        state["review"].save()
        return ok()

    @app.post("/api/bulk_accept")
    def bulk_accept():
        """Accept the top suggestion for every undecided song in a tier (default: confident)."""
        tier = request.get_json(force=True).get("tier", "confident")
        review, n = state["review"], 0
        for s in state["sugg"]["songs"]:
            if s["tier"] == tier and s["suggestions"] and s["track_id"] not in review.decisions:
                review.decide(s["track_id"], "add", [s["suggestions"][0]["playlist_id"]])
                n += 1
        review.save()
        return ok(accepted=n)

    @app.post("/api/cluster")
    def cluster():
        b = request.get_json(force=True)
        c = state["review"].set_cluster(b["cluster_id"], b.get("name"), b.get("approved"))
        state["review"].save()
        return ok(cluster=c)

    @app.get("/api/plan")
    def plan():
        p = build_plan(state["lib"], state["sugg"], state["review"])
        return ok(plan=p, summary=applier.summarize(p), demo=bool(state["lib"].get("synthetic_truth")),
                  applied=sorted((x.name for x in paths.APPLIED.glob("*.json")), reverse=True))

    @app.post("/api/apply")
    def apply():
        if state["lib"].get("synthetic_truth"):
            raise ValueError("Demo library: nothing on Spotify to change. Pull your real library to apply.")
        p = build_plan(state["lib"], state["sugg"], state["review"])
        if not p["n_songs"]:
            raise ValueError("Nothing to apply yet.")
        sp = (write_client or applier.write_client)()
        log = applier.apply_plan(sp, p, state["review"])
        return ok(log={k: v for k, v in log.items() if k != "path"}, changelog=Path(log["path"]).name)

    @app.post("/api/undo_apply")
    def undo_apply():
        name = Path(request.get_json(force=True)["changelog"]).name     # no path traversal
        sp = (write_client or applier.write_client)()
        return ok(result=applier.undo(sp, paths.APPLIED / name, state["review"]))

    return app
