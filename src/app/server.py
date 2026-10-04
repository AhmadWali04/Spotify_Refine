"""Flask API + single-page front end. Binds to 127.0.0.1 only: it's a personal tool.

Flow: log in with Spotify (or pick the demo library) -> run pull / tags / sort from the Home view
-> review suggestions -> apply -> browse the taste gallery.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote, urlparse

from flask import Flask, jsonify, redirect, request, send_from_directory

from src import paths
from src.app import auth, insights, jobs
from src.sorter import apply as applier
from src.sorter.review import Review, build_plan, cluster_members

STATIC = Path(__file__).parent / "static"


def create_app(lib_path: Path, write_client=None) -> Flask:
    """write_client: factory for a write-scoped Spotify client (tests pass a fake)."""
    app = Flask(__name__, static_folder=None)
    state: dict = {"lib_path": Path(lib_path), "key": None, "insights": None}
    session = auth.Session()

    def is_demo_path(p: Path) -> bool:
        return Path(p).resolve() == jobs.DEMO_LIBRARY.resolve() or Path(p).stem == "library_synthetic"

    def sugg_path() -> Path:
        return paths.for_library(state["lib_path"], "suggestions.json")

    def ensure() -> None:
        """(Re)load library, suggestions and review whenever the files on disk change."""
        lp, sp = state["lib_path"], sugg_path()
        if not sp.exists():
            raise FileNotFoundError(f"{sp.name} not found. Run the steps on the Home page "
                                    "(or `python -m src.sorter.run`) first.")
        key = (str(lp), lp.stat().st_mtime if lp.exists() else None, sp.stat().st_mtime)
        if state["key"] == key:
            return
        state["lib"] = paths.load_library(lp)
        state["sugg"] = json.loads(sp.read_text())
        state["review"] = Review.for_library(lp)
        state["key"], state["insights"] = key, None

    def step_done(_name: str) -> None:
        state["key"] = None                                  # force a reload on the next request

    runner = jobs.Jobs(on_step_done=step_done)

    def ok(**kw):
        return jsonify({"ok": True, **kw})

    @app.errorhandler(ValueError)
    @app.errorhandler(KeyError)
    @app.errorhandler(FileNotFoundError)
    @app.errorhandler(RuntimeError)
    def bad(e):
        return jsonify({"ok": False, "error": str(e).strip("'\"")}), 400

    # ------------------------------------------------------------ pages

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name):
        return send_from_directory(STATIC, name)

    # ------------------------------------------------------------ login

    def to_landing(msg: str):
        return redirect("/?error=" + quote(msg))

    @app.get("/login")
    def login():
        try:
            return redirect(session.login_url())
        except (RuntimeError, ValueError) as e:
            return to_landing(str(e))

    @app.get(urlparse(auth.redirect_uri()).path or "/callback")
    def callback():
        if request.args.get("error"):
            return to_landing(f"Spotify login was cancelled ({request.args['error']}).")
        try:
            session.finish(request.args.get("code", ""), request.args.get("state"))
        except Exception as e:                                  # surface any OAuth failure on the landing page
            return to_landing(f"Login failed: {e}")
        if is_demo_path(state["lib_path"]):
            state["lib_path"], state["key"] = paths.LIBRARY, None
        return redirect("/#home")

    @app.post("/api/logout")
    def logout():
        session.logout()
        return ok()

    @app.post("/api/mode")
    def mode():
        """Switch between the demo library and your own (real) library."""
        if runner.running:
            raise RuntimeError("Wait for the running step to finish first.")
        m = request.get_json(force=True).get("mode")
        if m not in ("demo", "spotify"):
            raise ValueError("mode must be 'demo' or 'spotify'")
        state["lib_path"] = jobs.DEMO_LIBRARY if m == "demo" else paths.LIBRARY
        state["key"] = None
        return ok()

    @app.get("/api/session")
    def get_session():
        lp = state["lib_path"]
        demo = is_demo_path(lp)
        user = None if demo else session.user()
        done = jobs.done_state(lp)
        lib_user = None
        if done["pull"] and not demo:
            lib_user = json.loads(lp.read_text()).get("user_id")
            if user and lib_user and lib_user != user["id"]:
                done = {"pull": False, "tags": False, "sort": False}   # someone else's library on disk
        return jsonify({
            "configured": auth.configured(), "redirect_uri": auth.redirect_uri(),
            "mode": "demo" if demo else "spotify", "user": user, "library_user": lib_user,
            "done": done, "ready": done["sort"], "jobs": runner.snapshot(),
            "has_decision": paths.for_library(lp, "decision.json").exists(),
        })

    # ------------------------------------------------------------ pipeline jobs

    @app.post("/api/run")
    def run_steps():
        lp = state["lib_path"]
        if not is_demo_path(lp) and not session.user():
            raise RuntimeError("Log in with Spotify first.")
        steps = request.get_json(force=True).get("steps") or list(jobs.STEPS)
        runner.start(lp, steps)
        return ok(jobs=runner.snapshot())

    @app.post("/api/stop")
    def stop_steps():
        runner.stop()
        return ok()

    # ------------------------------------------------------------ review

    @app.get("/api/state")
    def get_state():
        ensure()
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
        state["key"] = None
        ensure()
        return ok()

    @app.post("/api/decide")
    def decide():
        ensure()
        b = request.get_json(force=True)
        d = state["review"].decide(b["track_id"], b["action"], b.get("playlist_ids"), b.get("cluster_id"))
        state["review"].save()
        return ok(decision=d)

    @app.post("/api/undo")
    def undo():
        ensure()
        state["review"].undo(request.get_json(force=True)["track_id"])
        state["review"].save()
        return ok()

    @app.post("/api/bulk_accept")
    def bulk_accept():
        """Accept the top suggestion for every undecided song in a tier (default: confident)."""
        ensure()
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
        ensure()
        b = request.get_json(force=True)
        c = state["review"].set_cluster(b["cluster_id"], b.get("name"), b.get("approved"))
        state["review"].save()
        return ok(cluster=c)

    @app.get("/api/plan")
    def plan():
        ensure()
        p = build_plan(state["lib"], state["sugg"], state["review"])
        return ok(plan=p, summary=applier.summarize(p), demo=bool(state["lib"].get("synthetic_truth")),
                  applied=sorted((x.name for x in paths.APPLIED.glob("*.json")), reverse=True))

    @app.post("/api/apply")
    def apply():
        ensure()
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
        ensure()
        name = Path(request.get_json(force=True)["changelog"]).name     # no path traversal
        sp = (write_client or applier.write_client)()
        return ok(result=applier.undo(sp, paths.APPLIED / name, state["review"]))

    # ------------------------------------------------------------ gallery

    @app.get("/api/insights")
    def get_insights():
        ensure()
        if state["insights"] is None:
            state["insights"] = insights.build(state["lib"], state["sugg"], state["lib_path"])
        return ok(**state["insights"])

    return app
