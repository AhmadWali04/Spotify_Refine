"""Web front end: session/login plumbing, pipeline jobs and the taste-gallery data, on the synthetic library."""
import json
import sys

import pytest

from src import paths
from src.app import insights, jobs
from src.fetch.synthetic import make, make_tags
from tests.test_sorter import tmp_data  # noqa: F401  (fixture)


@pytest.fixture
def demo_app(tmp_data, monkeypatch):  # noqa: F811
    from src.app.server import create_app
    from src.sorter import run, spaces
    for k in ("SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET"):
        monkeypatch.delenv(k, raising=False)
    lib = make(n_songs=900, n_playlists=10, seed=2)
    lib_path = tmp_data / "library_synthetic.json"
    lib_path.write_text(json.dumps(lib))
    make_tags(lib).to_parquet(paths.tags_path(lib_path), index=False)
    paths.write_json(paths.for_library(lib_path, "suggestions.json"),
                     run.build(lib, lib_path, spaces.resolve(lib, lib_path)))
    return lib, lib_path, create_app(lib_path).test_client()


def test_session_in_demo_mode(demo_app):
    _, _, c = demo_app
    s = c.get("/api/session").get_json()
    assert s["mode"] == "demo" and s["ready"] and s["user"] is None and not s["configured"]
    assert s["done"] == {"pull": True, "tags": True, "sort": True}
    assert set(s["jobs"]["steps"]) == {"pull", "tags", "sort"}


def test_login_without_keys_goes_back_to_landing(demo_app):
    _, _, c = demo_app
    r = c.get("/login")
    assert r.status_code == 302 and r.headers["Location"].startswith("/?error=")
    assert c.post("/api/mode", json={"mode": "nope"}).status_code == 400


def test_real_library_needs_login_before_running(demo_app):
    _, _, c = demo_app
    assert c.post("/api/mode", json={"mode": "spotify"}).get_json()["ok"]
    s = c.get("/api/session").get_json()
    assert s["mode"] == "spotify" and not s["ready"]
    assert c.post("/api/run", json={"steps": ["pull"]}).status_code == 400
    assert c.get("/api/state").status_code == 400            # no suggestions for the real library yet


def test_insights(demo_app):
    lib, _, c = demo_app
    d = c.get("/api/insights").get_json()
    assert d["ok"] and d["demo"]
    assert d["summary"]["liked"] == len(lib["liked"])
    assert len(d["eras"]) == len(lib["liked"]) and all(e[1] for e in d["eras"])   # demo has liked-at times
    g = d["galaxy"]
    assert g and len(g["points"]) > 100 and all(-1.0001 <= p[0] <= 1.0001 for p in g["points"])
    assert max(i for p in g["points"] for i in p[4]) < len(d["groups"])
    k = d["kinship"]
    assert len(k["sim"]) == len(k["names"]) and all(abs(k["sim"][i][i] - 1) < 1e-6 for i in range(len(k["names"])))
    assert d["dna"]["rows"] and all(0 <= v <= 1 for r in d["dna"]["rows"] for v in r["share"])
    assert d["orbit"][0]["songs"] >= d["orbit"][-1]["songs"]


def test_insights_without_tags_or_space(demo_app):
    lib, lib_path, _ = demo_app
    paths.tags_path(lib_path).unlink()
    out = insights.build(lib, {"spaces": [], "clusters": [], "songs": []}, lib_path)
    assert out["galaxy"] is None and out["kinship"] is None and out["dna"] is None
    assert out["orbit"] and out["summary"]["liked"] == len(lib["liked"])


def test_jobs_run_and_report(tmp_data, monkeypatch):  # noqa: F811
    monkeypatch.setattr(jobs, "commands", lambda lp: {
        "pull": [sys.executable, "-c", "print('1/2'); print('2/2')"],
        "tags": None,
        "sort": [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"]})
    done = []
    j = jobs.Jobs(on_step_done=done.append)
    j.start(tmp_data / "lib.json", ["pull", "tags", "sort"])
    j.thread.join(30)
    snap = j.snapshot()
    assert snap["steps"]["pull"]["status"] == "done" and snap["steps"]["pull"]["progress"] == 1.0
    assert snap["steps"]["tags"]["status"] == "skipped"
    assert snap["steps"]["sort"]["status"] == "failed" and "boom" in snap["steps"]["sort"]["lines"]
    assert done == ["pull"]
    with pytest.raises(ValueError):
        j.start(tmp_data / "lib.json", ["bogus"])


def test_real_library_sorts_with_tags_until_phase1_decides(tmp_data):  # noqa: F811
    cmds = jobs.commands(paths.LIBRARY)
    assert cmds["sort"][-2:] == ["--config", jobs.TAGS_ONLY_CONFIG]
    assert (paths.CONFIGS / "e1b_tags_lsa.yaml").exists()
