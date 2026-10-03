"""Phase 2 checks: models, calibration, clustering, novelty, the review plan, Spotify writes
(against a fake client), the full pipeline on the synthetic library, and the review API."""
import json

import numpy as np
import pandas as pd
import pytest

from src import paths
from src.evaluate import topk_hits
from src.fetch.synthetic import make, make_tags
from src.sorter import calibrate, cluster, models, novelty
from src.sorter.review import Review, build_plan, cluster_members


# ---------------------------------------------------------------- helpers

def blobs(n_per=60, P=5, d=16, sep=4.0, seed=0, extra_labels=0.1):
    """P separable Gaussian blobs; some songs carry a second label."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((P, d)) * sep
    X, y = [], []
    for p in range(P):
        X.append(centers[p] + rng.standard_normal((n_per, d)))
        y += [{p} for _ in range(n_per)]
    y = [ls | ({(next(iter(ls)) + 1) % P} if rng.random() < extra_labels else set()) for ls in y]
    return np.vstack(X), y, centers


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    """Point every output path at a temp dir so tests never touch real data."""
    for name in ("DATA", "VECTORS", "APPLIED", "REPORTS"):
        d = tmp_path / name.lower()
        d.mkdir()
        monkeypatch.setattr(paths, name, d)
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "ITUNES", tmp_path / "itunes.parquet")
    monkeypatch.setattr(paths, "TAGS", tmp_path / "tags.parquet")
    return tmp_path


# ---------------------------------------------------------------- models

@pytest.mark.parametrize("name", list(models.MODELS))
def test_models_separate_blobs(name):
    X, y, _ = blobs()
    rng = np.random.default_rng(1)
    te = rng.random(len(X)) < 0.25
    m = models.make(name).fit(X[~te], [y[i] for i in np.where(~te)[0]], 5)
    S = m.scores(X[te])
    assert S.shape == (te.sum(), 5)
    assert topk_hits(S, [y[i] for i in np.where(te)[0]], 1).mean() > 0.9


def test_models_handle_empty_playlist():
    X, y, _ = blobs(P=3)
    for name in ("centroid", "lda", "mahalanobis"):
        S = models.make(name).fit(X, y, 4).scores(X[:5])      # playlist 3 has no songs
        assert np.all(S[:, 3] == -np.inf)


def test_lda_reduces_high_dims():
    X, y, _ = blobs(d=300, sep=1.5)
    m = models.LDA(max_dims=20).fit(X, y, 5)
    assert m.W.shape == (20, 5)


# ---------------------------------------------------------------- calibration

def test_softmax_rows_sum_to_one_with_inf():
    S = np.array([[1.0, 2.0, -np.inf], [0.0, 0.0, 0.0]])
    P = calibrate.softmax(S, 0.5)
    np.testing.assert_allclose(P.sum(axis=1), 1)
    assert P[0, 2] == 0


def test_temperature_beats_extremes():
    X, y, _ = blobs(sep=1.0)
    S = models.Centroid().fit(X, y, 5).scores(X)
    T = calibrate.fit_temperature(S, y)
    assert calibrate.nll(S, y, T) <= min(calibrate.nll(S, y, T * 30), calibrate.nll(S, y, T / 30))


def test_confident_threshold():
    conf = np.linspace(1, 0, 100)
    hit = np.r_[np.ones(50), np.zeros(50)]
    thr = calibrate.confident_threshold(conf, hit, target=0.9)
    assert 0.4 < thr < 0.5           # precision stays >= 90% until ~55 songs in
    assert calibrate.confident_threshold(conf, np.zeros(100)) == np.inf


# ---------------------------------------------------------------- clustering + novelty

def test_auto_cluster_finds_three_groups():
    X, y, _ = blobs(n_per=40, P=3, extra_labels=0)
    lab, k, sil = cluster.auto_cluster(X, min_size=8)
    assert k == 3 and sil > 0.5
    truth = np.array([next(iter(s)) for s in y])
    for j in range(3):
        assert np.bincount(truth[lab == j]).max() / (lab == j).sum() > 0.95


def test_novelty_flags_new_groups():
    X, y, centers = blobs(n_per=80, P=4, extra_labels=0)
    rng = np.random.default_rng(5)
    new_center = rng.standard_normal(16) * 4
    known = X[::2][:100]                                   # unsorted songs from known playlists
    new = new_center + rng.standard_normal((60, 16))
    mu = models.Centroid().fit(X, y, 4).mu
    U = np.vstack([known, new])
    margin = novelty.null_margin(X, y, novelty.default_k(len(U)), 8)
    flag = novelty.anchored_new(U, mu, margin=margin)
    assert flag[100:].mean() > 0.8
    assert flag[:100].mean() < 0.1


# ---------------------------------------------------------------- review plan

def _lib_and_sugg():
    lib = {"user_id": "me", "playlists": [{"id": "A", "name": "Alpha", "owner": "me", "track_ids": ["x"]},
                                          {"id": "B", "name": "Beta", "owner": "me", "track_ids": []}]}
    sugg = {"clusters": [{"id": "c0", "name": "Mix", "track_ids": ["n1", "n2", "n3"]}]}
    return lib, sugg


def test_plan_dedupes_and_respects_decisions(tmp_path):
    lib, sugg = _lib_and_sugg()
    r = Review(tmp_path / "review.json")
    r.decide("x", "add", ["A", "B"])            # already in A -> only B
    r.decide("y", "add", ["A"])
    r.decide("n2", "skip")                      # pulled out of the cluster
    r.decide("z", "new", cluster_id="c0")       # moved into the cluster
    plan = build_plan(lib, sugg, r)
    assert {a["playlist_id"]: a["track_ids"] for a in plan["add"]} == {"B": ["x"], "A": ["y"]}
    assert plan["create"] == []                 # cluster not approved yet
    r.set_cluster("c0", name="  Night Mix ", approved=True)
    plan = build_plan(lib, sugg, r)
    assert plan["create"][0]["name"] == "Night Mix"
    assert plan["create"][0]["track_ids"] == ["n1", "n3", "z"]
    assert cluster_members(sugg, r, "c0") == ["n1", "n3", "z"]


def test_review_validation(tmp_path):
    r = Review(tmp_path / "review.json")
    with pytest.raises(ValueError):
        r.decide("x", "add", [])
    with pytest.raises(ValueError):
        r.decide("x", "nope")
    with pytest.raises(ValueError):
        r.set_cluster("c0", name="   ")


# ---------------------------------------------------------------- Spotify writes (fake client)

class FakeSpotify:
    def __init__(self):
        self.calls = []

    def _post(self, url, payload=None, **kw):
        self.calls.append(("POST", url, payload))
        return {"id": "NEWPL"} if url == "me/playlists" else {"snapshot_id": "s"}

    def _delete(self, url, payload=None, **kw):
        self.calls.append(("DELETE", url, payload or kw))
        return {}


def test_apply_and_undo(tmp_data):
    from src.sorter import apply as applier
    lib, sugg = _lib_and_sugg()
    r = Review(tmp_data / "review.json")
    for i in range(150):
        r.decide(f"t{i}", "add", ["B"])
    r.set_cluster("c0", approved=True, name="Night Mix")
    sp = FakeSpotify()
    log = applier.apply_plan(sp, build_plan(lib, sugg, r), r, changelog_dir=paths.APPLIED)
    posts = [c for c in sp.calls if c[0] == "POST"]
    assert posts[0] == ("POST", "me/playlists", {"name": "Night Mix", "public": False, "description": applier.DESCRIPTION})
    assert ("POST", "playlists/NEWPL/items", {"uris": ["spotify:track:n1", "spotify:track:n2", "spotify:track:n3"]}) in posts
    sizes = [len(c[2]["uris"]) for c in posts if c[1] == "playlists/B/items"]
    assert sizes == [100, 50]                                   # chunked
    assert not log["errors"]
    assert build_plan(lib, sugg, r)["n_songs"] == 0             # nothing re-applied
    with pytest.raises(ValueError):
        r.decide("t0", "skip")                                  # applied songs are locked

    sp2 = FakeSpotify()
    res = applier.undo(sp2, log["path"], r)
    assert res == {"removed": 150, "unfollowed": 1, "errors": []}
    assert ("DELETE", "me/library", {"uris": "spotify:playlist:NEWPL"}) in sp2.calls
    assert build_plan(lib, sugg, r)["n_songs"] == 153           # back to pending


# ---------------------------------------------------------------- pipeline + app on the synthetic library

@pytest.fixture
def demo(tmp_data):
    lib = make(n_songs=1200, n_playlists=12, seed=4)
    lib_path = tmp_data / "library_synthetic.json"
    lib_path.write_text(json.dumps(lib))
    make_tags(lib).to_parquet(paths.tags_path(lib_path), index=False)
    return lib, lib_path


def test_pipeline_end_to_end(demo):
    from src.sorter import run, spaces
    lib, lib_path = demo
    out = run.build(lib, lib_path, spaces.resolve(lib, lib_path))
    unsorted = {x["id"] for x in lib["liked"]} - {t for p in lib["playlists"] for t in p["track_ids"]}
    assert {s["track_id"] for s in out["songs"]} == unsorted
    truth = lib["synthetic_truth"]
    pid = {f"g{int(p['id'][1:])}": p["id"] for p in lib["playlists"]}
    conf = [s for s in out["songs"] if s["tier"] == "confident"]
    assert conf and np.mean([s["suggestions"][0]["playlist_id"] in {pid[g] for g in truth[s["track_id"]] if g in pid}
                             for s in conf]) > 0.85
    hidden = [s for s in out["songs"] if truth[s["track_id"]][0].startswith("h")]
    assert np.mean([s["tier"] == "leftover" for s in hidden]) > 0.6
    assert out["clusters"] and all(c["name"] for c in out["clusters"])
    json.dumps(out)                                             # serializable for the app


def test_app_api(demo):
    from src.app.server import create_app
    from src.sorter import run, spaces
    lib, lib_path = demo
    paths.write_json(paths.for_library(lib_path, "suggestions.json"),
                     run.build(lib, lib_path, spaces.resolve(lib, lib_path)))
    c = create_app(lib_path).test_client()
    st = c.get("/api/state").get_json()
    song = st["songs"][0]
    assert c.post("/api/decide", json={"track_id": song["track_id"], "action": "add",
                                       "playlist_ids": [song["suggestions"][0]["playlist_id"]]}).get_json()["ok"]
    assert c.post("/api/decide", json={"track_id": "x", "action": "bad"}).status_code == 400
    assert c.post("/api/bulk_accept", json={}).get_json()["accepted"] >= 0
    if st["clusters"]:
        c.post("/api/cluster", json={"cluster_id": st["clusters"][0]["id"], "approved": True})
    plan = c.get("/api/plan").get_json()
    assert plan["plan"]["n_songs"] > 0 and plan["demo"]
    assert c.post("/api/apply", json={}).status_code == 400    # demo libraries never write
    assert c.get("/").status_code == 200


# ---------------------------------------------------------------- decide + iTunes matching

def test_decide_picks_simplest_within_one_std(tmp_data, monkeypatch):
    from src import decide
    cfgs = {"E1": {"run_id": "E1", "method": "tags_tfidf"}, "E2": {"run_id": "E2", "method": "essentia"},
            "E4": {"run_id": "E4", "method": "concat", "params": {"blocks": [{"method": "tags_lsa"}, {"method": "essentia"}]}}}
    monkeypatch.setattr(decide, "config_for_run", lambda r: paths.ROOT / f"{r}.yaml")
    monkeypatch.setattr(decide, "load_config", lambda p: cfgs[p.stem])
    rows = [
        dict(run_id="E0", method="random", top1_folds="5 5 5 5 5", top1_mean=5, top3_mean=15, top3_std=1, coverage_liked=100, dims=64, featurize_min=0),
        dict(run_id="E1", method="tags_tfidf", top1_folds="20 21 19 22 20", top1_mean=20, top3_mean=50, top3_std=2, coverage_liked=97, dims=900, featurize_min=1),
        dict(run_id="E2", method="essentia", top1_folds="30 31 29 30 30", top1_mean=30, top3_mean=60, top3_std=2, coverage_liked=85, dims=408, featurize_min=40),
        dict(run_id="E4", method="concat", top1_folds="31 32 30 31 31", top1_mean=31, top3_mean=61, top3_std=2, coverage_liked=78, dims=600, featurize_min=60),
    ]
    df = pd.DataFrame(rows).assign(timestamp="t", top1_std=1, macro_f1_mean=0.3)
    df["folds"] = df["top1_folds"].str.split().apply(lambda v: np.array(v, dtype=float))
    d = decide.decide(df)
    assert d["run_id"] == "E2"                      # E4 is within one std but uses more sources
    assert d["fallback"]["run_id"] == "E1"          # E2 covers < 90%; tags reach more songs
    assert "decision" in decide.note(d).lower()


def test_itunes_matching():
    from src.fetch.itunes import best_match
    track = {"name": "Dreams - 2004 Remaster", "artists": ["Fleetwood Mac"], "duration_ms": 257000}
    results = [
        {"trackName": "Dreams", "artistName": "Some Cover Band", "trackTimeMillis": 250000, "previewUrl": "a"},
        {"trackName": "Dreams (2004 Remaster)", "artistName": "Fleetwood Mac", "trackTimeMillis": 257800, "previewUrl": "b"},
        {"trackName": "Dreams", "artistName": "Fleetwood Mac", "trackTimeMillis": 257000},          # no preview
    ]
    assert best_match(track, results)["preview_url"] == "b"
    assert best_match(track, results[:1]) is None
