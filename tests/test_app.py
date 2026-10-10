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


# ---------------------------------------------------------------- listening history

def _export_zip(lib) -> bytes:
    """A small privacy export with both formats overlapping on one play, plus a podcast and a video file."""
    import io
    import zipfile
    t = lib["tracks"]
    tid = next(iter(t))
    ext = [
        {"ts": "2025-03-01T20:15:42Z", "ms_played": 200000, "master_metadata_track_name": t[tid]["name"],
         "master_metadata_album_artist_name": t[tid]["artists"][0], "master_metadata_album_album_name": t[tid]["album"],
         "spotify_track_uri": f"spotify:track:{tid}"},
        {"ts": "2025-03-02T08:00:00Z", "ms_played": 60000, "master_metadata_track_name": "Unknown Song",
         "master_metadata_album_artist_name": "Nobody", "master_metadata_album_album_name": "Nothing",
         "spotify_track_uri": "spotify:track:zzz"},
        {"ts": "2025-03-02T09:00:00Z", "ms_played": 900000, "master_metadata_track_name": None,
         "episode_name": "A podcast", "spotify_track_uri": None},
    ]
    acct = [{"endTime": "2025-03-01 20:15", "artistName": t[tid]["artists"][0], "trackName": t[tid]["name"], "msPlayed": 200000}]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Spotify Extended Streaming History/Streaming_History_Audio_2025.json", json.dumps(ext))
        z.writestr("Spotify Extended Streaming History/Streaming_History_Video_2025.json", json.dumps(ext))
        z.writestr("Spotify Account Data/StreamingHistory_music_0.json", json.dumps(acct))
        z.writestr("Spotify Account Data/StreamingHistory_podcast_0.json", json.dumps([{"podcastName": "x"}]))
    return buf.getvalue(), tid


def test_history_import_merges_formats(demo_app):
    from src.fetch import history
    lib, lib_path, _ = demo_app
    data, tid = _export_zip(lib)
    df, used = history.read_files([("my_spotify_data.zip", data)])
    assert sorted(used) == ["StreamingHistory_music_0.json", "Streaming_History_Audio_2025.json"]
    assert len(df) == 2                                      # same play in both formats counted once; podcast dropped
    assert df["ts"].dt.tz is not None and df["ms"].sum() == 260000
    df = history.link_tracks(df, lib)
    assert df.loc[df["track"] == lib["tracks"][tid]["name"], "track_id"].iloc[0] == tid
    assert history.parse_json("[]").empty


def test_upload_history_only_for_real_library(demo_app):
    import io
    lib, lib_path, c = demo_app
    data, _ = _export_zip(lib)
    r = c.post("/api/history", data={"files": (io.BytesIO(data), "export.zip")}, content_type="multipart/form-data")
    assert r.status_code == 400 and "demo" in r.get_json()["error"]


def test_listening_endpoints(demo_app):
    lib, lib_path, c = demo_app
    m = c.get("/api/listening/meta?tz=America/Toronto").get_json()
    assert m["ok"] and m["has_history"] and m["hours"] > 0 and m["tagged"] == 1.0
    assert paths.history_path(lib_path).exists()             # demo history generated on first use
    assert set(m["items"]) == {"artist", "album", "track", "genre", "playlist"}
    assert len(m["genre_rank"]) >= 8

    s = c.get("/api/listening/series?dim=genre&bucket=month").get_json()
    assert len(s["series"]) == 3 and all(len(x["hours"]) == len(s["periods"]) for x in s["series"])
    pl = m["items"]["playlist"][0][0]
    s = c.get("/api/listening/series", query_string={"dim": "playlist", "bucket": "year", "item": [pl, "__all__"]}).get_json()
    assert [x["key"] for x in s["series"]] == [pl, "__all__"]
    assert abs(sum(s["series"][1]["hours"]) - m["hours"]) < 0.5
    assert sum(s["series"][0]["hours"]) <= sum(s["series"][1]["hours"])
    assert c.get("/api/listening/series?dim=nope").status_code == 400

    sh = c.get(f"/api/listening/share?start={m['first']}&end={m['last']}").get_json()
    assert abs(sum(g[1] for g in sh["genres"]) - sh["total"]) < 0.5 and sh["untagged"] == 0

    year = m["years"][-1]
    cal = c.get(f"/api/listening/calendar?year={year}").get_json()
    assert all(d[0].startswith(str(year)) for d in cal["days"]) and cal["days"][0][2]
    artist = m["items"]["artist"][0][0]
    one = c.get("/api/listening/calendar", query_string={"year": year, "dim": "artist", "item": artist}).get_json()
    assert one["total"] < cal["total"] and one["top_kind"] == "track"

    r = c.get(f"/api/listening/radar?a_start={year - 1}-01-01&a_end={year - 1}-12-31&b_start={year}-01-01&b_end={year}-12-31").get_json()
    assert len(r["axes"]) == 8 and len(r["webs"]) == 2 and all(len(w["values"]) == 8 for w in r["webs"])
    o = c.get("/api/listening/overall").get_json()
    assert [w["label"] for w in o["webs"]] == ["What you play", "What you like"]

    ch = c.get("/api/listening/chord").get_json()
    assert ch["links"] and all(0 < l["w"] <= 1.0001 and l["s"] < l["t"] < len(ch["artists"]) for l in ch["links"])
    assert c.get("/api/listening/search?dim=artist&q=" + artist[:4]).get_json()["items"]


def test_listening_without_history_still_has_radar_and_chord(demo_app):
    from src.app.listening import Listening
    lib, lib_path, _ = demo_app
    L = Listening(lib, lib_path)                              # direct: no demo history generated
    assert not L.has_history and not L.meta()["has_history"]
    assert [w["label"] for w in L.overall()["webs"]] == ["What you like"]
    assert L.chord()["unit"] == "songs"
    with pytest.raises(FileNotFoundError):
        L.series("artist", "month", [])


def test_history_save_merges_without_duplicates(demo_app):
    from src.fetch import history
    lib, lib_path, _ = demo_app
    data, _ = _export_zip(lib)
    df, _ = history.read_files([("a.zip", data)])
    history.save(df, lib_path, replace=True)
    again = history.save(df, lib_path)                       # importing the same export twice
    assert len(again) == 2
    assert len(history.save(df.iloc[:1], lib_path, replace=True)) == 1
