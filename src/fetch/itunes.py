"""M4: 30-second audio previews from the iTunes Search API. Resumable.

    python -m src.fetch.itunes                 # match + download every track in library.json
    python -m src.fetch.itunes --limit 50      # quick smoke test
    python -m src.fetch.itunes --spot-check 30 # print 30 random matches to eyeball (PRD risk)

Writes data/raw/itunes.parquet, one row per track:
    track_id, status ("ok" | "no_match" | "error"), itunes_id, preview_url,
    matched_artist, matched_title, title_sim, artist_sim, duration_diff_s, score
and data/raw/audio/<track_id>.m4a for every match.

iTunes allows roughly 20 searches per minute, so a full library takes a couple of hours
across runs. Preview downloads come from a CDN and are not throttled the same way.
"""
from __future__ import annotations

import argparse
import re
import time
from difflib import SequenceMatcher

import pandas as pd
import requests

from src.paths import AUDIO, ITUNES, load_library

SEARCH_URL = "https://itunes.apple.com/search"
SEARCH_INTERVAL_S = 3.2          # ~19 searches / minute
FLUSH_EVERY = 25
MIN_TITLE_SIM = 0.75
MIN_ARTIST_SIM = 0.6
MAX_DURATION_DIFF_S = 15

_PAREN = re.compile(r"[\(\[].*?[\)\]]")
_SUFFIX = re.compile(r"\s+-\s+.*(remaster|version|edit|mix|live|mono|stereo|feat).*$", re.I)
_FEAT = re.compile(r"\b(feat\.?|ft\.?|featuring|with)\b.*$", re.I)
_PUNCT = re.compile(r"[^\w\s]")


def norm_title(s: str) -> str:
    s = _SUFFIX.sub("", s or "")
    s = _PAREN.sub("", s)
    s = _FEAT.sub("", s)
    return " ".join(_PUNCT.sub(" ", s.lower()).split())


def norm_artist(s: str) -> str:
    s = _FEAT.sub("", (s or "").split(",")[0].split("&")[0])
    return " ".join(_PUNCT.sub(" ", s.lower()).split())


def sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def best_match(track: dict, results: list[dict]) -> dict | None:
    """Pick the result that matches on artist + title + duration (PRD mitigation for mismatches)."""
    title = norm_title(track["name"])
    artists = [norm_artist(a) for a in track.get("artists", [])] or [""]
    dur = (track.get("duration_ms") or 0) / 1000
    best = None
    for r in results:
        if not r.get("previewUrl"):
            continue
        ts = sim(title, norm_title(r.get("trackName", "")))
        a_s = max(sim(a, norm_artist(r.get("artistName", ""))) for a in artists)
        dd = abs(dur - r.get("trackTimeMillis", 0) / 1000) if dur and r.get("trackTimeMillis") else None
        dur_score = 1.0 if dd is None else max(0.0, 1 - dd / 30)
        score = 0.5 * ts + 0.35 * a_s + 0.15 * dur_score
        if ts < MIN_TITLE_SIM or a_s < MIN_ARTIST_SIM:
            continue
        if dd is not None and dd > MAX_DURATION_DIFF_S and ts < 0.95:
            continue
        if best is None or score > best["score"]:
            best = {"itunes_id": r.get("trackId"), "preview_url": r["previewUrl"],
                    "matched_artist": r.get("artistName"), "matched_title": r.get("trackName"),
                    "title_sim": round(ts, 3), "artist_sim": round(a_s, 3),
                    "duration_diff_s": None if dd is None else round(dd, 1), "score": round(score, 3)}
    return best


def search(session: requests.Session, term: str) -> list[dict]:
    for attempt in range(5):
        r = session.get(SEARCH_URL, params={"term": term, "entity": "song", "limit": 15}, timeout=20)
        if r.status_code in (403, 429, 503):          # throttled: back off hard
            wait = 60 * (attempt + 1)
            print(f"  throttled (HTTP {r.status_code}); waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json().get("results", [])
    raise RuntimeError("iTunes kept throttling; try again later (progress is saved)")


def download(session: requests.Session, url: str, track_id: str) -> None:
    out = AUDIO / f"{track_id}.m4a"
    if out.exists() and out.stat().st_size > 0:
        return
    r = session.get(url, timeout=30)
    r.raise_for_status()
    tmp = out.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(out)


def spot_check(n: int) -> None:
    df = pd.read_parquet(ITUNES)
    ok = df[df["status"] == "ok"]
    lib = load_library()
    for _, r in ok.sample(min(n, len(ok)), random_state=0).iterrows():
        t = lib["tracks"].get(r["track_id"], {})
        print(f"{r['score']:.2f}  {', '.join(t.get('artists', []))} - {t.get('name')}\n"
              f"      -> {r['matched_artist']} - {r['matched_title']}  (dur diff {r['duration_diff_s']}s)")
    print(f"\nStatus counts:\n{df['status'].value_counts().to_string()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--spot-check", type=int, default=0, metavar="N")
    ap.add_argument("--retry-errors", action="store_true", help="re-try tracks that errored last time")
    args = ap.parse_args()
    if args.spot_check:
        spot_check(args.spot_check)
        return

    lib = load_library()
    cols = ["track_id", "status", "itunes_id", "preview_url", "matched_artist", "matched_title",
            "title_sim", "artist_sim", "duration_diff_s", "score"]
    old = pd.read_parquet(ITUNES) if ITUNES.exists() else pd.DataFrame(columns=cols)
    if args.retry_errors:
        old = old[old["status"] != "error"]
    done = set(old["track_id"])
    todo = [t for t in lib["tracks"] if t not in done][: args.limit]
    print(f"{len(done)} cached, {len(todo)} to fetch (~{len(todo) * SEARCH_INTERVAL_S / 60:.0f} min)")

    session = requests.Session()
    rows, t0 = [], time.time()

    def flush():
        nonlocal old, rows
        if rows:
            old = pd.concat([old, pd.DataFrame(rows, columns=cols)], ignore_index=True)
            old.to_parquet(ITUNES, index=False)
            rows = []

    for i, tid in enumerate(todo, 1):
        tr = lib["tracks"][tid]
        artist = tr["artists"][0] if tr.get("artists") else ""
        t_start = time.time()
        try:
            m = best_match(tr, search(session, f"{artist} {norm_title(tr['name'])}"))
            if m:
                download(session, m["preview_url"], tid)
                rows.append({"track_id": tid, "status": "ok", **m})
            else:
                rows.append({"track_id": tid, "status": "no_match"})
        except requests.RequestException as e:
            print(f"  ! {artist} - {tr['name']}: {e}")
            rows.append({"track_id": tid, "status": "error"})
        if i % FLUSH_EVERY == 0:
            flush()
            print(f"  {i}/{len(todo)}  ({(time.time() - t0) / 60:.1f} min)")
        time.sleep(max(0.0, SEARCH_INTERVAL_S - (time.time() - t_start)))
    flush()
    print(f"Saved {ITUNES}\n{old['status'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
