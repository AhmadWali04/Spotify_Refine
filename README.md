# Spotify_Refine
A project that uses Machine Learning to sort and organize your messy Spotify Liked Songs!

It learns what each of your playlists sounds like from the songs already in it, suggests a playlist
for every liked song that isn't in one yet, groups the songs that fit nowhere into candidate new
playlists, and lets you review everything in a local web app before anything is written to Spotify.

- **Phase 1** ([PRD1.MD](PRD1.MD)): pick how to turn a song into a vector, by experiment.
- **Phase 2** ([PHASE2.md](PHASE2.md)): the product built on that choice. Assignment models,
  calibrated suggestions, new-playlist discovery, the review app and safe write-back.
- **Phase 3:** the web front end. Log in with Spotify, run the pipeline from the browser, sort, and
  explore a gallery of visualizations of your taste.

## Quick start (web app)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.app                 # http://127.0.0.1:8888
```

Pick **Explore the demo** to try everything on a synthetic library, or **Log in with Spotify** (needs the
keys below). The Home page runs the three steps (pull your library, fetch Last.fm tags, sort) with live
progress. Then use Sort songs, New playlists and Apply, and open the **Taste gallery**. The app serves
the OAuth callback itself, so it runs on the port in `SPOTIPY_REDIRECT_URI` (8888 by default).

## Try it now, no keys needed

A synthetic library (25 playlists, 3,000 songs, three hidden "new genres") runs the whole product:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.fetch.synthetic
python -m src.sorter.run --library data/raw/library_synthetic.json
python -m src.app --library data/raw/library_synthetic.json      # http://127.0.0.1:8888
```

On the demo, the confident tier is ~97% precise and ~9 in 10 songs from the hidden genres are routed to
new-playlist clusters (`src.sorter.run` prints these checks). Applying is disabled for the demo library.

## Setup for your real library

```bash
cp .env.example .env   # then fill in the keys
```

- **Spotify:** create an app at <https://developer.spotify.com/dashboard>, add the redirect URI `http://127.0.0.1:8888/callback` (it must be `127.0.0.1`, not `localhost`), and copy the Client ID and secret into `.env`. Development Mode needs the app owner to have Premium.
- **Last.fm:** get a free API key at <https://www.last.fm/api/account/create>.
- **Audio (optional, for E2+):** `brew install ffmpeg`, then `pip install essentia-tensorflow` and/or `pip install laion-clap`. Check with `python check_audio_install.py`.

## Workflow

### 1. Pick the featurizer (Phase 1)

```bash
python -m src.fetch.spotify                  # M1: pull the library once (read-only), print playlist sizes
python run_experiment.py --config configs/e0_random.yaml          # M2: baseline, should sit at chance

python -m src.fetch.lastfm                   # M3: tags (resumable; re-run to continue)
python run_experiment.py --config configs/e1_tags_tfidf.yaml
python run_experiment.py --config configs/e1b_tags_lsa.yaml
python run_experiment.py --config configs/e1b_tags_lsa_k50.yaml

python -m src.fetch.itunes                   # M4: 30 s previews, ~3 s per song, resumable
python -m src.fetch.itunes --spot-check 30   #     eyeball 30 matches
python run_experiment.py --config configs/e2_essentia.yaml
python run_experiment.py --config configs/e3_clap.yaml

python run_experiment.py --config configs/e4_concat.yaml          # M5: only if E1-E3 leave a gap
python run_experiment.py --config configs/e5_concat_pca.yaml
python run_experiment.py --config configs/e6_interactions.yaml

python -m src.decide                         # applies the PRD1 decision rules -> data/decision.json + reports/decision.md
```

You can stop after M3: if tags alone pass the gates, `decide` will pick them. If the winner covers under 90%
of liked songs, `decide` also names a fallback featurizer (usually tags) for the rest.

### 2. Sort (Phase 2)

```bash
python -m src.sorter.compare                 # optional: model comparison table (run.py does it automatically)
python -m src.sorter.run                     # suggestions for every unsorted liked song
python -m src.app                            # review at http://127.0.0.1:8888
python -m src.sorter.apply                   # dry run of the reviewed changes (the app has the same button)
python -m src.sorter.apply --yes             # write them to Spotify
python -m src.sorter.apply --undo data/applied/<changelog>.json
```

After applying, re-pull (`python -m src.fetch.spotify`) and re-run the sorter so it learns from the new memberships.
To skip Phase 1, pass a featurizer by hand: `python -m src.sorter.run --config configs/e1b_tags_lsa.yaml`.

## Listening history (the Listening view)

Spotify's API only returns your last 50 plays, so the listening charts read your data export:

1. At <https://www.spotify.com/account/privacy/> request **Extended streaming history** (every play since you
   joined; up to 30 days) and/or **Account data** (last year only; about 5 days).
2. Drop the .zip on the Listening page, or: `python -m src.fetch.history ~/Downloads/my_spotify_data.zip`
   (merges with what's there; `--replace` starts over). Writes `data/raw/history.parquet`.
3. Run the **Tags** step again (`python -m src.fetch.lastfm`): it also tags the 500 artists you play most that
   have no tagged liked songs, so their plays get a genre.

| Chart | What it shows |
| --- | --- |
| Hours over time | Hours per week / month / year for up to 6 artists, albums, songs, genres or playlists, plus all listening |
| Genre mix | Donut of your top genres by hours in any date range (presets or custom dates) |
| Listening calendar | GitHub-style year grid, optionally for one artist / album / song / genre / playlist |
| Then vs. now | Radar of genre shares in two periods, one web each |
| Artist web | Chord diagram: top artists joined by how much their genres overlap, colored by the genre they share most |
| Your taste, overall | Radar of all-time genre shares: what you play (hours) next to what you like (songs) |

A play's genre is its song's strongest Last.fm tag (else its artist's), so genre hours add up to the total.
The last two charts work without a history, from your liked songs. The demo library ships a synthetic history.

## MIDI analysis (PRD3, in progress)

A symbolic view of each song: MIDI -> dataframes and matrices you can inspect -> a Markov or vector
playlist model. Done so far: M13 (representations), and M15/M16 models + toggle on synthetic MIDI.
Still to come: the Lakh dataset (M14), audio -> MIDI transcription (M17), your playlists (M18) and the app page (M19).

```bash
python -m src.midi.inspect song.mid                  # or --synthetic jazz; writes data/inspect/<id>/
python -m src.midi.experiment --data synthetic --suite                   # S1-S5 -> symbolic_synthetic.csv
python -m src.midi.experiment --data labels.csv --symbolic-model vector --assignment A2
```

`configs/symbolic.yaml` picks the model (`model: markov | vector`) and its settings. A labels CSV has
`track_id, path, label` (and optionally `fold`). Key-relative matrices (chords, melody) put the tonic at 0,
so the same progression in any key looks the same. The synthetic genres are easy (tempo and drums give
them away), so they test the plumbing, not the model.

## Plots of your whole library

```bash
python -m src.plots                       # PCA + SVD of every song, colored by playlist
python -m src.plots --color tier          # colored by sorter tier (after src.sorter.run)
python -m src.plots --color cluster       # leftover songs colored by new-playlist cluster
python -m src.plots --config configs/e1b_tags_lsa.yaml --liked-only
```

Writes `reports/plots/<run_id>/`: `pca_scatter.png`, `pca_pairs.png` (PC1-PC4), `pca_variance.png`
(scree + cumulative, marking 50/80/90%), `svd_scatter.png` and `svd_spectrum.png`. PCA centers the songs
first; the SVD is uncentered (as LSA uses it), so its first direction mostly tracks the average song.

## The web app

| View | What you do there |
| --- | --- |
| Landing | Log in with Spotify (OAuth, token cached in `.spotify_cache` and shared with the CLI) or open the demo library. |
| Home | Run pull → tags → sort (each is the CLI command below, run in the background with progress and logs). Until Phase 1 writes `data/decision.json`, sorting uses tags + LSA (`configs/e1b_tags_lsa.yaml`). |
| Sort songs, New playlists, Your playlists, Apply | The review flow below. |
| Listening | Your streaming history: hours over time, genre mix, listening calendar, then-vs-now and overall taste radars, artist web. Needs only a pulled library (see above). |
| Taste gallery | Taste galaxy (PCA map of every song; light up any playlist or new cluster), listening clock (likes by weekday × hour), time machine (release year vs. when you liked it), genre DNA (tag share per playlist), artist orbit, and playlist kinship (centroid similarity, a hint for merges). Every chart has a table view. |

One Spotify account at a time: the app keeps one library on disk. Development-mode Spotify apps allow up
to 5 users, who must be added on the developer dashboard.

### Review views

| Tab | What you do there |
| --- | --- |
| Review | One song at a time: hear it, see the top-3 playlists with probabilities and what each playlist "sounds like", then press <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd>, <kbd>N</kbd> (new playlist), <kbd>S</kbd> (skip), <kbd>Z</kbd> (undo). "Accept all confident" does the easy ones in one click. |
| New playlists | Clusters of leftover songs with a suggested name from their tags. Rename, drop songs, switch on the ones to create. |
| Your playlists | The learned "vibe" of each playlist: distinctive tags, mood axes, top artists. |
| Apply | The exact change list, an Apply button, and undo for every previous apply. |

Songs are sorted into tiers: **confident** (held-out precision ≥ 90% at that confidence), **suggested**,
**leftover** (low confidence or a new kind of music, so it goes to clusters) and **no data** (no tags or audio).

## Adding a featurizer

1. Write `featurize(lib, track_ids, params) -> (covered_ids, X)` in `src/featurize/`.
2. Register it in `src/featurize/__init__.py`.
3. Add a YAML config with a new `run_id`.

Songs the method can't vectorize are left out of `covered_ids`, and that is what the coverage metric measures.

## Layout

```
configs/             one YAML per experiment run (run_id, method, params, standardize); demo_synthetic.yaml for the demo
src/fetch/           spotify.py, lastfm.py, itunes.py, synthetic.py, history.py (streaming-history import)
src/featurize/       random (E0), tags (E1, E1b), essentia (E2), clap (E3), combine (E4-E6), synthetic (demo)
src/linalg.py        z-score, block weighting, PCA and truncated SVD (hand-written)
src/evaluate.py      fixed 5 folds, nearest-centroid and kNN-10 cosine scoring, metrics
src/vectors.py       cached featurization shared by experiments and the sorter
src/decide.py        PRD1 decision rules over experiments.csv
src/plots.py         PCA / SVD plots of every song in a space
src/midi/            represent (dataframes + matrices), chords, beats_key, synthetic MIDI, inspect, experiment
src/models/symbolic/ base (toggle), markov, vector (wraps the src/sorter models)
src/sorter/          models, calibrate, compare, novelty, cluster, describe, run, review, apply
src/app/             Flask API (server.py), Spotify login (auth.py), background pipeline steps (jobs.py), listening charts (listening.py),
                     gallery data (insights.py), single-page front end (static/: app.js, review.js, gallery.js)
run_experiment.py    experiment loop entry point
tests/               math vs scikit-learn, evaluator at chance / perfect separation, Phase 2 end to end
```

## Reading the logs

`experiments.csv` (one row per featurizer run):

| Column | Meaning |
| --- | --- |
| `top1_mean`, `top3_mean` | Nearest-centroid accuracy (primary), mean ± std over 5 folds |
| `knn_top1_mean`, `knn_top3_mean` | kNN-10 accuracy (second view) |
| `chance_top1`, `chance_top3` | Expected accuracy if playlists were ranked at random |
| `majority_top1` | Accuracy of always guessing the biggest playlist; weak vectors push kNN toward this number |
| `top1_folds` | Per-fold top-1, used to check "at least 2x E0 on all 5 folds" |
| `coverage_liked` | Share of liked songs the method can vectorize (the 90% gate) |

`models.csv` (one row per assignment model per space): the same top-1 / top-3 / macro-F1 on the same folds,
plus held-out `nll_mean` and `ece_mean` (calibration error after temperature scaling) and which model was `chosen`.

Synthetic runs log to `*_library_synthetic.*` files, so they never mix with real results.
