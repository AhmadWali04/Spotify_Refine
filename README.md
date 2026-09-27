# Spotify_Refine
A project that uses Machine Learning to sort and organize your messy Spotify Liked Songs!

**Current phase:** Phase 1, song vectorization ([PRD1.MD](PRD1.MD)). We test several ways to turn a song into a vector and measure how well each one predicts which playlist a song belongs to.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the keys
```

- **Spotify:** create an app at <https://developer.spotify.com/dashboard>, add the redirect URI `http://127.0.0.1:8888/callback` (it must be `127.0.0.1`, not `localhost`), and copy the Client ID and secret into `.env`.
- **Last.fm:** get a free API key at <https://www.last.fm/api/account/create>.

## Workflow

```bash
# M1: pull the library once (read-only) and print playlist sizes
python -m src.fetch.spotify
python check_audio_install.py         # test the Essentia / CLAP installs early (needed at M4)

# M2: baseline. E0 top-1 should sit near chance
python run_experiment.py --config configs/e0_random.yaml

# M3: tag methods
python -m src.fetch.lastfm            # resumable; re-run to continue
python run_experiment.py --config configs/e1_tags_tfidf.yaml
python run_experiment.py --config configs/e1b_tags_lsa.yaml
python run_experiment.py --config configs/e1b_tags_lsa_k50.yaml
```

Each run appends one row to `experiments.csv` and saves `reports/<run_id>_pca.png`.
Vectors are cached in `data/vectors/`, and a run only recomputes them when its config or the library changes (`--force` recomputes anyway).

**No Spotify login yet?** You can test the loop on a fake library:

```bash
python -m src.fetch.synthetic
python run_experiment.py --config configs/e0_random.yaml --library data/raw/library_synthetic.json
```

Synthetic runs log to `experiments_library_synthetic.csv`, so they never mix with real results.

## Adding a method

1. Write `featurize(lib, track_ids, params) -> (covered_ids, X)` in `src/featurize/`.
2. Register it in `src/featurize/__init__.py`.
3. Add a YAML config with a new `run_id`.

Songs the method can't vectorize are left out of `covered_ids`, and that is what the coverage metric measures.

## Layout

```
configs/           one YAML per run (run_id, method, params, standardize)
src/fetch/         spotify.py, lastfm.py, synthetic.py
src/featurize/     random_feats.py (E0), tags.py (E1, E1b)
src/linalg.py      z-score, block weighting, PCA and truncated SVD (hand-written)
src/evaluate.py    fixed 5 folds, nearest-centroid and kNN-10 cosine scoring, metrics
run_experiment.py  loop entry point
tests/             checks the math against scikit-learn and the evaluator at chance and at perfect separation
```

## Reading the log

| Column | Meaning |
| --- | --- |
| `top1_mean`, `top3_mean` | Nearest-centroid accuracy (primary), mean ± std over 5 folds |
| `knn_top1_mean`, `knn_top3_mean` | kNN-10 accuracy (second view) |
| `chance_top1`, `chance_top3` | Expected accuracy if playlists were ranked at random |
| `majority_top1` | Accuracy of always guessing the biggest playlist; weak vectors push kNN toward this number |
| `top1_folds` | Per-fold top-1, used to check "at least 2x E0 on all 5 folds" |
| `coverage_liked` | Share of liked songs the method can vectorize (the 90% gate) |
