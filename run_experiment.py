"""One experiment run: featurize (cached) -> build X -> 5-fold evaluate -> log row + PCA plot.

    python run_experiment.py --config configs/e0_random.yaml
    python run_experiment.py --config configs/e1_tags_tfidf.yaml --force   # ignore vector cache

Test the loop without Spotify:
    python -m src.fetch.synthetic
    python run_experiment.py --config configs/e0_random.yaml --library data/raw/library_synthetic.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src import paths  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.featurize import FEATURIZERS  # noqa: E402
from src.linalg import block_weight, pca, zscore  # noqa: E402

LOG_COLUMNS = [
    "timestamp", "run_id", "method", "dims", "coverage_liked", "coverage_labeled",
    "top1_mean", "top1_std", "top3_mean", "top3_std",
    "knn_top1_mean", "knn_top1_std", "knn_top3_mean", "knn_top3_std",
    "macro_f1_mean", "macro_f1_std", "silhouette",
    "chance_top1", "chance_top3", "majority_top1", "top1_folds",
    "n_playlists", "n_labeled", "n_eval", "featurize_min", "vector_mb",
    "config_hash", "notes",
]


def cached_featurize(cfg: dict, lib: dict, lib_path: Path, force: bool):
    """Vectors live in data/vectors/<run_id>.npy + .index.csv; reused while the config and library match."""
    run_id = cfg["run_id"]
    key = json.dumps({"method": cfg["method"], "params": cfg.get("params", {}),
                      "library": str(lib_path.name), "pulled_at": lib.get("pulled_at")},
                     sort_keys=True)
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    npy = paths.VECTORS / f"{run_id}.npy"
    idx = paths.VECTORS / f"{run_id}.index.csv"
    meta = paths.VECTORS / f"{run_id}.meta.json"

    if not force and npy.exists() and meta.exists() and json.loads(meta.read_text())["hash"] == h:
        m = json.loads(meta.read_text())
        print(f"Using cached vectors {npy.name}")
        return pd.read_csv(idx)["track_id"].tolist(), np.load(npy), m["featurize_min"], h

    fn = FEATURIZERS[cfg["method"]]
    track_ids = list(lib["tracks"])
    t0 = time.time()
    ids, X = fn(lib, track_ids, cfg.get("params", {}))
    minutes = (time.time() - t0) / 60
    np.save(npy, X)
    pd.DataFrame({"track_id": ids}).to_csv(idx, index=False)
    meta.write_text(json.dumps({"hash": h, "featurize_min": minutes, "config": cfg}, indent=1))
    return ids, X, minutes, h


def build_X(X: np.ndarray, cfg: dict) -> np.ndarray:
    if cfg.get("standardize", False):
        X = zscore(X)
        if cfg.get("block_weight", False):
            X = block_weight(X)
    return X


def pca_plot(X: np.ndarray, res: dict, run_id: str, out: Path, top_n: int = 10) -> None:
    row_of = {t: i for i, t in enumerate(res["ids"])}
    Xe = X[[row_of[t] for t in res["eval_ids"]]]
    Z, _, evr = pca(Xe, 2)
    lab = res["primary_label"]
    names = res["playlist_names"]
    top = pd.Series(lab).value_counts().index[:top_n]

    fig, ax = plt.subplots(figsize=(9, 7))
    other = ~np.isin(lab, top)
    ax.scatter(Z[other, 0], Z[other, 1], s=6, c="lightgrey", label="other", alpha=0.5)
    cmap = plt.get_cmap("tab10")
    for c, p in enumerate(top):
        m = lab == p
        ax.scatter(Z[m, 0], Z[m, 1], s=8, color=cmap(c % 10), label=names[p][:30], alpha=0.8)
    ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}% var)")
    ax.set_title(f"{run_id}: top-1 {res['top1_mean']:.1f}%  top-3 {res['top3_mean']:.1f}%")
    ax.legend(fontsize=7, markerscale=2, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def append_log(row: dict, path: Path) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--library", default=str(paths.LIBRARY))
    ap.add_argument("--force", action="store_true", help="recompute vectors even if cached")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    lib_path = Path(args.library)
    lib = paths.load_library(lib_path)
    # Keep synthetic / alternate libraries out of the real log and fold file.
    suffix = "" if lib_path.resolve() == paths.LIBRARY.resolve() else f"_{lib_path.stem}"
    log_path = paths.ROOT / f"experiments{suffix}.csv"
    folds_path = paths.DATA / f"folds{suffix}.json"

    run_id = cfg["run_id"]
    print(f"== {run_id}: {cfg.get('description', cfg['method'])}")
    ids, X_raw, feat_min, h = cached_featurize(cfg, lib, lib_path, args.force)
    X = build_X(X_raw, cfg)
    print(f"Vectors: {X.shape[0]} songs x {X.shape[1]} dims")

    res = evaluate(ids, X, lib, folds_path)
    res["ids"] = ids

    row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()
           if k in LOG_COLUMNS}
    row.update({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id, "method": cfg["method"], "dims": X.shape[1],
        "top1_folds": " ".join(f"{100 * v:.1f}" for v in res["per_fold"]["top1"]),
        "featurize_min": round(feat_min, 2), "vector_mb": round(X_raw.nbytes / 1e6, 2),
        "config_hash": h, "notes": cfg.get("notes", ""),
    })
    append_log(row, log_path)

    plot = paths.REPORTS / f"{run_id}{suffix}_pca.png"
    pca_plot(X, res, run_id, plot)

    print(f"Playlists scored: {res['n_playlists']}   labeled songs: {res['n_labeled']}   "
          f"evaluated: {res['n_eval']}")
    print(f"Coverage: {res['coverage_liked']:.1f}% of liked, {res['coverage_labeled']:.1f}% of labeled")
    print(f"Centroid  top-1 {res['top1_mean']:5.1f} ± {res['top1_std']:.1f}%   "
          f"top-3 {res['top3_mean']:5.1f} ± {res['top3_std']:.1f}%   (chance {res['chance_top1']:.1f} / {res['chance_top3']:.1f})")
    print(f"kNN-10    top-1 {res['knn_top1_mean']:5.1f} ± {res['knn_top1_std']:.1f}%   "
          f"top-3 {res['knn_top3_mean']:5.1f} ± {res['knn_top3_std']:.1f}%   (majority {res['majority_top1']:.1f})")
    print(f"Macro-F1 {res['macro_f1_mean']:.3f} ± {res['macro_f1_std']:.3f}   silhouette {res['silhouette']:.3f}")
    print(f"Logged to {log_path.name}; plot {plot.relative_to(paths.ROOT)}")

    if cfg["method"] == "random":
        gap = abs(res["top1_mean"] - res["chance_top1"])
        if gap > max(3 * res["top1_std"], 2.0):
            print(f"!! E0 top-1 is {gap:.1f} points from chance; check the evaluation for a bug.")
        else:
            print("E0 sits at chance level: evaluation gate passes.")


if __name__ == "__main__":
    main()
