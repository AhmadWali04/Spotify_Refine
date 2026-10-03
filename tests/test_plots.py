"""The PCA / SVD plot command draws every song and every file."""
import numpy as np

from src.fetch.synthetic import make
from src.plots import make_plots, playlist_labels


def test_playlist_labels_cover_every_song():
    lib = make(n_songs=600, n_playlists=8, seed=1)
    ids = list(lib["tracks"])
    labels, order = playlist_labels(lib, ids, top_n=3)
    assert len(labels) == len(ids) and len(order) == 3
    sorted_ids = {t for p in lib["playlists"] for t in p["track_ids"]}
    assert all((lab == "not in a playlist") == (t not in sorted_ids) for t, lab in zip(ids, labels))


def test_make_plots_writes_all_files(tmp_path):
    X = np.random.default_rng(0).standard_normal((120, 10))
    labels = ["a"] * 40 + ["b"] * 40 + ["not in a playlist"] * 40
    files = make_plots(X, labels, ["a", "b"], "T", tmp_path, n_components=8)
    assert {f.name for f in files} == {"pca_scatter.png", "pca_pairs.png", "pca_variance.png",
                                       "svd_scatter.png", "svd_spectrum.png"}
    assert all(f.stat().st_size > 0 for f in files)
