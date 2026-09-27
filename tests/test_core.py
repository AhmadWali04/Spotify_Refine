"""Sanity checks: hand-written math matches scikit-learn, and the evaluator behaves at the extremes."""
import numpy as np
import pytest
from sklearn.decomposition import PCA, TruncatedSVD

from src.evaluate import chance_rates, evaluate
from src.fetch.synthetic import make
from src.linalg import pca, truncated_svd, zscore


def test_pca_matches_sklearn():
    X = np.random.default_rng(0).standard_normal((200, 12)) @ np.diag(np.arange(1, 13))
    Z, _, evr = pca(X, 3)
    ref = PCA(3).fit(X)
    np.testing.assert_allclose(evr, ref.explained_variance_ratio_, rtol=1e-6)
    # Components match up to sign.
    Zr = ref.transform(X)
    for j in range(3):
        assert np.allclose(Z[:, j], Zr[:, j], atol=1e-6) or np.allclose(Z[:, j], -Zr[:, j], atol=1e-6)


def test_truncated_svd_matches_sklearn():
    X = np.abs(np.random.default_rng(1).standard_normal((100, 30)))
    _, _, S = truncated_svd(X, 5)
    ref = TruncatedSVD(5, algorithm="arpack").fit(X)
    np.testing.assert_allclose(S, ref.singular_values_, rtol=1e-5)


def test_zscore():
    X = np.random.default_rng(2).standard_normal((50, 4)) * 7 + 3
    Z = zscore(X)
    np.testing.assert_allclose(Z.mean(0), 0, atol=1e-10)
    np.testing.assert_allclose(Z.std(0), 1, atol=1e-10)


def test_chance_rates():
    t1, t3 = chance_rates([{0}, {1, 2}], 4)
    assert t1 == pytest.approx((1 / 4 + 2 / 4) / 2)
    assert t3 == pytest.approx((3 / 4 + 1.0) / 2)


@pytest.fixture(scope="module")
def lib():
    return make(n_songs=1500, n_playlists=12, seed=3)


def test_random_vectors_near_chance(lib, tmp_path_factory):
    ids = list(lib["tracks"])
    X = np.random.default_rng(0).standard_normal((len(ids), 64))
    res = evaluate(ids, X, lib, tmp_path_factory.mktemp("f") / "folds.json")
    assert abs(res["top1_mean"] - res["chance_top1"]) < 5


def test_separable_vectors_score_high(lib, tmp_path_factory):
    """Each song placed near a direction for its playlist -> near-perfect accuracy."""
    rng = np.random.default_rng(0)
    P = len(lib["playlists"])
    dirs = rng.standard_normal((P, 32)) * 5
    ids = list(lib["tracks"])
    X = rng.standard_normal((len(ids), 32))
    for j, p in enumerate(lib["playlists"]):
        for t in p["track_ids"]:
            X[ids.index(t)] += dirs[j]
    res = evaluate(ids, X, lib, tmp_path_factory.mktemp("f") / "folds.json")
    assert res["top1_mean"] > 85
    assert res["top3_mean"] > 95


def test_folds_are_stable(lib, tmp_path):
    ids = list(lib["tracks"])
    X = np.random.default_rng(0).standard_normal((len(ids), 8))
    p = tmp_path / "folds.json"
    a = evaluate(ids, X, lib, p)["per_fold"]
    b = evaluate(ids, X, lib, p)["per_fold"]
    assert a == b


def test_tag_featurizers(lib, tmp_path, monkeypatch):
    """E1 / E1b on fake tags: each playlist gets its own tag pool, so tags should beat chance."""
    import pandas as pd
    from src.featurize import tags

    rng = np.random.default_rng(0)
    rows = []
    for j, p in enumerate(lib["playlists"]):
        for t in p["track_ids"]:
            for g in rng.choice(8, size=4, replace=False):
                rows.append({"track_id": t, "tag": f"pl{j}_tag{g}", "weight": int(rng.integers(10, 100)), "source": "track"})
            rows.append({"track_id": t, "tag": "seen live", "weight": 100, "source": "track"})
    path = tmp_path / "tags.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    monkeypatch.setattr(tags, "TAGS", path)

    ids = list(lib["tracks"])
    cov, X = tags.featurize_tfidf(lib, ids, {"min_df": 2})
    assert len(cov) < len(ids)                     # unsorted songs have no tags -> not covered
    cov2, Z = tags.featurize_lsa(lib, ids, {"min_df": 2, "k": 20})
    assert Z.shape == (len(cov2), 20)
    res = evaluate(cov, X, lib, tmp_path / "folds.json")
    assert res["top1_mean"] > 3 * res["chance_top1"]
