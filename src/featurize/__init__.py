"""Featurizer registry.

Every featurizer has the signature
    fn(lib: dict, track_ids: list[str], params: dict) -> (covered_ids: list[str], X: np.ndarray)
where X has one row per covered id, in the same order. Songs a method cannot
vectorize are simply left out of covered_ids; that is what "coverage" measures.
"""
from __future__ import annotations

from src.featurize import random_feats, tags

FEATURIZERS = {
    "random": random_feats.featurize,
    "tags_tfidf": tags.featurize_tfidf,
    "tags_lsa": tags.featurize_lsa,
    # M4: "essentia": essentia_feats.featurize, "clap": clap_feats.featurize
    # M5: "concat": combine.featurize, ...
}
