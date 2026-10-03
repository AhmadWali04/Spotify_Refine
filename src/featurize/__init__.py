"""Featurizer registry.

Every featurizer has the signature
    fn(lib: dict, track_ids: list[str], params: dict) -> (covered_ids: list[str], X: np.ndarray)
where X has one row per covered id, in the same order. Songs a method cannot
vectorize are simply left out of covered_ids; that is what "coverage" measures.

Audio featurizers import essentia / laion_clap lazily, so the registry loads without them.
"""
from __future__ import annotations

from src.featurize import clap_feats, combine, essentia_feats, random_feats, synthetic_feats, tags

FEATURIZERS = {
    "random": random_feats.featurize,
    "tags_tfidf": tags.featurize_tfidf,
    "tags_lsa": tags.featurize_lsa,
    "essentia": essentia_feats.featurize,
    "clap": clap_feats.featurize,
    "concat": combine.featurize_concat,
    "concat_pca": combine.featurize_concat_pca,
    "interactions": combine.featurize_interactions,
    "synthetic": synthetic_feats.featurize,
}

# How many sources (tags / essentia / clap) a method uses; the PRD tiebreak prefers fewer.
AUDIO_METHODS = {"essentia", "clap"}
EXPLAINABLE_METHODS = {"essentia", "tags_tfidf", "tags_lsa"}
