"""PRD3 symbolic models: Markov math on toy sequences, the vector model, the config toggle and the harness."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.midi import experiment, represent, synthetic
from src.models.symbolic import base
from src.models.symbolic.markov import MarkovModel
from src.models.symbolic.vector import VectorModel


def toy(seq, s="chords"):
    """A stand-in SongRep with just the sequences the Markov model reads."""
    return SimpleNamespace(sequences={"chords": np.zeros(0, int), "pitch_classes": np.zeros(0, int),
                                      "drums": np.zeros(0, int), s: np.array(seq)})


UP = [0, 7, 21, 5] * 6      # I V vi IV
DOWN = [5, 21, 7, 0] * 6    # the same chords, other direction


@pytest.mark.parametrize("order", [1, 2])
def test_rows_sum_to_one(order):
    m = MarkovModel(order=order).fit([toy(UP), toy(DOWN)], [{"a"}, {"b"}])
    assert m.logP["chords"].shape == (2, 25 ** order, 25)
    assert np.allclose(np.exp(m.logP["chords"]).sum(axis=-1), 1)
    assert np.allclose(np.exp(m.logBG["chords"]).sum(axis=-1), 1)


def test_likelihood_matches_formula():
    m = MarkovModel(alpha=0.5, prior="uniform").fit([toy([0, 7, 0, 7])], [{"a"}])
    # counts: 0->7 twice, 7->0 once; K = 25
    p07, p70 = (2 + 0.5) / (2 + 25 * 0.5), (1 + 0.5) / (1 + 25 * 0.5)
    expected = np.mean(np.log([p07, p70, p07])) + np.log(1.0)
    assert np.isclose(m.scores([toy([0, 7, 0, 7])])[0, 0], expected)


@pytest.mark.parametrize("order", [1, 2])
def test_direction_is_learned(order):
    m = MarkovModel(order=order).fit([toy(UP)] * 3 + [toy(DOWN)] * 3, [{"up"}] * 3 + [{"down"}] * 3)
    P = m.predict_proba([toy(UP[:9]), toy(DOWN[:9])])
    assert m.classes_ == ["down", "up"]
    assert P[0].argmax() == 1 and P[1].argmax() == 0
    assert np.allclose(P.sum(axis=1), 1)


def test_songs_without_a_sequence_fall_back_to_the_prior():
    m = MarkovModel(sequences=["chords", "drums"], prior="size").fit([toy(UP), toy(UP), toy(DOWN)], [{"a"}, {"a"}, {"b"}])
    S = m.scores([toy([])])
    assert np.allclose(S[0], m.log_prior) and S[0, 0] > S[0, 1]


def test_explain_names_the_characteristic_transitions():
    m = MarkovModel().fit([toy(UP)] * 3 + [toy(DOWN)] * 3, [{"up"}] * 3 + [{"down"}] * 3)
    why = m.explain(toy(UP[:8]), "up", k=3)
    assert len(why) == 3 and all(r["ratio"] > 1 for r in why)
    assert {(r["from"], r["to"]) for r in why} <= {("I", "V"), ("V", "vi"), ("vi", "IV"), ("IV", "I")}


@pytest.fixture(scope="module")
def synth():
    data = synthetic.dataset(per_genre=12, seed=1)
    return [represent.from_midi(pm, t) for t, _, pm in data], [{g} for _, g, _ in data]


@pytest.mark.parametrize("cfg", [
    {"model": "markov", "markov": {"sequences": ["chords", "pitch_classes", "drums"]}},
    {"model": "vector", "vector": {"assignment_model": "A0"}, "augment": {"transpose": True}},
    {"model": "vector", "vector": {"assignment_model": "A2", "pca_k": 16}},
])
def test_toggle_and_harness_beat_chance(synth, cfg):
    reps, labels = synth
    res, S, classes = experiment.evaluate(cfg, reps, labels, experiment.make_folds([r.track_id for r in reps]))
    assert S.shape == (len(reps), len(synthetic.GENRES)) and classes == sorted(synthetic.GENRES)
    assert res["top1_mean"] > 2 * res["majority_top1"]
    assert np.isfinite(res["nll_mean"]) and 0 <= res["ece_mean"] <= 1


def test_config_toggle_switches_model_only():
    cfg = base.load_config()
    assert isinstance(base.make(cfg | {"model": "markov"}), MarkovModel)
    v = base.make(base.load_config(model="vector", assignment="A3"))
    assert isinstance(v, VectorModel) and v.assignment == "knn10"
    with pytest.raises(ValueError):
        base.make(cfg | {"model": "sheet_music"})


def test_vector_block_order_is_respected(synth):
    reps, labels = synth
    m = VectorModel("A0", blocks=["scalars", "chroma_dft_mag"]).fit(reps, labels)
    assert m.block_scale[:7].tolist() == pytest.approx([1 / np.sqrt(7)] * 7)
    assert m.raw(reps[:2]).shape == (2, 14)


def test_folds_are_balanced_and_deterministic():
    ids = [f"s{i}" for i in range(23)]
    f1, f2 = experiment.make_folds(ids), experiment.make_folds(list(reversed(ids)))
    assert np.bincount(f1).tolist() == [5, 5, 5, 4, 4]
    assert dict(zip(ids, f1)) == dict(zip(reversed(ids), f2))
