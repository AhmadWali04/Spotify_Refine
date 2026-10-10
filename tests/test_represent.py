"""PRD3 representations: the matrix identities, chord/key estimation on known MIDI, and the inspect output."""
import numpy as np
import pretty_midi
import pytest

from src.midi import chords, represent, synthetic
from src.midi.inspect import inspect


def triads(progression, tonic=0, tempo=120.0, minor=False, drums=True, key_sig=None):
    """One chord per bar (absolute roots), kick on 1 and 3, snare on 2 and 4, hats on 8ths."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    if key_sig is not None:
        pm.key_signature_changes.append(pretty_midi.KeySignature(key_sig, 0.0))
    pad = pretty_midi.Instrument(0, name="other")
    kit = pretty_midi.Instrument(0, is_drum=True, name="drums")
    beat = 60 / tempo
    for bar, (root, is_minor) in enumerate(progression):
        t0 = bar * 4 * beat
        for iv in (0, 3 if is_minor else 4, 7):
            pad.notes.append(pretty_midi.Note(80, 48 + root + tonic + iv, t0, t0 + 4 * beat - 0.01))
        if drums:
            for s, p in [(0, 36), (8, 36), (4, 38), (12, 38)] + [(s, 42) for s in range(0, 16, 2)]:
                kit.notes.append(pretty_midi.Note(100, p, t0 + s * beat / 4, t0 + s * beat / 4 + 0.05))
    pm.instruments += [pad] + ([kit] if drums else [])
    return pm


POP = [(0, 0), (7, 0), (9, 1), (5, 0)] * 2          # I V vi IV, twice


def test_chroma_is_fold_of_piano_roll():
    rep = represent.from_midi(synthetic.song("jazz", 1), "x")
    m = rep.matrices
    assert np.array_equal(m["chroma"], represent.fold_matrix() @ m["piano_roll"])
    assert np.array_equal(m["chroma_keynorm"], np.roll(m["chroma"], -rep.tonic, axis=0))


def test_chord_templates_and_names():
    T = chords.templates()
    assert np.allclose(np.linalg.norm(T, axis=1), 1)
    c_major = np.zeros(12)
    c_major[[0, 4, 7]] = 1
    a_minor = np.zeros(12)
    a_minor[[9, 0, 4]] = 1
    assert chords.estimate(np.array([c_major, a_minor, np.zeros(12)])).tolist() == [0, 21, chords.NO_CHORD]
    assert chords.name(21) == "Am" and chords.roman(21) == "vi" and chords.roman(10) == "bVII"
    assert chords.transpose(chords.NO_CHORD, 5) == chords.NO_CHORD


def test_known_progression_chords_key_and_drums():
    rep = represent.from_midi(triads(POP, key_sig=0), "pop")
    assert rep.key_name == "C major"
    per_bar = rep.beats.groupby("bar")["chord"].first().tolist()
    assert per_bar == ["C", "G", "Am", "F"] * 2
    C = rep.matrices["chord_trans"]
    assert C[0, 7] == 2 and C[7, 21] == 2 and C[21, 5] == 2 and C[5, 0] == 1
    assert C.sum() == len(rep.beats) - 1
    grid = rep.matrices["drum_grid"]
    assert grid[0, [0, 8]].tolist() == [1, 1] and grid[1, [4, 12]].tolist() == [1, 1]
    assert grid[2, ::2].tolist() == [1] * 8 and grid[2, 1::2].sum() == 0
    assert np.allclose(rep.matrices["drum_grid_var"], 0)          # the same groove every bar
    assert rep.notes.loc[rep.notes.is_drum, "pitch_class"].eq(-1).all()


def test_krumhansl_finds_the_key_without_a_signature():
    for tonic in (0, 2, 7):
        rep = represent.from_midi(triads(POP, tonic=tonic), "x", use_key_signature=False)
        assert (rep.tonic, rep.mode) == (tonic, "major")


def test_key_relative_features_are_transposition_invariant():
    a = represent.from_midi(triads(POP, tonic=0, key_sig=0), "a")
    b = represent.from_midi(triads(POP, tonic=3, key_sig=3), "b")
    for k in ("chord_trans", "pc_trans", "chroma_dft_mag", "interval_hist", "drum_grid"):
        assert np.allclose(a.matrices[k], b.matrices[k]), k
    assert np.array_equal(a.sequences["chords"], b.sequences["chords"])
    assert not np.allclose(a.matrices["chroma"], b.matrices["chroma"])


def test_chroma_dft_magnitude_ignores_circular_shift():
    rng = np.random.default_rng(0)
    v = rng.random(12)
    mags = [np.abs(np.fft.fft(np.roll(v, k)))[:7] for k in range(12)]
    assert all(np.allclose(m, mags[0]) for m in mags)


def test_transpose_matrices_matches_shifted_sequences():
    rep = represent.from_midi(synthetic.song("pop", 3), "x")
    for k in (1, 5, 11):
        t = represent.transpose_matrices(rep.matrices, k)
        assert np.array_equal(t["chord_trans"], represent.transition_counts(t["chord_seq"], chords.N_CHORDS))
        assert t["chord_trans"].sum() == rep.matrices["chord_trans"].sum()
    back = represent.transpose_matrices(represent.transpose_matrices(rep.matrices, 5), 7)
    assert np.array_equal(back["chord_trans"], rep.matrices["chord_trans"])
    assert np.array_equal(back["pc_trans"], rep.matrices["pc_trans"])


def test_transition_counts_order_2():
    C = represent.transition_counts(np.array([0, 1, 2, 0, 1, 2]), 3, order=2)
    assert C.shape == (9, 3) and C.sum() == 4
    assert C[0 * 3 + 1, 2] == 2 and C[1 * 3 + 2, 0] == 1


def test_dataframes_have_prd_columns():
    rep = represent.from_midi(synthetic.song("edm", 0), "x")
    assert list(rep.notes.columns) == ["track_id", "instrument", "program", "is_drum", "pitch", "pitch_name",
                                       "pitch_class", "start_s", "end_s", "start_beat", "duration_beats",
                                       "bar", "step_in_bar", "velocity"]
    for c in ["beat", "bar", "chord", "chord_root", "chord_quality", "n_active_notes", "c0", "c11",
              "kick", "snare", "hat"]:
        assert c in rep.beats.columns
    assert len(rep.summary) == 1 and "chord_trans_0" in rep.summary and "tempo" in rep.summary


def test_empty_midi_is_rejected():
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(pretty_midi.Instrument(0))
    pm.instruments[0].notes.append(pretty_midi.Note(80, 60, 0, 0.0))
    pm.instruments[0].notes.clear()
    with pytest.raises(ValueError):
        represent.from_midi(pm, "empty")


def test_inspect_writes_every_output(tmp_path):
    path = tmp_path / "song.mid"
    triads(POP).write(str(path))
    out = inspect(path, out_root=tmp_path / "inspect", quiet=True)
    for f in ["notes.csv", "beats.csv", "summary.csv", "matrices.npz", "piano_roll.png", "chroma.png",
              "chord_trans.png", "drum_grid.png", "ssm.png", "transcribed.mid"]:
        assert (out / f).exists(), f
    assert {"piano_roll", "chroma", "chord_trans", "ssm"} <= set(np.load(out / "matrices.npz").files)
