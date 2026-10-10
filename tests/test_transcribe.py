"""PRD3 M17: beat tracking, drum onsets and MIDI assembly on synthesized audio; the full pipeline with
Demucs / Basic Pitch stubbed out (they are slow, heavy installs; their own projects test them)."""
import json
import shutil

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

from src import paths
from src.midi import beats_key, represent, transcribe

SR = 22050
ROCK = {36: [0, 8, 10], 38: [4, 12], 42: list(range(0, 16, 2))}     # 16th steps in a 4/4 bar


def hit(pitch, rng):
    """A 0.3 s drum sound: pitched-down sine (kick), band noise (snare) or high noise (hat), faded out
    so the end doesn't click."""
    t = np.arange(int(0.3 * SR)) / SR
    fade = np.minimum(1, (t[-1] - t) / 0.05)
    if pitch == 36:
        return np.sin(2 * np.pi * (50 + 60 * np.exp(-t / 0.02)) * t) * np.exp(-t / 0.08) * fade
    lo, hi, decay = (1200, 4500, 0.05) if pitch == 38 else (8000, 10500, 0.02)
    noise = sosfilt(butter(4, [lo, hi], "band", fs=SR, output="sos"), rng.standard_normal(len(t)))
    return 0.6 * noise * np.exp(-t / decay) * fade


def render(pattern, bpm=100.0, bars=8, offset=0.3, seed=0):
    """Audio plus the true (time, pitch) of every hit. The first bar starts at `offset` seconds."""
    rng = np.random.default_rng(seed)
    step = 60 / bpm / 4
    y = np.zeros(int((offset + bars * 16 * step + 1) * SR))
    truth = []
    for b in range(bars):
        for pitch, steps in pattern.items():
            for s in steps:
                t = offset + (b * 16 + s) * step
                h = hit(pitch, rng)
                i = int(t * SR)
                y[i: i + len(h)] += h
                truth.append((t, pitch))
    return y / np.abs(y).max() * 0.8, sorted(truth)


def test_tempo_and_beats_follow_a_drum_loop():
    y, truth = render(ROCK, bpm=100)
    beats, bpm = beats_key.beats_from_audio(y, SR)
    assert abs(bpm - 100) < 2
    true_beats = 0.3 + np.arange(32) * 0.6
    inner = beats[(beats > 1) & (beats < true_beats[-1] - 1)]
    assert len(inner) > 20
    assert np.max(np.min(np.abs(inner[:, None] - true_beats[None, :]), axis=1)) < 0.05      # MIR evaluation uses +-70 ms; frames are 23 ms


@pytest.mark.parametrize("sr,bpm", [(22050, 100), (44100, 100), (44100, 128), (44100, 150)])
def test_drum_onsets_find_kick_snare_hat(sr, bpm, monkeypatch):
    monkeypatch.setitem(globals(), "SR", sr)
    y, truth = render(ROCK, bpm=bpm, offset=0.17, seed=bpm)
    found = transcribe.drum_onsets(y, sr)
    for pitch in (36, 38, 42):
        t_true = np.array([t for t, p in truth if p == pitch])
        t_found = np.array([s for s, _, p, _ in found if p == pitch])
        recall = np.mean([np.min(np.abs(t_found - t)) < 0.03 for t in t_true])
        precision = np.mean([np.min(np.abs(t_true - t)) < 0.03 for t in t_found])
        assert recall >= 0.8 and precision >= 0.8, (pitch, recall, precision)


def test_downbeat_phase_from_kicks():
    beats = 0.3 + np.arange(16) * 0.6
    kicks = beats[2::4]                        # bars start on the third detected beat
    assert beats_key.downbeat_phase(beats, kicks) == 2
    assert beats_key.downbeat_phase(beats, np.array([])) == 0


def test_assembled_midi_carries_the_beat_grid():
    beats = 0.42 + np.cumsum(np.r_[0, np.full(15, 0.5) + np.linspace(0, 0.04, 15)])   # slight ritardando
    parts = {"bass": [(beats[4], beats[5], 40, 90)], "other": [(beats[4], beats[8], 64, 70)], "vocals": []}
    drums = [(b, b + 0.05, 36, 100) for b in beats[::2]]
    pm, grid = transcribe.assemble(beats, float(beats[1]), parts, drums, key=(9, "minor"))
    got = pm.get_beats()
    assert np.allclose(got, grid[: len(got)], atol=2e-3) and len(got) >= len(beats)
    assert np.allclose(grid[np.searchsorted(grid, beats[0] - 1e-6):][:len(beats)], beats, atol=1e-6)
    rep = represent.from_midi(pm, "x")
    assert rep.key_name == "A minor"
    notes = rep.notes.set_index("instrument")
    bass_beat = notes.loc["bass", "start_beat"]
    assert bass_beat == pytest.approx(np.searchsorted(grid, beats[4] - 1e-6), abs=1e-3)
    assert notes.loc["bass", "step_in_bar"] == 12      # downbeat is beats[1]; beats[4] is beat 4 of that bar


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_pipeline_end_to_end_with_stub_models(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    y, _ = render(ROCK, bpm=110, bars=6)
    audio = tmp_path / "song.wav"
    wavfile.write(audio, SR, (y * 32767).astype(np.int16))
    calls = {"separate": 0, "notes": 0}

    def separate(mix, out):
        calls["separate"] += 1
        sr, mono = wavfile.read(mix)
        assert sr == transcribe.SR
        for s in transcribe.STEMS:
            wavfile.write(out / f"{s}.wav", sr, mono if s == "drums" else np.zeros_like(mono))
        return {s: out / f"{s}.wav" for s in transcribe.STEMS}

    def notes(stem, wav):
        calls["notes"] += 1
        beat = 60 / 110
        return [(0.3 + i * beat, 0.3 + (i + 1) * beat - 0.02, 57 + [0, 3, 7][i % 3], 80) for i in range(16)] \
            if stem == "other" else []

    out = transcribe.transcribe(audio, separate=separate, notes_for_stem=notes)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert out == tmp_path / "midi" / "song.mid" and out.exists()
    assert abs(meta["tempo"] - 110) < 3 and meta["n_notes"]["other"] == 16 and meta["n_notes"]["drums"] > 40
    rep = represent.from_path(out)
    assert rep.matrices["drum_grid"][0, 0] > 0.8                 # kick on 1 in most bars
    assert set(rep.notes["instrument"]) == {"other", "drums"}
    assert not (tmp_path / "stems" / "song").exists()            # stems removed by default
    transcribe.transcribe(audio, separate=separate, notes_for_stem=notes)
    assert calls == {"separate": 1, "notes": 3}                  # cached: no second run


# ---------------------------------------------------------------- S6 harness

def test_excerpt_takes_the_30_to_60_second_window():
    from src.midi import gap, synthetic
    pm = synthetic.song("pop", 0, n_bars=60)                     # ~2 min
    ex = gap.excerpt(pm)
    assert ex.get_end_time() <= gap.EXCERPT_S + 1e-6
    starts = [n.start for i in pm.instruments for n in i.notes if 30 <= n.start < 60]
    assert sum(len(i.notes) for i in ex.instruments) == len(starts)
    short = synthetic.song("pop", 0, n_bars=20)                  # ~40 s: last 30 s
    assert gap.excerpt(short).get_end_time() == pytest.approx(gap.EXCERPT_S, abs=0.1)


def test_gap_is_zero_when_transcription_is_perfect():
    from src.midi import experiment, gap, synthetic
    data = synthetic.dataset(per_genre=10, seed=2)
    reps = [represent.from_midi(pm, t).compact() for t, _, pm in data]
    labels = [{g} for _, g, _ in data]
    fold = experiment.make_folds([r.track_id for r in reps])
    pairs = {r.track_id: (r, r) for r in reps[::3]}
    res = gap.evaluate_gap({"model": "markov", "markov": {"sequences": ["chords", "drums"]}}, reps, labels, fold, pairs)
    assert res["gap_top1"] == 0 and res["top1_excerpt"] == res["top1_full"] and res["n_songs"] == len(pairs)
    assert res["key_agreement"] == 100 and res["chord_trans_cos"] == pytest.approx(1)
