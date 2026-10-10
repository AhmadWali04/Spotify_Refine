"""Audio -> MIDI (PRD3 FR-M1 to FR-M6).

    python -m src.midi.transcribe song.m4a                    # -> data/midi/<stem>.mid + .json
    python -m src.midi.transcribe --previews --limit 50       # iTunes previews in data/raw/audio/
    python -m src.midi.inspect data/midi/<id>.mid             # look at the result

  1. ffmpeg: decode anything (mp3 / m4a / wav / mp4) to mono 44.1 kHz WAV        FR-M1
  2. Demucs htdemucs: vocals / drums / bass / other stems in data/stems/ (--keep-stems) FR-M2
  3. Basic Pitch: bass, other and vocals stems to notes, one instrument each        FR-M3
  4. Drums: band-split onset detection on the drum stem -> GM 36 / 38 / 42          FR-M4
  5. Beats from the mix, downbeats from kicks, key from the notes                   FR-M5
  6. One MIDI whose tempo map is the beat grid, plus <id>.json metadata; cached     FR-M6

Steps 2 and 3 are passed in as functions (`separate`, `notes_for_stem`), so tests and other backends can
swap them. Install the real ones with `pip install demucs basic-pitch`; ffmpeg via `brew install ffmpeg`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pretty_midi
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from src import paths
from src.midi import beats_key

SR = 44100
STEMS = ("vocals", "drums", "bass", "other")
PITCHED = {"bass": 33, "other": 0, "vocals": 53}            # GM programs: fingered bass, piano, voice oohs
# Basic Pitch frequency limits per stem keep, e.g., hi-hat bleed out of the bass line.
PITCH_RANGE = {"bass": (30.0, 400.0), "other": (60.0, 4200.0), "vocals": (80.0, 1100.0)}
MIN_GAP = 0.06          # s between hits on one drum: a 16th at 250 bpm; closer peaks are one hit's ringing
REL_STRENGTH = 0.3      # min peak height relative to the band's 90th-percentile hit
BLEED_WINDOW = 0.025    # s: a peak this close to a lower drum's hit ...
BLEED_STRENGTH = 0.5    # ... and weaker than this (relative) is that hit's bleed
DRUM_BANDS = {36: (20.0, 150.0), 38: (1000.0, 5000.0), 42: (7000.0, 20000.0)}   # kick, snare, hat (Hz)

NoteList = list[tuple[float, float, int, int]]               # (start_s, end_s, pitch, velocity)


def stems_dir() -> Path:
    return paths.DATA / "stems"


def midi_dir() -> Path:
    return paths.DATA / "midi"


# ---------------------------------------------------------------- 1. decode

def decode(src: Path, out: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found; `brew install ffmpeg`")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-vn",
                    "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(out)], check=True)
    return out


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Mono float in [-1, 1]."""
    sr, y = wavfile.read(path)
    scale = float(np.iinfo(y.dtype).max) + 1 if np.issubdtype(y.dtype, np.integer) else 1.0
    y = y.astype(np.float32) / scale
    return (y.mean(axis=1) if y.ndim == 2 else y), sr


# ---------------------------------------------------------------- 2. separate (Demucs)

def demucs_separate(mix: Path, out: Path) -> dict[str, Path]:
    """htdemucs via its CLI (stable across demucs versions); stems written as out/<stem>.wav."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs", "-o", tmp,
                        "--filename", "{stem}.{ext}", str(mix)], check=True,
                       stdout=subprocess.DEVNULL)
        found = {p.stem: p for p in Path(tmp).rglob("*.wav")}
        missing = set(STEMS) - set(found)
        if missing:
            raise RuntimeError(f"demucs wrote no {sorted(missing)} stems")
        for s in STEMS:
            shutil.move(found[s], out / f"{s}.wav")
    return {s: out / f"{s}.wav" for s in STEMS}


# ---------------------------------------------------------------- 3. pitched stems (Basic Pitch)

_BP_MODEL = None


def basic_pitch_notes(stem: str, wav: Path) -> NoteList:
    global _BP_MODEL
    import logging
    logging.getLogger().setLevel(logging.ERROR)     # basic_pitch warns about every backend it doesn't find
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict
    if _BP_MODEL is None:
        _BP_MODEL = Model(ICASSP_2022_MODEL_PATH)
    lo, hi = PITCH_RANGE[stem]
    _, midi, _ = predict(str(wav), _BP_MODEL, minimum_frequency=lo, maximum_frequency=hi,
                         minimum_note_length=80 if stem == "vocals" else 58)
    return [(n.start, n.end, n.pitch, n.velocity) for inst in midi.instruments for n in inst.notes]


# ---------------------------------------------------------------- 4. drums

def pick_peaks(env: np.ndarray, fps: float, delta: float = 0.08, wait_s: float = 0.05) -> np.ndarray:
    """Frames that are a local max within +-wait_s, above a moving median by delta (env scaled to [0, 1])."""
    w = max(1, int(round(wait_s * fps)))
    local = env == maximum_filter1d(env, 2 * w + 1)
    thresh = median_filter(env, size=max(3, int(round(0.5 * fps)) | 1)) + delta
    frames = np.flatnonzero(local & (env > thresh))
    keep = []                                   # enforce the minimum gap between onsets
    for f in frames:
        if not keep or f - keep[-1] >= w:
            keep.append(f)
    return np.array(keep, dtype=int)


def refine_onset(amp: np.ndarray, sr: int, t: float, win: float) -> float:
    """Sharpen a frame-level onset to the sample where the band's amplitude envelope rises through
    half its local range (STFT frames alone are tens of ms off)."""
    a, b = max(0, int((t - win) * sr)), min(len(amp), int((t + win) * sr))
    seg = amp[a:b]
    if len(seg) < 2 or seg.max() <= 0:
        return t
    peak = int(np.argmax(seg))
    floor_i = int(np.argmin(seg[: peak + 1]))
    level = seg[floor_i] + 0.5 * (seg[peak] - seg[floor_i])
    return (a + floor_i + int(np.argmax(seg[floor_i: peak + 1] >= level))) / sr


def drum_onsets(y: np.ndarray, sr: int) -> NoteList:
    """Kick / snare / hat hits from the drum stem: onset envelope per frequency band, peak picking,
    then each onset refined in the time domain on the band-filtered signal.

    Bands are done low to high, and a weak peak that coincides with a hit already found in a lower
    band is that hit's bleed (a kick's beater click reaches the snare band), so it is dropped."""
    n_fft = 2048 if sr > 30000 else 1024
    S, freqs = beats_key.stft_mag(y, sr, n_fft=n_fft)
    fps = sr / beats_key.HOP
    out: NoteList = []
    for pitch, (lo, hi) in DRUM_BANDS.items():
        hi = min(hi, 0.45 * sr)
        if lo >= hi:
            continue
        env = beats_key.onset_envelope(S, freqs, lo, hi)
        sos = butter(4, [max(lo, 20.0), hi], "band", fs=sr, output="sos")
        amp = uniform_filter1d(np.abs(sosfiltfilt(sos, y)), max(1, int(0.002 * sr)))
        peaks = pick_peaks(env, fps)
        if not len(peaks):
            continue
        rel = env[peaks] / np.percentile(env[peaks], 90)       # 1 = this drum's typical strong hit
        lower = np.array([s for s, _, _, _ in out])
        times = []
        for f, r in zip(peaks, rel):
            t = refine_onset(amp, sr, f / fps, n_fft / sr)
            bleed = len(lower) and np.min(np.abs(lower - t)) < BLEED_WINDOW and r < BLEED_STRENGTH
            if r < REL_STRENGTH or bleed or (times and t - times[-1] <= MIN_GAP):
                continue
            times.append(t)
            out.append((t, t + 0.05, pitch, int(np.clip(40 + 87 * env[f], 1, 127))))
    return sorted(out)


# ---------------------------------------------------------------- 6. assemble

def _grid_covering(beats: np.ndarray, end: float) -> np.ndarray:
    """Beat times starting at 0 (a short pickup beat if needed) and running past `end`."""
    b = np.asarray(beats, dtype=float)
    period = float(np.median(np.diff(b))) if len(b) > 1 else 0.5
    while b[0] - period >= 0:
        b = np.concatenate([[b[0] - period], b])
    if b[0] > 1e-3:
        b = np.concatenate([[0.0], b])
    while b[-1] < end + period:
        b = np.append(b, b[-1] + period)
    return b


def assemble(beats: np.ndarray, downbeat_time: float, parts: dict[str, NoteList], drums: NoteList,
             key: tuple[int, str] | None = None, resolution: int = 480) -> tuple[pretty_midi.PrettyMIDI, np.ndarray]:
    """A MIDI whose tempo map puts one quarter note on every detected beat. Returns (midi, beat grid)."""
    end = max([n[1] for ns in list(parts.values()) + [drums] for n in ns], default=beats[-1])
    grid = _grid_covering(beats, end)
    pm = pretty_midi.PrettyMIDI(resolution=resolution)
    pm._tick_scales = [(i * resolution, (grid[i + 1] - grid[i]) / resolution) for i in range(len(grid) - 1)]
    pm._update_tick_to_time(len(grid) * resolution)
    first_db = int(np.argmin(np.abs(grid - downbeat_time)))
    pickup = first_db % 4
    if pickup:
        pm.time_signature_changes.append(pretty_midi.TimeSignature(pickup, 4, 0.0))
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, float(grid[pickup])))
    if key is not None:
        pm.key_signature_changes.append(pretty_midi.KeySignature(key[0] + (12 if key[1] == "minor" else 0), 0.0))
    for stem, notes in parts.items():
        inst = pretty_midi.Instrument(PITCHED[stem], name=stem)
        inst.notes = [pretty_midi.Note(v, p, s, e) for s, e, p, v in notes]
        pm.instruments.append(inst)
    kit = pretty_midi.Instrument(0, is_drum=True, name="drums")
    kit.notes = [pretty_midi.Note(v, p, s, e) for s, e, p, v in drums]
    pm.instruments.append(kit)
    return pm, grid


# ---------------------------------------------------------------- pipeline

def transcribe(audio: Path, track_id: str | None = None, force: bool = False, keep_stems: bool = False,
               separate: Callable[[Path, Path], dict[str, Path]] = demucs_separate,
               notes_for_stem: Callable[[str, Path], NoteList] = basic_pitch_notes) -> Path:
    """Audio file -> data/midi/<track_id>.mid (and .json). Cached: a song is never transcribed twice.
    Stems are deleted afterwards unless keep_stems (Demucs writes ~20 MB per 30 s clip)."""
    track_id = track_id or Path(audio).stem
    out_mid = midi_dir() / f"{track_id}.mid"
    if out_mid.exists() and not force:
        return out_mid
    t0 = time.time()
    sdir = stems_dir() / track_id
    sdir.mkdir(parents=True, exist_ok=True)
    mix = sdir / "mix.wav"
    if not mix.exists() or force:
        decode(Path(audio), mix)
    if force or not all((sdir / f"{s}.wav").exists() for s in STEMS):
        separate(mix, sdir)

    parts = {s: notes_for_stem(s, sdir / f"{s}.wav") for s in PITCHED}
    drum_y, drum_sr = read_wav(sdir / "drums.wav")
    drums = drum_onsets(drum_y, drum_sr)
    y, sr = read_wav(mix)
    beats, tempo = beats_key.beats_from_audio(y, sr)
    kicks = np.array([s for s, _, p, _ in drums if p == 36])
    phase = beats_key.downbeat_phase(beats, kicks)

    pitched = [n for ns in parts.values() for n in ns]
    hist = np.bincount([p % 12 for _, _, p, _ in pitched], weights=[e - s for s, e, _, _ in pitched],
                       minlength=12) if pitched else np.zeros(12)
    key = beats_key.krumhansl(hist)
    pm, grid = assemble(beats, float(beats[phase]), parts, drums, key)
    out_mid.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_mid))
    meta = {"track_id": track_id, "source": str(audio), "tempo": round(tempo, 2),
            "key": f"{key[0]} {key[1]}", "beats": np.round(grid, 4).tolist(),
            "downbeat": float(beats[phase]), "n_notes": {s: len(ns) for s, ns in parts.items()} | {"drums": len(drums)},
            "seconds": round(time.time() - t0, 1)}
    out_mid.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    if not keep_stems:
        shutil.rmtree(sdir, ignore_errors=True)
    return out_mid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="*", type=Path)
    ap.add_argument("--previews", action="store_true", help="every iTunes preview in data/raw/audio/")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true", help="re-transcribe even if cached")
    ap.add_argument("--keep-stems", action="store_true", help="keep data/stems/<id>/ (~20 MB per clip)")
    a = ap.parse_args()
    files = list(a.audio) + (sorted(paths.AUDIO.glob("*.m4a")) if a.previews else [])
    if not files:
        ap.error("give audio files or --previews (fetch previews with `python -m src.fetch.itunes`)")
    for i, f in enumerate(files[: a.limit]):
        try:
            out = transcribe(f, force=a.force, keep_stems=a.keep_stems)
            meta = json.loads(out.with_suffix(".json").read_text())
            print(f"[{i + 1}/{len(files[: a.limit])}] {f.name} -> {out.name}  {meta['tempo']} bpm, "
                  f"notes {meta['n_notes']}  ({meta['seconds']}s)", flush=True)
        except (subprocess.CalledProcessError, RuntimeError, ValueError) as e:
            print(f"[{i + 1}] {f.name}: failed ({e})", flush=True)


if __name__ == "__main__":
    main()
