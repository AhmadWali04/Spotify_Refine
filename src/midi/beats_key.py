"""Beat grid and key (PRD3 FR-M5).

From MIDI: beats and downbeats come from the file's tempo and time-signature maps (pretty_midi);
the key comes from its key signature when present, else Krumhansl-Schmuckler profile correlation
over the duration-weighted pitch-class histogram.

From audio (transcription): a spectral-flux onset envelope, tempo by autocorrelation with a prior
around 120 bpm, and beats by dynamic programming (Ellis 2007), all in NumPy. madmom does not install
on Python 3.12, and this is enough for a 16th-note grid; transcribe.py writes the grid into the MIDI.
"""
from __future__ import annotations

import numpy as np

HOP = 512


def stft_mag(y: np.ndarray, sr: int, n_fft: int = 2048, hop: int = HOP) -> tuple[np.ndarray, np.ndarray]:
    """(|STFT| frames x bins, bin frequencies in Hz). Frame i is centered at i * hop samples."""
    y = np.pad(np.asarray(y, dtype=float), n_fft // 2)
    n = 1 + max(0, (len(y) - n_fft) // hop)
    frames = np.lib.stride_tricks.as_strided(y, (n, n_fft), (y.strides[0] * hop, y.strides[0]))
    return np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)), np.fft.rfftfreq(n_fft, 1 / sr)


def onset_envelope(S: np.ndarray, freqs: np.ndarray, lo: float = 0.0, hi: float = np.inf) -> np.ndarray:
    """Half-wave rectified flux of log-compressed magnitude, summed over the bins in [lo, hi) Hz."""
    band = (freqs >= lo) & (freqs < hi)
    L = np.log1p(100 * S[:, band])
    flux = np.maximum(np.diff(L, axis=0, prepend=L[:1]), 0).sum(axis=1)
    return flux / max(flux.max(), 1e-12)


def estimate_tempo(env: np.ndarray, fps: float, lo: float = 60, hi: float = 200) -> float:
    """Autocorrelation peak, weighted by a log-normal prior centered on 120 bpm (one octave wide)."""
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    lags = np.arange(1, len(ac))
    bpm = 60 * fps / lags
    ok = (bpm >= lo) & (bpm <= hi)
    if not ok.any() or ac[0] <= 0:
        return 120.0
    w = np.exp(-0.5 * np.log2(bpm / 120) ** 2)
    i = np.argmax(np.where(ok, ac[1:] * w, -np.inf))
    return float(bpm[i])


def track_beats(env: np.ndarray, fps: float, bpm: float, tightness: float = 100.0) -> np.ndarray:
    """Beat frames maximizing onset strength plus a penalty for straying from the beat period."""
    period = 60 * fps / bpm
    n = len(env)
    score = np.zeros(n)
    back = -np.ones(n, dtype=int)
    lo, hi = int(round(period / 2)), int(round(2 * period))
    for t in range(n):
        prev = np.arange(max(0, t - hi), max(0, t - lo + 1))
        if len(prev):
            pen = -tightness * np.log((t - prev) / period) ** 2
            j = np.argmax(score[prev] + pen)
            if score[prev[j]] + pen[j] > 0:
                score[t], back[t] = score[prev[j]] + pen[j], prev[j]
        score[t] += env[t]
    # End on the best-scoring frame within the last period, then follow the back-links.
    tail = np.arange(max(0, n - int(np.ceil(period))), n)
    t = tail[np.argmax(score[tail])] if len(tail) else 0
    beats = [t]
    while back[beats[-1]] >= 0:
        beats.append(back[beats[-1]])
    return np.array(beats[::-1])


def beats_from_audio(y: np.ndarray, sr: int, hop: int = HOP) -> tuple[np.ndarray, float]:
    """(beat times in s, tempo in bpm). Falls back to a 120 bpm grid for silence."""
    S, freqs = stft_mag(y, sr, hop=hop)
    env = onset_envelope(S, freqs)
    fps = sr / hop
    if env.max() <= 0:
        return np.arange(0, len(y) / sr, 0.5), 120.0
    bpm = estimate_tempo(env, fps)
    beats = track_beats(env, fps, bpm) / fps
    if len(beats) < 2:
        return np.arange(0, len(y) / sr, 60 / bpm), bpm
    return beats, float(60 / np.median(np.diff(beats)))


def downbeat_phase(beats: np.ndarray, kick_times: np.ndarray, beats_per_bar: int = 4) -> int:
    """Which of the first beats_per_bar beats starts a bar: the phase with the most kicks on it
    (kicks land on 1 far more than on 2 or 4 in most popular music). 0 without kicks."""
    if len(kick_times) == 0 or len(beats) < beats_per_bar:
        return 0
    near = np.abs(kick_times[:, None] - beats[None, :]).argmin(axis=1)
    tol = 0.25 * np.median(np.diff(beats))
    hit = np.abs(kick_times - beats[near]) < tol
    counts = np.bincount(near[hit] % beats_per_bar, minlength=beats_per_bar)
    return int(np.argmax(counts))

# Krumhansl-Kessler key profiles, tonic first.
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def grid(pm) -> tuple[np.ndarray, np.ndarray, float]:
    """(beat times in s, downbeat beat indices, typical tempo in bpm). Pickup bars count as bar 0.
    Tempo is the median beat length, so a short pickup beat or a brief tempo change can't skew it."""
    beats = pm.get_beats()
    if len(beats) < 2:
        raise ValueError("MIDI file has fewer than 2 beats")
    db = np.unique(np.round(np.interp(pm.get_downbeats(), beats, np.arange(len(beats))))).astype(int)
    if len(db) == 0 or db[0] > 0:
        db = np.concatenate([[0], db])
    return beats, db, float(60.0 / np.median(np.diff(beats)))


def krumhansl(pc_hist: np.ndarray) -> tuple[int, str]:
    """(tonic 0-11, 'major' | 'minor') with the highest profile correlation."""
    if pc_hist.sum() == 0:
        return 0, "major"
    best = max(((np.corrcoef(pc_hist, np.roll(prof, t))[0, 1], t, mode)
                for mode, prof in (("major", MAJOR), ("minor", MINOR)) for t in range(12)),
               key=lambda r: np.nan_to_num(r[0], nan=-1))
    return best[1], best[2]


def key(pm, pc_hist: np.ndarray, use_signature: bool = True) -> tuple[int, str]:
    if use_signature and pm.key_signature_changes:
        k = pm.key_signature_changes[0].key_number          # 0-11 major, 12-23 minor
        return k % 12, "minor" if k >= 12 else "major"
    return krumhansl(pc_hist)
