"""Beat grid and key (PRD3 FR-M5).

From MIDI: beats and downbeats come from the file's tempo and time-signature maps (pretty_midi);
the key comes from its key signature when present, else Krumhansl-Schmuckler profile correlation
over the duration-weighted pitch-class histogram. Estimating beats from audio (madmom / Essentia)
is part of the transcription milestone (M17) and will write the same grid into the MIDI it makes.
"""
from __future__ import annotations

import numpy as np

# Krumhansl-Kessler key profiles, tonic first.
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def grid(pm) -> tuple[np.ndarray, np.ndarray, float]:
    """(beat times in s, downbeat beat indices, initial tempo in bpm). Pickup bars count as bar 0."""
    beats = pm.get_beats()
    if len(beats) < 2:
        raise ValueError("MIDI file has fewer than 2 beats")
    db = np.unique(np.round(np.interp(pm.get_downbeats(), beats, np.arange(len(beats))))).astype(int)
    if len(db) == 0 or db[0] > 0:
        db = np.concatenate([[0], db])
    _, tempi = pm.get_tempo_changes()
    return beats, db, float(tempi[0]) if len(tempi) else 120.0


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
