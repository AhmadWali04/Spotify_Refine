"""Template chord estimation (PRD3 §5.2) and chord naming.

Chord index: root + 12 * quality, quality 0 = major, 1 = minor; NO_CHORD = 24.
Per beat, the beat-averaged chroma is scored against the 24 triad templates by cosine similarity;
below tau the beat is "no chord". Indices are absolute or key-relative depending on the caller:
represent.py keeps absolute names for reading and key-relative indices for modeling.
"""
from __future__ import annotations

import numpy as np

from src.linalg import l2_normalize

NO_CHORD = 24
N_CHORDS = 25
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
DEGREES = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]


def templates() -> np.ndarray:
    """24 x 12, rows L2-normalized: 12 major triads then 12 minor triads."""
    T = np.zeros((24, 12))
    for r in range(12):
        T[r, [r, (r + 4) % 12, (r + 7) % 12]] = 1
        T[12 + r, [r, (r + 3) % 12, (r + 7) % 12]] = 1
    return l2_normalize(T)


def estimate(beat_chroma: np.ndarray, tau: float = 0.6) -> np.ndarray:
    """beat_chroma: B x 12 -> chord index per beat (0-23, or NO_CHORD when silent or ambiguous)."""
    if len(beat_chroma) == 0:
        return np.zeros(0, dtype=int)
    S = l2_normalize(beat_chroma) @ templates().T
    idx = S.argmax(axis=1)
    return np.where(S.max(axis=1) >= tau, idx, NO_CHORD).astype(int)


def transpose(idx: np.ndarray | int, k: int):
    """Shift chord roots by k semitones; NO_CHORD stays put."""
    idx = np.asarray(idx)
    out = np.where(idx == NO_CHORD, NO_CHORD, (idx % 12 + k) % 12 + 12 * (idx // 12))
    return out.astype(int) if out.ndim else int(out)


def name(idx: int) -> str:
    """Absolute name: 'C', 'Am', or 'N'."""
    return "N" if idx == NO_CHORD else PITCH_NAMES[idx % 12] + ("m" if idx >= 12 else "")


def roman(idx: int) -> str:
    """Key-relative name: 'I', 'vi', 'bVII', or 'N'."""
    if idx == NO_CHORD:
        return "N"
    d = DEGREES[idx % 12]
    return d.lower() if idx >= 12 else d
