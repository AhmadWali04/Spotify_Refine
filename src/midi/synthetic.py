"""Synthetic MIDI "genres" for testing the symbolic pipeline before Lakh is downloaded.

Each genre has its own chord progressions, mode, tempo range, drum groove and melody habits, and
every song gets a random key, length and some noise (chord substitutions, dropped and extra drum
hits, wandering melody). Songs carry stem-style instrument names (bass / other / vocals / drums),
the same names transcription will write.

    python -m src.midi.synthetic --per-genre 40      # writes data/midi/synthetic/*.mid + labels.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pretty_midi

from src import paths

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

# Progressions as (semitones above tonic, minor?) per bar.
GENRES = {
    "pop": {"mode": "major", "tempo": (100, 126), "progs": [[(0, 0), (7, 0), (9, 1), (5, 0)],
                                                            [(9, 1), (5, 0), (0, 0), (7, 0)]],
            "kick": [0, 8, 10], "snare": [4, 12], "hat": list(range(0, 16, 2)), "chord_beats": 4},
    "blues": {"mode": "major", "tempo": (78, 100),
              "progs": [[(0, 0)] * 4 + [(5, 0)] * 2 + [(0, 0)] * 2 + [(7, 0), (5, 0), (0, 0), (7, 0)]],
              "kick": [0, 6, 8], "snare": [4, 12], "hat": [0, 3, 4, 7, 8, 11, 12, 15], "chord_beats": 4},
    "ballad": {"mode": "minor", "tempo": (62, 84), "progs": [[(0, 1), (8, 0), (3, 0), (10, 0)],
                                                             [(0, 1), (5, 1), (8, 0), (7, 0)]],
               "kick": [0], "snare": [8], "hat": [0, 4, 8, 12], "chord_beats": 4},
    "edm": {"mode": "minor", "tempo": (120, 132), "progs": [[(0, 1), (0, 1), (8, 0), (10, 0)],
                                                            [(0, 1), (10, 0), (8, 0), (10, 0)]],
            "kick": [0, 4, 8, 12], "snare": [4, 12], "hat": [2, 6, 10, 14], "chord_beats": 2},
    "jazz": {"mode": "major", "tempo": (110, 160), "progs": [[(2, 1), (7, 0), (0, 0), (0, 0)],
                                                             [(0, 0), (9, 1), (2, 1), (7, 0)]],
             "kick": [], "snare": [6, 14], "hat": [0, 4, 6, 8, 12, 14], "chord_beats": 2},
}


def song(genre: str, seed: int) -> pretty_midi.PrettyMIDI:
    g = GENRES[genre]
    rng = np.random.default_rng(seed)
    tempo = rng.uniform(*g["tempo"])
    tonic = int(rng.integers(12))
    scale = MAJOR_SCALE if g["mode"] == "major" else MINOR_SCALE
    beat = 60.0 / tempo
    step = beat / 4
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    bass = pretty_midi.Instrument(33, name="bass")
    pad = pretty_midi.Instrument(0, name="other")
    voc = pretty_midi.Instrument(73, name="vocals")
    drums = pretty_midi.Instrument(0, is_drum=True, name="drums")

    prog = g["progs"][rng.integers(len(g["progs"]))]
    n_bars = int(rng.integers(8, 17))
    cb = g["chord_beats"]
    chords_per_bar = 4 // cb
    deg = 0                                                          # melody scale degree
    for bar in range(n_bars):
        t0 = bar * 4 * beat
        for c in range(chords_per_bar):
            root, minor = prog[(bar * chords_per_bar + c) % len(prog)]
            if rng.random() < 0.1:                                   # occasional substitution
                root, minor = int(rng.choice([2, 4, 5, 7, 9])), int(rng.integers(2))
            r = (tonic + root) % 12
            start, end = t0 + c * cb * beat, t0 + (c + 1) * cb * beat
            for iv in (0, 3 if minor else 4, 7):
                pad.notes.append(pretty_midi.Note(70, 60 + (r + iv) % 12, start, end - 0.01))
            for b in range(cb):
                bass.notes.append(pretty_midi.Note(90, 36 + r, start + b * beat, start + (b + 0.9) * beat))
        # Melody: a step-wise walk on the scale, one note per 8th, with rests.
        for s in range(0, 16, 2):
            if rng.random() < 0.3:
                continue
            deg = int(np.clip(deg + rng.choice([-2, -1, 0, 1, 1, 2]), -3, 9))
            mel = 64 + tonic + 12 * (deg // 7) + scale[deg % 7]
            voc.notes.append(pretty_midi.Note(80, mel, t0 + s * step, t0 + (s + 2) * step - 0.01))
        for pitch, steps in ((36, g["kick"]), (38, g["snare"]), (42, g["hat"])):
            for s in steps:
                if rng.random() < 0.9:
                    drums.notes.append(pretty_midi.Note(100, pitch, t0 + s * step, t0 + s * step + 0.05))
            if rng.random() < 0.2:
                s = int(rng.integers(16))
                drums.notes.append(pretty_midi.Note(80, pitch, t0 + s * step, t0 + s * step + 0.05))
    pm.instruments += [bass, pad, voc, drums]
    return pm


def dataset(per_genre: int = 30, seed: int = 0) -> list[tuple[str, str, pretty_midi.PrettyMIDI]]:
    """[(track_id, genre, midi)]"""
    out = []
    for gi, genre in enumerate(GENRES):
        for i in range(per_genre):
            out.append((f"{genre}_{i:03d}", genre, song(genre, seed * 100_000 + gi * 1000 + i)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-genre", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = paths.DATA / "midi" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tid, genre, pm in dataset(a.per_genre, a.seed):
        pm.write(str(out / f"{tid}.mid"))
        rows.append({"track_id": tid, "path": str(out / f"{tid}.mid"), "label": genre})
    pd.DataFrame(rows).to_csv(out / "labels.csv", index=False)
    print(f"Wrote {len(rows)} songs ({len(GENRES)} genres) to {out}")


if __name__ == "__main__":
    main()
