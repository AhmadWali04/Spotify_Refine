"""Look inside one song (PRD3 §5.3): every dataframe, matrix and plot in one folder.

    python -m src.midi.inspect path/to/song.mid
    python -m src.midi.inspect --synthetic jazz          # a generated song, no file needed

Writes data/inspect/<track_id>/: notes.csv, beats.csv, summary.csv, matrices.npz, piano_roll.png,
chroma.png, chord_trans.png, drum_grid.png, ssm.png and transcribed.mid; prints the first notes,
key, tempo and top chord transitions. Audio input arrives with transcription (M17).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402

from src import paths  # noqa: E402
from src.midi import chords, represent, synthetic  # noqa: E402

MIDI_EXT = {".mid", ".midi"}


def top_transitions(rep: represent.SongRep, k: int = 5) -> list[tuple[str, str, int]]:
    """Most frequent chord changes (self-transitions left out), as key-relative numerals."""
    C = rep.matrices["chord_trans"].copy()
    np.fill_diagonal(C, 0)
    flat = np.argsort(-C, axis=None)[:k]
    return [(chords.roman(i), chords.roman(j), int(C[i, j]))
            for i, j in zip(*np.unravel_index(flat, C.shape)) if C[i, j] > 0]


def _heat(M, title, out, xlabel, ylabel, yticks=None, xticks=None, cmap="magma"):
    fig, ax = plt.subplots(figsize=(10, 4) if M.shape[1] > 3 * M.shape[0] else (6, 5))
    ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yticks is not None:
        ax.set_yticks(range(len(yticks)), yticks, fontsize=7)
    if xticks is not None:
        ax.set_xticks(range(len(xticks)), xticks, fontsize=7, rotation=90)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plots(rep: represent.SongRep, out: Path) -> None:
    m = rep.matrices
    roll = m["piano_roll"]
    used = np.flatnonzero(roll.any(axis=1))
    lo, hi = (used[0], used[-1] + 1) if len(used) else (0, 128)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(roll[lo:hi], aspect="auto", origin="lower", cmap="Greys", interpolation="nearest",
              extent=(0, roll.shape[1] / 4, lo - 0.5, hi - 0.5))
    ax.set(title=f"{rep.track_id}: piano roll", xlabel="beat", ylabel="MIDI pitch")
    fig.tight_layout()
    fig.savefig(out / "piano_roll.png", dpi=110)
    plt.close(fig)

    _heat(m["chroma"], f"chroma ({rep.key_name})", out / "chroma.png", "16th step", "pitch class",
          yticks=chords.PITCH_NAMES)
    names = [chords.roman(i) for i in range(chords.N_CHORDS)]
    _heat(represent.row_normalize(m["chord_trans"]), "chord transitions P(next | current), key-relative",
          out / "chord_trans.png", "next chord", "current chord", yticks=names, xticks=names)
    _heat(m["drum_grid"], "average drum bar", out / "drum_grid.png", "16th step in bar", "",
          yticks=represent.DRUM_ROWS, cmap="Blues")
    _heat(m["ssm"], "bar self-similarity (chroma cosine)", out / "ssm.png", "bar", "bar", cmap="viridis")


def inspect(path: Path | None = None, pm: pretty_midi.PrettyMIDI | None = None, track_id: str | None = None,
            out_root: Path | None = None, quiet: bool = False) -> Path:
    if pm is None:
        if path.suffix.lower() not in MIDI_EXT:
            raise SystemExit(f"{path.suffix} input needs audio transcription, which arrives in M17. Pass a .mid file.")
        pm = pretty_midi.PrettyMIDI(str(path))
    track_id = track_id or path.stem
    rep = represent.from_midi(pm, track_id)
    out = (out_root or paths.DATA / "inspect") / track_id
    out.mkdir(parents=True, exist_ok=True)
    rep.notes.to_csv(out / "notes.csv", index=False)
    rep.beats.to_csv(out / "beats.csv", index=False)
    rep.summary.to_csv(out / "summary.csv", index=False)
    np.savez_compressed(out / "matrices.npz", **rep.matrices)
    plots(rep, out)
    if path is not None and path.suffix.lower() in MIDI_EXT:
        shutil.copyfile(path, out / "transcribed.mid")
    else:
        pm.write(str(out / "transcribed.mid"))
    if not quiet:
        print(rep.notes.head(10).to_string(index=False))
        print(f"\nkey {rep.key_name}   tempo {rep.tempo:.1f} bpm   {len(rep.beats)} beats   "
              f"{int(rep.summary['n_bars'][0])} bars")
        print("top chord changes: " + ", ".join(f"{a} -> {b} ({n}x)" for a, b, n in top_transitions(rep)))
        print(f"\nWrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", type=Path)
    ap.add_argument("--synthetic", choices=list(synthetic.GENRES), help="inspect a generated song instead")
    a = ap.parse_args()
    if a.synthetic:
        inspect(pm=synthetic.song(a.synthetic, 0), track_id=f"synthetic_{a.synthetic}")
    elif a.path:
        inspect(a.path)
    else:
        ap.error("give a .mid path or --synthetic GENRE")


if __name__ == "__main__":
    main()
