"""MIDI -> the PRD3 §5 representations: three dataframes to read, a dict of matrices to compute with.

    rep = from_path("song.mid")
    rep.notes, rep.beats, rep.summary      # notes_df, beats_df, summary_df
    rep.matrices["chord_trans"]            # 25 x 25 counts
    rep.sequences["chords"]                # integer sequences for the Markov model
    feature_blocks(rep)                    # flattened blocks for the vector model

Time is on the beat grid (16th-note steps), so tempo never changes a matrix. Drum notes only feed
drum_grid and the drum sequence; everything pitched feeds the piano roll. Matrices used for
modeling (chroma_keynorm, chord_seq, chord_trans, pc_trans) are key-relative: the tonic is 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

from src.linalg import l2_normalize
from src.midi import beats_key, chords

VERSION = 3            # bump when a representation changes; invalidates data/midi/reps/
STEPS_PER_BEAT = 4
STEPS_PER_BAR = 16
DRUM_ROWS = ("kick", "snare", "hat")
DRUM_MAP = {35: 0, 36: 0, 37: 1, 38: 1, 39: 1, 40: 1, 22: 2, 26: 2, 42: 2, 44: 2, 46: 2}
BASS_PROGRAMS = range(32, 40)
MELODY_NAMES = ("other", "vocals")   # stem names written by transcription (M17)
ALPHABETS = {"chords": chords.N_CHORDS, "pitch_classes": 12, "drums": 8}
BLOCKS = ("chord_trans", "pc_trans", "interval_hist", "drum_grid", "drum_grid_var",
          "chroma_dft_mag", "scalars")
BIG_MATRICES = ("piano_roll", "chroma", "chroma_keynorm", "ssm")
SCALARS = ("tempo", "minor", "note_density", "pitch_range", "syncopation", "n_unique_chords",
           "drum_density")


@dataclass
class SongRep:
    track_id: str
    tempo: float
    tonic: int
    mode: str
    notes: pd.DataFrame
    beats: pd.DataFrame
    summary: pd.DataFrame
    matrices: dict[str, np.ndarray]
    sequences: dict[str, np.ndarray]
    scalars: dict[str, float]

    @property
    def key_name(self) -> str:
        return f"{chords.PITCH_NAMES[self.tonic]} {self.mode}"

    def compact(self) -> "SongRep":
        """Only what the models read (no dataframes, piano roll, chroma or SSM): ~1% of the size, for caching."""
        return SongRep(self.track_id, self.tempo, self.tonic, self.mode, pd.DataFrame(), pd.DataFrame(),
                       self.summary[["track_id", "tempo", "key", "n_beats", "n_bars"]],
                       {k: v for k, v in self.matrices.items() if k not in BIG_MATRICES},
                       self.sequences, self.scalars)


def fold_matrix() -> np.ndarray:
    """F in {0,1}^{12 x 128}: F[c, p] = 1 when pitch p has pitch class c."""
    F = np.zeros((12, 128))
    F[np.arange(128) % 12, np.arange(128)] = 1
    return F


def pitch_name(p: int) -> str:
    return f"{chords.PITCH_NAMES[p % 12]}{p // 12 - 1}"


def quantize(beat_pos: np.ndarray) -> np.ndarray:
    """Fractional beat position -> nearest 16th-note step."""
    return np.floor(np.asarray(beat_pos) * STEPS_PER_BEAT + 0.5).astype(int)


def _beat_pos(times: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Seconds -> fractional beat index, extended past the last beat at the final beat length."""
    pos = np.interp(times, beats, np.arange(len(beats)))
    after = times > beats[-1]
    pos[after] = len(beats) - 1 + (times[after] - beats[-1]) / (beats[-1] - beats[-2])
    return np.maximum(pos, 0.0)


def _notes_df(pm, track_id: str, beats, downbeats, tonic: int) -> pd.DataFrame:
    rows = []
    for inst in pm.instruments:
        label = inst.name.strip().lower() or ("drums" if inst.is_drum else
                                              pretty_midi.program_to_instrument_name(inst.program).lower())
        rows += [(label, inst.program, inst.is_drum, n.pitch, n.start, n.end, n.velocity) for n in inst.notes]
    if not rows:
        raise ValueError(f"{track_id}: no notes")
    df = pd.DataFrame(rows, columns=["instrument", "program", "is_drum", "pitch", "start_s", "end_s", "velocity"])
    df = df.sort_values(["start_s", "pitch"], kind="stable").reset_index(drop=True)
    sb, eb = _beat_pos(df["start_s"].to_numpy(), beats), _beat_pos(df["end_s"].to_numpy(), beats)
    # Onsets snap to the nearest 16th (performed and transcribed notes land a little early or late).
    step = quantize(sb)
    bar = np.searchsorted(downbeats, step // STEPS_PER_BEAT, side="right") - 1
    df.insert(0, "track_id", track_id)
    drum = df["is_drum"].to_numpy()
    df.insert(5, "pitch_name", [pretty_midi.note_number_to_drum_name(p) or str(p) if d else pitch_name(p)
                                for p, d in zip(df["pitch"], drum)])
    df.insert(6, "pitch_class", np.where(drum, -1, (df["pitch"] - tonic) % 12))   # -1: drums
    df["start_beat"] = sb
    df["duration_beats"] = eb - sb
    df["bar"] = bar
    df["step_in_bar"] = step - downbeats[bar] * STEPS_PER_BEAT
    df["velocity"] = df.pop("velocity")
    return df


def _melody_mask(notes: pd.DataFrame) -> np.ndarray:
    """Melody voices: the transcribed 'other' / 'vocals' stems if present, else every pitched
    non-bass instrument (clean MIDI files)."""
    pitched = ~notes["is_drum"].to_numpy()
    named = notes["instrument"].isin(MELODY_NAMES).to_numpy()
    if (named & pitched).any():
        return named & pitched
    return pitched & ~notes["program"].isin(BASS_PROGRAMS).to_numpy()


def from_midi(pm: pretty_midi.PrettyMIDI, track_id: str, tau_chord: float = 0.6,
              use_key_signature: bool = True) -> SongRep:
    beats, downbeats, tempo = beats_key.grid(pm)
    B = len(beats)
    T = B * STEPS_PER_BEAT

    # Key from the pitched notes first, since pitch_class in notes_df is key-relative.
    pre = [(n.pitch, n.end - n.start) for i in pm.instruments if not i.is_drum for n in i.notes]
    hist = np.bincount([p % 12 for p, _ in pre], weights=[d for _, d in pre], minlength=12) if pre else np.zeros(12)
    tonic, mode = beats_key.key(pm, hist, use_key_signature)
    notes = _notes_df(pm, track_id, beats, downbeats, tonic)

    step = quantize(notes["start_beat"].to_numpy())
    end_step = quantize((notes["start_beat"] + notes["duration_beats"]).to_numpy())
    drum = notes["is_drum"].to_numpy()

    # Piano roll (binary) and chroma.
    roll = np.zeros((128, T))
    for p, s, e in zip(notes["pitch"][~drum], step[~drum], end_step[~drum]):
        roll[p, min(s, T - 1): min(max(e, s + 1), T)] = 1
    chroma = fold_matrix() @ roll
    chroma_keynorm = np.roll(chroma, -tonic, axis=0)

    # Chords per beat: absolute names for reading, key-relative indices for modeling.
    beat_chroma = chroma.reshape(12, B, STEPS_PER_BEAT).sum(axis=2).T
    chord_abs = chords.estimate(beat_chroma, tau_chord)
    chord_seq = chords.transpose(chord_abs, -tonic)
    chord_trans = transition_counts(chord_seq, chords.N_CHORDS)

    # Melody: highest onset per 16th step in the melody voices.
    mel = notes[_melody_mask(notes)].assign(step=step[_melody_mask(notes)])
    mel_pitch = mel.groupby("step")["pitch"].max().to_numpy()
    pcs = (mel_pitch - tonic) % 12
    pc_trans = transition_counts(pcs, 12)
    interval_hist = np.bincount(np.clip(np.diff(mel_pitch), -12, 12) + 12, minlength=25).astype(float)

    # Drums: kick / snare / hat activation per step, then the average bar.
    dmask = drum & notes["pitch"].isin(list(DRUM_MAP)).to_numpy()
    drow = notes["pitch"][dmask].map(DRUM_MAP).to_numpy()
    D = np.zeros((3, T))
    D[drow, np.minimum(step[dmask], T - 1)] = 1
    n_bars = len(downbeats)
    bars = np.zeros((n_bars, 3, STEPS_PER_BAR))
    sib, dbar = notes["step_in_bar"].to_numpy()[dmask], notes["bar"].to_numpy()[dmask]
    ok = (sib >= 0) & (sib < STEPS_PER_BAR)
    bars[dbar[ok], drow[ok], sib[ok]] = 1
    played = bars.sum(axis=(1, 2)) > 0
    drum_grid = bars[played].mean(axis=0) if played.any() else np.zeros((3, STEPS_PER_BAR))
    drum_grid_var = bars[played].var(axis=0) if played.any() else np.zeros((3, STEPS_PER_BAR))
    if dmask.any():
        lo, hi = np.flatnonzero(D.any(axis=0))[[0, -1]]
        drum_states = (D[0] + 2 * D[1] + 4 * D[2])[lo: hi + 1].astype(int)
    else:
        drum_states = np.zeros(0, dtype=int)

    # Structure and key-free harmony.
    edges = np.append(downbeats, B) * STEPS_PER_BEAT
    bar_chroma = np.array([chroma[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])])
    bc = l2_normalize(bar_chroma)
    ssm = bc @ bc.T
    mean_chroma = chroma.sum(axis=1)
    chroma_dft_mag = np.abs(np.fft.fft(mean_chroma / max(mean_chroma.sum(), 1e-12)))[:7]

    matrices = {
        "piano_roll": roll, "chroma": chroma, "chroma_keynorm": chroma_keynorm,
        "chord_seq": chord_seq, "chord_trans": chord_trans, "pc_trans": pc_trans,
        "interval_hist": interval_hist, "drum_grid": drum_grid, "drum_grid_var": drum_grid_var,
        "ssm": ssm, "chroma_dft_mag": chroma_dft_mag,
    }
    sequences = {"chords": chord_seq, "pitch_classes": pcs.astype(int), "drums": drum_states}

    pitched = notes[~drum]
    scalars = {
        "tempo": tempo,
        "minor": float(mode == "minor"),
        "note_density": len(pitched) / B,
        "pitch_range": float(pitched["pitch"].max() - pitched["pitch"].min()) if len(pitched) else 0.0,
        "syncopation": float(np.mean(step % STEPS_PER_BEAT != 0)),
        "n_unique_chords": float(len(set(chord_seq.tolist()) - {chords.NO_CHORD})),
        "drum_density": float(dmask.sum()) / B,
    }

    beats_df = pd.DataFrame({
        "beat": np.arange(B),
        "bar": np.searchsorted(downbeats, np.arange(B), side="right") - 1,
        "chord": [chords.name(c) for c in chord_abs],
        "chord_root": ["" if c == chords.NO_CHORD else chords.PITCH_NAMES[c % 12] for c in chord_abs],
        "chord_quality": ["none" if c == chords.NO_CHORD else "min" if c >= 12 else "maj" for c in chord_abs],
        "n_active_notes": [int(roll[:, b * STEPS_PER_BEAT:(b + 1) * STEPS_PER_BEAT].any(axis=1).sum()) for b in range(B)],
        **{f"c{c}": beat_chroma[:, c] / STEPS_PER_BEAT for c in range(12)},
        **{name: D[r].reshape(B, STEPS_PER_BEAT).sum(axis=1).astype(int) for r, name in enumerate(DRUM_ROWS)},
    })

    rep = SongRep(track_id, tempo, tonic, mode, notes, beats_df, pd.DataFrame(), matrices, sequences, scalars)
    blocks = feature_blocks(rep)
    rep.summary = pd.DataFrame([{
        "track_id": track_id, "tempo": round(tempo, 2), "key": rep.key_name, "mode": mode,
        "duration_s": round(pm.get_end_time(), 2), "n_notes": int(len(pitched)), "n_beats": B, "n_bars": n_bars,
        "note_density": scalars["note_density"], "pitch_min": int(pitched["pitch"].min()) if len(pitched) else None,
        "pitch_max": int(pitched["pitch"].max()) if len(pitched) else None,
        "pitch_range": scalars["pitch_range"], "syncopation": scalars["syncopation"],
        "n_unique_chords": int(scalars["n_unique_chords"]),
        **{f"{b}_{i}": v for b in BLOCKS for i, v in enumerate(blocks[b])},
    }])
    return rep


def from_path(path: str | Path, track_id: str | None = None, **kw) -> SongRep:
    path = Path(path)
    return from_midi(pretty_midi.PrettyMIDI(str(path)), track_id or path.stem, **kw)


# ---------------------------------------------------------------- shared math

def transition_counts(seq: np.ndarray, K: int, order: int = 1) -> np.ndarray:
    """(K^order) x K counts of context -> next symbol. Order-2 contexts are x[t-1] * K + x[t]."""
    C = np.zeros((K ** order, K))
    seq = np.asarray(seq, dtype=int)
    if len(seq) <= order:
        return C
    ctx = np.zeros(len(seq) - order, dtype=int)
    for j in range(order):
        ctx = ctx * K + seq[j: len(seq) - order + j]
    np.add.at(C, (ctx, seq[order:]), 1)
    return C


def row_normalize(M: np.ndarray) -> np.ndarray:
    s = M.sum(axis=-1, keepdims=True)
    return np.divide(M, s, out=np.zeros_like(M, dtype=float), where=s > 0)


def transpose_matrices(m: dict[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    """The key-relative matrices as if the song were shifted by k semitones (augmentation)."""
    perm = chords.transpose(np.arange(chords.N_CHORDS), k)      # chord i -> perm[i]
    out = dict(m)
    ct = np.zeros_like(m["chord_trans"])
    ct[np.ix_(perm, perm)] = m["chord_trans"]
    out["chord_trans"] = ct
    out["pc_trans"] = np.roll(m["pc_trans"], (k, k), axis=(0, 1))
    out["chord_seq"] = perm[m["chord_seq"]]
    if "chroma_keynorm" in m:
        out["chroma_keynorm"] = np.roll(m["chroma_keynorm"], k, axis=0)
    return out


def feature_blocks(rep: SongRep, shift: int = 0) -> dict[str, np.ndarray]:
    """Flattened vector-mode blocks; transition matrices are row-normalized first (PRD3 §6.2)."""
    m = transpose_matrices(rep.matrices, shift) if shift % 12 else rep.matrices
    ih = m["interval_hist"]
    return {
        "chord_trans": row_normalize(m["chord_trans"]).ravel(),
        "pc_trans": row_normalize(m["pc_trans"]).ravel(),
        "interval_hist": ih / max(ih.sum(), 1),
        "drum_grid": m["drum_grid"].ravel(),
        "drum_grid_var": m["drum_grid_var"].ravel(),
        "chroma_dft_mag": m["chroma_dft_mag"],
        "scalars": np.array([rep.scalars[s] for s in SCALARS]),
    }
