"""PRD3 M14: the Lakh subset builder on a tiny fake LMD archive (no download)."""
import io
import json
import tarfile

import numpy as np
import pandas as pd
import pretty_midi
import pytest

from src import paths
from src.data import lakh
from src.midi import synthetic

GENRES = ["pop", "edm", "ballad"]


@pytest.fixture
def fake_lmd(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    dest = tmp_path / "lmd"
    dest.mkdir()
    scores, cd2 = {}, ["# comment", "# Fields: trackId, majority-genre, minority-genre?"]
    with tarfile.open(dest / "lmd_matched.tar.gz", "w:gz") as tf:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        for g, genre in enumerate(GENRES):
            for i in range(12):
                msd = f"TR{chr(65 + g)}{chr(65 + i)}X{i:02d}{g}128F0000000"[:18]
                buf = io.BytesIO()
                synthetic.song(genre, g * 100 + i, n_bars=24).write(buf)
                add(lakh.member_name(msd, "good"), buf.getvalue())
                add(lakh.member_name(msd, "worse"), b"not a midi file")
                scores[msd] = {"good": 0.9, "worse": 0.5}
                cd2.append(f"{msd}\t{genre}" + ("\tRock" if i == 0 else ""))
        # A labeled song too short to keep, and one that is unreadable.
        short = pretty_midi.PrettyMIDI(initial_tempo=120)
        inst = pretty_midi.Instrument(0)
        inst.notes.append(pretty_midi.Note(80, 60, 0, 1))
        short.instruments.append(inst)
        buf = io.BytesIO()
        short.write(buf)
        add(lakh.member_name("TRSHORT00000000000", "s"), buf.getvalue())
        add(lakh.member_name("TRBROKEN0000000000", "b"), b"garbage")
        scores.update({"TRSHORT00000000000": {"s": 1.0}, "TRBROKEN0000000000": {"b": 1.0}})
        cd2 += ["TRSHORT00000000000\tpop", "TRBROKEN0000000000\tpop"]
    (dest / "match_scores.json").write_text(json.dumps(scores))
    (dest / "msd_tagtraum_cd2.cls").write_text("\n".join(cd2) + "\n")
    return dest


def test_labels_and_best_match(fake_lmd):
    labels = lakh.load_labels(fake_lmd)
    assert len(labels) == 38 and all(len(v) == 1 for v in labels.values())
    multi = lakh.load_labels(fake_lmd, multi=True)
    assert sum(len(v) == 2 for v in multi.values()) == 3
    assert {md5 for md5, _ in lakh.best_matches(fake_lmd).values()} == {"good", "s", "b"}


def test_extract_build_and_freeze(fake_lmd):
    assert lakh.extract(fake_lmd) == 38
    df = lakh.build(fake_lmd, cap=10, min_genre=5, workers=1)
    assert sorted(df["genre"].unique()) == sorted(GENRES)
    assert df["genre"].value_counts().tolist() == [10, 10, 10]
    assert (df["duration_s"] >= lakh.MIN_SECONDS).all()
    assert df.groupby("genre")["fold"].apply(lambda f: sorted(np.bincount(f))).tolist() == [[2] * 5] * 3
    rejects = pd.read_csv(fake_lmd / "rejects.csv")
    assert set(rejects["track_id"]) == {"TRSHORT00000000000", "TRBROKEN0000000000"}
    with pytest.raises(SystemExit):
        lakh.build(fake_lmd, workers=1)                    # frozen
    with pytest.raises(SystemExit):
        lakh.build(fake_lmd, cap=10, min_genre=13, workers=1, force=True)     # every genre too small


def test_subset_feeds_the_experiment_harness(fake_lmd):
    from src.midi import experiment
    lakh.extract(fake_lmd)
    lakh.build(fake_lmd, cap=10, min_genre=5, workers=1)
    reps, labels, fold = experiment.load_data(str(fake_lmd / "subset.csv"), {})
    assert len(reps) == 30 and fold is not None and "piano_roll" not in reps[0].matrices
    res, _, _ = experiment.evaluate({"model": "markov", "markov": {"sequences": ["chords", "drums"]}},
                                    reps, labels, fold)
    assert res["top1_mean"] > 2 * res["majority_top1"]
