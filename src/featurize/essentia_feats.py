"""E2: Essentia pretrained classifiers on the iTunes previews.

Vector per song (~408 named axes):
    mood_happy, mood_sad, mood_aggressive, mood_relaxed, mood_party, danceability,
    valence, arousal, genre:<400 Discogs styles>

Needs `pip install essentia-tensorflow` (and ffmpeg for .m4a decoding). Model graphs are
downloaded once to data/models/. Per-track results are cached in data/cache/essentia/<id>.npy,
so a song is only ever processed once.
"""
from __future__ import annotations

import json

import numpy as np
import requests

from src.paths import AUDIO, CACHE, MODELS

BASE = "https://essentia.upf.edu/models"
EFFNET = ("feature-extractors/discogs-effnet", "discogs-effnet-bs64-1")
MUSICNN = ("feature-extractors/musicnn", "msd-musicnn-1")
# (feature name, model folder, model file stem, positive class in the model's metadata)
MOOD_HEADS = [
    ("mood_happy", "classification-heads/mood_happy", "mood_happy-discogs-effnet-1", "happy"),
    ("mood_sad", "classification-heads/mood_sad", "mood_sad-discogs-effnet-1", "sad"),
    ("mood_aggressive", "classification-heads/mood_aggressive", "mood_aggressive-discogs-effnet-1", "aggressive"),
    ("mood_relaxed", "classification-heads/mood_relaxed", "mood_relaxed-discogs-effnet-1", "relaxed"),
    ("mood_party", "classification-heads/mood_party", "mood_party-discogs-effnet-1", "party"),
    ("danceability", "classification-heads/danceability", "danceability-discogs-effnet-1", "danceable"),
]
GENRE = ("classification-heads/genre_discogs400", "genre_discogs400-discogs-effnet-1")
EMO = ("classification-heads/emomusic", "emomusic-msd-musicnn-2")   # valence, arousal on a 1-9 scale

CACHE_DIR = CACHE / "essentia"
NAMES_FILE = CACHE_DIR / "_names.json"
NAMED_AXES = [h[0] for h in MOOD_HEADS] + ["valence", "arousal"]


def _fetch(folder: str, stem: str, ext: str) -> str:
    out = MODELS / f"{stem}.{ext}"
    if not out.exists():
        print(f"Downloading {out.name}")
        r = requests.get(f"{BASE}/{folder}/{stem}.{ext}", timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)
    return str(out)


def _classes(folder: str, stem: str) -> list[str]:
    return json.loads(open(_fetch(folder, stem, "json")).read())["classes"]


class _Models:
    """Loads every graph once; essentia is imported lazily so the rest of the repo works without it."""

    def __init__(self):
        try:
            from essentia.standard import (MonoLoader, TensorflowPredict2D,
                                           TensorflowPredictEffnetDiscogs, TensorflowPredictMusiCNN)
        except ImportError as e:
            raise ImportError("E2 needs essentia-tensorflow: `pip install essentia-tensorflow` "
                              "(run `python check_audio_install.py` to test)") from e
        self.MonoLoader = MonoLoader
        self.effnet = TensorflowPredictEffnetDiscogs(graphFilename=_fetch(*EFFNET, "pb"),
                                                     output="PartitionedCall:1")
        self.musicnn = TensorflowPredictMusiCNN(graphFilename=_fetch(*MUSICNN, "pb"),
                                                output="model/dense/BiasAdd")
        self.heads = []
        for name, folder, stem, positive in MOOD_HEADS:
            col = _classes(folder, stem).index(positive)
            self.heads.append((name, col, TensorflowPredict2D(graphFilename=_fetch(folder, stem, "pb"),
                                                              output="model/Softmax")))
        self.genre_names = _classes(*GENRE)
        self.genre = TensorflowPredict2D(graphFilename=_fetch(*GENRE, "pb"),
                                         input="serving_default_model_Placeholder",
                                         output="PartitionedCall:0")
        emo_classes = _classes(*EMO)
        self.emo_cols = (emo_classes.index("valence"), emo_classes.index("arousal"))
        self.emo = TensorflowPredict2D(graphFilename=_fetch(*EMO, "pb"), output="model/Identity")

    def names(self) -> list[str]:
        return [h[0] for h in self.heads] + ["valence", "arousal"] + [f"genre:{g}" for g in self.genre_names]

    def __call__(self, path: str) -> np.ndarray:
        audio = self.MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()
        emb = self.effnet(audio)
        moods = [float(np.mean(head(emb)[:, col])) for _, col, head in self.heads]
        genres = np.mean(self.genre(emb), axis=0)
        emo = np.mean(self.emo(self.musicnn(audio)), axis=0)
        # Rescale valence / arousal from the 1-9 annotation scale to roughly [-1, 1].
        va = [(float(emo[self.emo_cols[0]]) - 5) / 4, (float(emo[self.emo_cols[1]]) - 5) / 4]
        return np.asarray(moods + va + list(genres), dtype=np.float32)


def feature_names() -> list[str]:
    if not NAMES_FILE.exists():
        raise FileNotFoundError("Run the essentia featurizer once to create feature names.")
    return json.loads(NAMES_FILE.read_text())


def cached_features(track_ids: list[str]) -> tuple[list[str], np.ndarray]:
    """Read already-computed vectors without loading any models (used by the sorter's 'vibes')."""
    if not NAMES_FILE.exists():
        return [], np.zeros((0, 0), dtype=np.float32)
    ids = [t for t in track_ids if (CACHE_DIR / f"{t}.npy").exists()]
    X = np.stack([np.load(CACHE_DIR / f"{t}.npy") for t in ids]) if ids else np.zeros((0, len(feature_names())))
    return ids, X


def featurize(lib: dict, track_ids: list[str], params: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    have_audio = [t for t in track_ids if (AUDIO / f"{t}.m4a").exists()]
    todo = [t for t in have_audio if not (CACHE_DIR / f"{t}.npy").exists()]
    print(f"Essentia: {len(have_audio)} songs with audio, {len(todo)} to process")
    if todo or not NAMES_FILE.exists():
        models = _Models()
        NAMES_FILE.write_text(json.dumps(models.names()))
        for i, t in enumerate(todo, 1):
            try:
                np.save(CACHE_DIR / f"{t}.npy", models(str(AUDIO / f"{t}.m4a")))
            except Exception as e:  # noqa: BLE001 - one bad clip must not stop the run
                print(f"  ! {t}: {type(e).__name__}: {e}")
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")

    ids, X = cached_features(have_audio)
    blocks = params.get("blocks", "all")          # "all" | "moods" (8 named axes only)
    if blocks == "moods":
        X = X[:, : len(NAMED_AXES)]
    return ids, X
