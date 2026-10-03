"""E3: LAION-CLAP audio embeddings (512 dims) on the iTunes previews.

Optional text-prompt axes: params.prompt_axes is a list of [positive, negative] prompts.
Each axis is the CLAP text direction t(pos) - t(neg); the song's projection onto it is
appended to the vector (a dot product with a direction in the shared audio-text space).

Needs `pip install laion-clap` (pulls torch). The default checkpoint downloads on first use.
Per-track embeddings are cached in data/cache/clap/<id>.npy.
"""
from __future__ import annotations

import numpy as np

from src.linalg import l2_normalize
from src.paths import AUDIO, CACHE

CACHE_DIR = CACHE / "clap"
BATCH = 16
_model = None


def _load():
    global _model
    if _model is None:
        try:
            import laion_clap
        except ImportError as e:
            raise ImportError("E3 needs LAION-CLAP: `pip install laion-clap` "
                              "(run `python check_audio_install.py` to test)") from e
        _model = laion_clap.CLAP_Module(enable_fusion=False)
        _model.load_ckpt()   # default 630k-audioset checkpoint
    return _model


def text_directions(prompt_axes: list[list[str]]) -> np.ndarray:
    m = _load()
    flat = [p for pair in prompt_axes for p in pair]
    T = l2_normalize(np.asarray(m.get_text_embedding(flat, use_tensor=False)))
    return l2_normalize(T[0::2] - T[1::2])


def featurize(lib: dict, track_ids: list[str], params: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    have_audio = [t for t in track_ids if (AUDIO / f"{t}.m4a").exists()]
    todo = [t for t in have_audio if not (CACHE_DIR / f"{t}.npy").exists()]
    print(f"CLAP: {len(have_audio)} songs with audio, {len(todo)} to process")
    if todo:
        m = _load()
        for s in range(0, len(todo), BATCH):
            batch = todo[s:s + BATCH]
            try:
                E = m.get_audio_embedding_from_filelist(x=[str(AUDIO / f"{t}.m4a") for t in batch],
                                                        use_tensor=False)
                for t, e in zip(batch, E):
                    np.save(CACHE_DIR / f"{t}.npy", np.asarray(e, dtype=np.float32))
            except Exception as e:  # noqa: BLE001 - retry one by one so one bad file can't sink a batch
                for t in batch:
                    try:
                        e1 = m.get_audio_embedding_from_filelist(x=[str(AUDIO / f"{t}.m4a")], use_tensor=False)
                        np.save(CACHE_DIR / f"{t}.npy", np.asarray(e1[0], dtype=np.float32))
                    except Exception as e2:  # noqa: BLE001
                        print(f"  ! {t}: {type(e2).__name__}: {e2}")
            if (s // BATCH) % 10 == 0:
                print(f"  {min(s + BATCH, len(todo))}/{len(todo)}")

    ids = [t for t in have_audio if (CACHE_DIR / f"{t}.npy").exists()]
    X = np.stack([np.load(CACHE_DIR / f"{t}.npy") for t in ids]) if ids else np.zeros((0, 512), np.float32)
    if params.get("prompt_axes"):
        X = np.hstack([X, l2_normalize(X) @ text_directions(params["prompt_axes"]).T]).astype(np.float32)
    return ids, X
