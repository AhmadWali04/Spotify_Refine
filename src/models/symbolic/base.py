"""The interface both symbolic models share (PRD3 §6), and the config toggle that picks one.

    model = make(load_config())          # configs/symbolic.yaml: model: markov | vector
    model.fit(songs, labels)             # labels[i]: set of playlist / genre names
    model.scores(songs)                  # n x |classes_|, higher = better fit (log-likelihoods for markov)
    model.predict_proba(songs)           # softmax of the scores; the harness recalibrates with a fitted T
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from src import paths
from src.midi.represent import SongRep

CONFIG = paths.CONFIGS / "symbolic.yaml"


class SymbolicModel(Protocol):
    classes_: list[str]

    def fit(self, songs: list[SongRep], labels: list[set[str]]) -> "SymbolicModel": ...
    def scores(self, songs: list[SongRep]) -> np.ndarray: ...
    def predict_proba(self, songs: list[SongRep]) -> np.ndarray: ...


def class_index(labels: list[set[str]]) -> tuple[list[str], list[set[int]]]:
    classes = sorted(set().union(*labels))
    idx = {c: i for i, c in enumerate(classes)}
    return classes, [{idx[c] for c in ls} for ls in labels]


def load_config(path: Path = CONFIG, model: str | None = None, assignment: str | None = None) -> dict:
    """The YAML config, with the CLI overrides (--symbolic-model, --assignment) applied."""
    cfg = yaml.safe_load(Path(path).read_text())
    if model:
        cfg["model"] = model
    if assignment:
        cfg.setdefault("vector", {})["assignment_model"] = assignment
    return cfg


def make(cfg: dict) -> SymbolicModel:
    from src.models.symbolic.markov import MarkovModel
    from src.models.symbolic.vector import VectorModel
    if cfg["model"] == "markov":
        return MarkovModel(**cfg.get("markov", {}))
    if cfg["model"] == "vector":
        return VectorModel(**cfg.get("vector", {}), transpose=cfg.get("augment", {}).get("transpose", False))
    raise ValueError(f"model must be 'markov' or 'vector', got {cfg['model']!r}")
