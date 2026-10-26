"""Atomic checkpoints containing only tensors and weights_only-safe primitives."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .labels import NUM_CLASSES, validate_class_to_idx


def capture_rng_state() -> dict[str, Any]:
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": state[0],
            "keys": state[1].tolist(),
            "position": state[2],
            "has_gauss": state[3],
            "cached_gaussian": state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state(
        (
            ns["algorithm"],
            np.asarray(ns["keys"], dtype=np.uint32),
            ns["position"],
            ns["has_gauss"],
            ns["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def atomic_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format/version")
    for key in ("model", "config", "class_to_idx", "data_preprocessing"):
        if key not in value:
            raise ValueError(f"Checkpoint is missing {key}")
    validate_class_to_idx(value["class_to_idx"], context="Checkpoint class_to_idx")
    if value.get("task", "binary_classification") != "binary_classification":
        raise ValueError("Checkpoint task must be binary_classification")
    num_classes = value.get("num_classes", NUM_CLASSES)
    if type(num_classes) is not int or num_classes != NUM_CLASSES:
        raise ValueError("Checkpoint must have exactly two classes")
    return value
