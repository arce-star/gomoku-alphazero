from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

from alphazero.training.trainer import Trainer
from alphazero.utils.seed import (
    capture_rng_state,
    restore_rng_state,
)


CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CheckpointMetadata:
    iteration: int
    training_steps: int
    config: dict[str, Any]
    metrics: dict[str, Any]
    path: Path


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    trainer: Trainer,
    iteration: int,
    config: Mapping[str, Any],
    metrics: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically save the complete training state."""
    if iteration < 0:
        raise ValueError("iteration must not be negative")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "iteration": int(iteration),
        "training_steps": int(trainer.training_steps),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer_state_dict(),
        "scaler_state_dict": trainer.scaler_state_dict(),
        "config": dict(config),
        "metrics": dict(metrics or {}),
        "rng_state": capture_rng_state(),
    }

    temporary_path = path.with_name(path.name + ".tmp")

    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    trainer: Optional[Trainer] = None,
    map_location: torch.device | str = "cpu",
    restore_rng: bool = True,
    strict: bool = True,
) -> CheckpointMetadata:
    """Load model and optional trainer state from a checkpoint."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root object must be a dictionary")

    required_keys = {
        "format_version",
        "iteration",
        "training_steps",
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "config",
        "metrics",
        "rng_state",
    }

    missing_keys = required_keys - set(checkpoint)

    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing keys: {sorted(missing_keys)}"
        )

    format_version = int(checkpoint["format_version"])

    if format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported checkpoint format version: "
            f"{format_version}"
        )

    iteration = int(checkpoint["iteration"])
    training_steps = int(checkpoint["training_steps"])

    if iteration < 0 or training_steps < 0:
        raise ValueError("Checkpoint contains invalid counters")

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )

    if trainer is not None:
        if trainer.model is not model:
            raise ValueError(
                "trainer.model and model must be the same object"
            )

        trainer.load_optimizer_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        trainer.load_scaler_state_dict(
            checkpoint["scaler_state_dict"]
        )
        trainer.training_steps = training_steps

    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])

    return CheckpointMetadata(
        iteration=iteration,
        training_steps=training_steps,
        config=dict(checkpoint["config"]),
        metrics=dict(checkpoint["metrics"]),
        path=path,
    )
