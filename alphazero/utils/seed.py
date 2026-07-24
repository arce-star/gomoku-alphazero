from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_everything(
    seed: int,
    *,
    deterministic: bool = False,
) -> None:
    """Seed Python, NumPy and PyTorch."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if seed < 0:
        raise ValueError("seed must not be negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    if deterministic:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    else:
        torch.use_deterministic_algorithms(False)


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG states for checkpointing."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(
    state: dict[str, Any],
) -> None:
    """Restore RNG states captured by capture_rng_state."""
    required_keys = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }

    missing = required_keys - set(state)

    if missing:
        raise ValueError(
            f"RNG state is missing keys: {sorted(missing)}"
        )

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    # map_location="cuda" may move RNG tensors to CUDA, but
    # PyTorch RNG restore APIs expect CPU uint8 tensors.
    torch_cpu_state = state["torch_cpu"]

    if not torch.is_tensor(torch_cpu_state):
        raise TypeError("torch_cpu RNG state must be a tensor")

    torch.set_rng_state(
        torch_cpu_state.detach().to(
            device="cpu",
            dtype=torch.uint8,
        )
    )

    cuda_state = state["torch_cuda"]

    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG state, "
                "but CUDA is unavailable"
            )

        current_devices = torch.cuda.device_count()

        if len(cuda_state) != current_devices:
            raise RuntimeError(
                "CUDA device count does not match RNG state: "
                f"{current_devices} != {len(cuda_state)}"
            )

        normalized_cuda_state = []

        for device_state in cuda_state:
            if not torch.is_tensor(device_state):
                raise TypeError(
                    "CUDA RNG state must contain tensors"
                )

            normalized_cuda_state.append(
                device_state.detach().to(
                    device="cpu",
                    dtype=torch.uint8,
                )
            )

        torch.cuda.set_rng_state_all(
            normalized_cuda_state
        )
