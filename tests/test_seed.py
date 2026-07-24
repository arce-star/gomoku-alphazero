import random

import numpy as np
import pytest
import torch

from alphazero.utils.seed import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)


def generate_cpu_values():
    return (
        random.random(),
        np.random.random(4),
        torch.rand(4),
    )


def test_seed_everything_is_reproducible() -> None:
    seed_everything(42)
    first = generate_cpu_values()

    seed_everything(42)
    second = generate_cpu_values()

    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])


def test_different_seeds_differ() -> None:
    seed_everything(1)
    first = generate_cpu_values()

    seed_everything(2)
    second = generate_cpu_values()

    assert first[0] != second[0]
    assert not np.array_equal(first[1], second[1])
    assert not torch.equal(first[2], second[2])


def test_capture_and_restore_cpu_rng() -> None:
    seed_everything(123)

    state = capture_rng_state()
    expected = generate_cpu_values()

    generate_cpu_values()
    restore_rng_state(state)
    restored = generate_cpu_values()

    assert expected[0] == restored[0]
    assert np.array_equal(expected[1], restored[1])
    assert torch.equal(expected[2], restored[2])


def test_invalid_seed() -> None:
    with pytest.raises(TypeError):
        seed_everything("42")

    with pytest.raises(ValueError):
        seed_everything(-1)


def test_missing_rng_key() -> None:
    state = capture_rng_state()
    del state["numpy"]

    with pytest.raises(ValueError, match="missing keys"):
        restore_rng_state(state)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_capture_and_restore_cuda_rng() -> None:
    seed_everything(456)

    state = capture_rng_state()
    expected = torch.rand(8, device="cuda")

    torch.rand(8, device="cuda")
    restore_rng_state(state)
    restored = torch.rand(8, device="cuda")

    assert torch.equal(expected, restored)
