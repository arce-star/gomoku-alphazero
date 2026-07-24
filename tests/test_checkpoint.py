from pathlib import Path
import random

import numpy as np
import pytest
import torch

from alphazero.networks.residual_net import (
    NetworkConfig,
    PolicyValueNet,
)
from alphazero.selfplay.episode import TrainingExample
from alphazero.training.replay_buffer import ReplayBuffer
from alphazero.training.trainer import Trainer, TrainerConfig
from alphazero.utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from alphazero.utils.seed import seed_everything


def make_model() -> PolicyValueNet:
    return PolicyValueNet(
        NetworkConfig(
            board_size=9,
            input_channels=3,
            channels=16,
            residual_blocks=1,
            value_hidden_channels=8,
            value_hidden_size=16,
        )
    )


def make_trainer(model: PolicyValueNet) -> Trainer:
    return Trainer(
        model=model,
        device="cpu",
        config=TrainerConfig(use_amp=False),
    )


def train_once(trainer: Trainer) -> None:
    buffer = ReplayBuffer(capacity=16, seed=42)

    for index in range(8):
        state = np.zeros((3, 9, 9), dtype=np.float32)
        state[0, index, index] = 1.0

        policy = np.zeros(81, dtype=np.float32)
        policy[index] = 1.0

        buffer.add(
            TrainingExample(
                state=state,
                policy=policy,
                value=float((-1, 0, 1)[index % 3]),
            )
        )

    batch = buffer.sample_torch(8, device="cpu")
    trainer.train_step(batch)


def test_save_and_load_complete_state(tmp_path: Path) -> None:
    seed_everything(42)

    model = make_model()
    trainer = make_trainer(model)
    train_once(trainer)
    model.eval()

    inputs = torch.randn(2, 3, 9, 9)

    with torch.no_grad():
        expected_policy, expected_value = model(inputs)

    path = tmp_path / "checkpoint.ckpt"

    save_checkpoint(
        path,
        model=model,
        trainer=trainer,
        iteration=7,
        config={"game": {"board_size": 9}},
        metrics={"loss": 1.25},
    )

    restored_model = make_model()
    restored_trainer = make_trainer(restored_model)

    metadata = load_checkpoint(
        path,
        model=restored_model,
        trainer=restored_trainer,
        map_location="cpu",
    )

    restored_model.eval()

    with torch.no_grad():
        actual_policy, actual_value = restored_model(inputs)

    assert torch.allclose(expected_policy, actual_policy)
    assert torch.allclose(expected_value, actual_value)
    assert metadata.iteration == 7
    assert metadata.training_steps == 1
    assert metadata.config["game"]["board_size"] == 9
    assert metadata.metrics["loss"] == pytest.approx(1.25)
    assert restored_trainer.training_steps == 1
    assert len(restored_trainer.optimizer.state) > 0


def test_rng_state_is_restored(tmp_path: Path) -> None:
    seed_everything(123)

    model = make_model()
    trainer = make_trainer(model)
    path = tmp_path / "rng.ckpt"

    save_checkpoint(
        path,
        model=model,
        trainer=trainer,
        iteration=0,
        config={},
    )

    expected_python = random.random()
    expected_numpy = np.random.random(4)
    expected_torch = torch.rand(4)

    random.random()
    np.random.random(4)
    torch.rand(4)

    load_checkpoint(
        path,
        model=model,
        trainer=trainer,
        restore_rng=True,
    )

    assert random.random() == expected_python
    assert np.array_equal(np.random.random(4), expected_numpy)
    assert torch.equal(torch.rand(4), expected_torch)


def test_load_model_without_trainer(tmp_path: Path) -> None:
    model = make_model()
    trainer = make_trainer(model)
    path = tmp_path / "model-only-load.ckpt"

    save_checkpoint(
        path,
        model=model,
        trainer=trainer,
        iteration=3,
        config={"name": "test"},
    )

    restored_model = make_model()

    metadata = load_checkpoint(
        path,
        model=restored_model,
        trainer=None,
        restore_rng=False,
    )

    assert metadata.iteration == 3
    assert metadata.config == {"name": "test"}


def test_trainer_and_model_must_match(tmp_path: Path) -> None:
    model = make_model()
    trainer = make_trainer(model)
    path = tmp_path / "mismatch.ckpt"

    save_checkpoint(
        path,
        model=model,
        trainer=trainer,
        iteration=0,
        config={},
    )

    other_model = make_model()

    with pytest.raises(ValueError, match="same object"):
        load_checkpoint(
            path,
            model=other_model,
            trainer=trainer,
        )


def test_atomic_save_leaves_no_temp_file(tmp_path: Path) -> None:
    model = make_model()
    trainer = make_trainer(model)
    path = tmp_path / "atomic.ckpt"

    save_checkpoint(
        path,
        model=model,
        trainer=trainer,
        iteration=0,
        config={},
    )

    assert path.exists()
    assert not (tmp_path / "atomic.ckpt.tmp").exists()


def test_invalid_iteration(tmp_path: Path) -> None:
    model = make_model()
    trainer = make_trainer(model)

    with pytest.raises(ValueError):
        save_checkpoint(
            tmp_path / "invalid.ckpt",
            model=model,
            trainer=trainer,
            iteration=-1,
            config={},
        )


def test_missing_checkpoint(tmp_path: Path) -> None:
    model = make_model()

    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            tmp_path / "missing.ckpt",
            model=model,
        )
