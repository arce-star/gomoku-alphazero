from __future__ import annotations

import numpy as np
import pytest
import torch

from alphazero.networks.residual_net import (
    NetworkConfig,
    PolicyValueNet,
)
from alphazero.selfplay.episode import TrainingExample
from alphazero.training.replay_buffer import (
    ReplayBuffer,
)
from alphazero.training.trainer import (
    Trainer,
    TrainerConfig,
)


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


def make_buffer(
    sample_count: int = 32,
    seed: int = 42,
) -> ReplayBuffer:
    rng = np.random.default_rng(seed)

    buffer = ReplayBuffer(
        capacity=max(64, sample_count),
        seed=seed,
    )

    for index in range(sample_count):
        state = np.zeros(
            (3, 9, 9),
            dtype=np.float32,
        )

        row = index % 9
        col = (index * 3) % 9

        state[0, row, col] = 1.0

        policy = rng.random(
            81
        ).astype(np.float32)

        policy /= policy.sum()

        value = float(
            (-1.0, 0.0, 1.0)[index % 3]
        )

        buffer.add(
            TrainingExample(
                state=state,
                policy=policy,
                value=value,
            )
        )

    return buffer


def test_invalid_trainer_config() -> None:
    with pytest.raises(ValueError):
        TrainerConfig(
            learning_rate=0
        ).validate()

    with pytest.raises(ValueError):
        TrainerConfig(
            weight_decay=-1
        ).validate()

    with pytest.raises(ValueError):
        TrainerConfig(
            value_loss_weight=-1
        ).validate()

    with pytest.raises(ValueError):
        TrainerConfig(
            max_grad_norm=0
        ).validate()


def test_policy_loss_with_perfect_logits() -> None:
    target = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    logits = torch.tensor(
        [
            [20.0, -20.0, -20.0],
            [-20.0, 20.0, -20.0],
        ],
        dtype=torch.float32,
    )

    loss = Trainer.policy_loss(
        policy_logits=logits,
        target_policy=target,
    )

    assert loss.item() < 1e-5


def test_cpu_train_step_returns_finite_metrics() -> None:
    model = make_model()

    trainer = Trainer(
        model=model,
        device="cpu",
        config=TrainerConfig(
            use_amp=False,
        ),
    )

    buffer = make_buffer()
    batch = buffer.sample_torch(
        batch_size=8,
        device="cpu",
    )

    metrics = trainer.train_step(batch)

    assert np.isfinite(metrics.total_loss)
    assert np.isfinite(metrics.policy_loss)
    assert np.isfinite(metrics.value_loss)
    assert np.isfinite(metrics.policy_entropy)
    assert np.isfinite(metrics.grad_norm)

    assert metrics.total_loss >= 0
    assert metrics.policy_loss >= 0
    assert metrics.value_loss >= 0

    assert metrics.batch_size == 8
    assert metrics.learning_rate == pytest.approx(
        1e-3
    )

    assert trainer.training_steps == 1


def test_train_step_changes_parameters() -> None:
    model = make_model()

    trainer = Trainer(
        model=model,
        device="cpu",
        config=TrainerConfig(
            use_amp=False,
        ),
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    buffer = make_buffer()

    batch = buffer.sample_torch(
        batch_size=8,
        device="cpu",
    )

    trainer.train_step(batch)

    changed = any(
        not torch.equal(
            before[name],
            parameter.detach(),
        )
        for name, parameter
        in model.named_parameters()
    )

    assert changed


def test_train_from_buffer() -> None:
    model = make_model()

    trainer = Trainer(
        model=model,
        device="cpu",
        config=TrainerConfig(
            use_amp=False,
        ),
    )

    buffer = make_buffer(
        sample_count=32
    )

    metrics = trainer.train_from_buffer(
        replay_buffer=buffer,
        batch_size=8,
        steps=3,
    )

    assert metrics.steps == 3
    assert metrics.batch_size == pytest.approx(8.0)

    assert np.isfinite(metrics.total_loss)
    assert np.isfinite(metrics.grad_norm)

    assert trainer.training_steps == 3

    metrics_dict = metrics.as_dict()

    assert "loss/total" in metrics_dict
    assert "loss/policy" in metrics_dict
    assert "loss/value" in metrics_dict


def test_train_from_buffer_invalid_steps() -> None:
    trainer = Trainer(
        model=make_model(),
        device="cpu",
        config=TrainerConfig(
            use_amp=False,
        ),
    )

    buffer = make_buffer()

    with pytest.raises(ValueError):
        trainer.train_from_buffer(
            replay_buffer=buffer,
            batch_size=8,
            steps=0,
        )


def test_batch_on_wrong_device_is_rejected() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA 不可用")

    trainer = Trainer(
        model=make_model(),
        device="cuda",
        config=TrainerConfig(
            use_amp=False,
        ),
    )

    buffer = make_buffer()

    cpu_batch = buffer.sample_torch(
        batch_size=8,
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="设备",
    ):
        trainer.train_step(cpu_batch)


def test_optimizer_state_can_be_restored() -> None:
    model1 = make_model()

    trainer1 = Trainer(
        model=model1,
        device="cpu",
        config=TrainerConfig(
            use_amp=False,
        ),
    )

    buffer = make_buffer()

    batch = buffer.sample_torch(
        batch_size=8,
        device="cpu",
    )

    trainer1.train_step(batch)

    optimizer_state = (
        trainer1.optimizer_state_dict()
    )

    model2 = make_model()

    trainer2 = Trainer(
        model=model2,
        device="cpu",
        config=TrainerConfig(
            use_amp=False,
        ),
    )

    trainer2.load_optimizer_state_dict(
        optimizer_state
    )

    assert len(
        trainer2.optimizer.state
    ) > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA 不可用",
)
@pytest.mark.parametrize(
    "use_amp",
    [False, True],
)
def test_cuda_train_step(
    use_amp: bool,
) -> None:
    model = make_model()

    trainer = Trainer(
        model=model,
        device="cuda",
        config=TrainerConfig(
            use_amp=use_amp,
        ),
    )

    buffer = make_buffer(
        sample_count=32
    )

    batch = buffer.sample_torch(
        batch_size=16,
        device="cuda",
    )

    metrics = trainer.train_step(batch)

    torch.cuda.synchronize()

    assert np.isfinite(metrics.total_loss)
    assert np.isfinite(metrics.grad_norm)
    assert trainer.training_steps == 1

    assert next(
        model.parameters()
    ).device.type == "cuda"

    assert trainer.use_amp is use_amp
