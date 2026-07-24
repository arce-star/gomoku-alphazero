from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from alphazero.selfplay.episode import TrainingExample
from alphazero.training.replay_buffer import (
    ReplayBatch,
    ReplayBuffer,
)


def make_example(
    index: int,
    board_size: int = 9,
) -> TrainingExample:
    """
    创建可识别的测试样本。

    state[0, 0, 0] 保存 index，方便检查循环覆盖顺序。
    """
    state = np.zeros(
        (3, board_size, board_size),
        dtype=np.float32,
    )

    state[0, 0, 0] = float(index)

    action_size = board_size * board_size

    policy = np.zeros(
        action_size,
        dtype=np.float32,
    )

    policy[index % action_size] = 1.0

    values = (-1.0, 0.0, 1.0)
    value = values[index % 3]

    return TrainingExample(
        state=state,
        policy=policy,
        value=value,
    )


def test_empty_buffer() -> None:
    buffer = ReplayBuffer(
        capacity=10,
        seed=42,
    )

    assert len(buffer) == 0
    assert buffer.is_empty
    assert not buffer.is_full
    assert buffer.total_added == 0
    assert buffer.state_shape is None
    assert buffer.policy_shape is None


def test_invalid_capacity() -> None:
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)

    with pytest.raises(ValueError):
        ReplayBuffer(capacity=-1)


def test_add_one_example() -> None:
    buffer = ReplayBuffer(
        capacity=10,
        seed=42,
    )

    buffer.add(make_example(0))

    assert len(buffer) == 1
    assert not buffer.is_empty
    assert not buffer.is_full
    assert buffer.total_added == 1

    assert buffer.state_shape == (3, 9, 9)
    assert buffer.policy_shape == (81,)


def test_extend_examples() -> None:
    buffer = ReplayBuffer(
        capacity=10,
        seed=42,
    )

    added = buffer.extend(
        make_example(index)
        for index in range(5)
    )

    assert added == 5
    assert len(buffer) == 5
    assert buffer.total_added == 5


def test_capacity_removes_oldest_examples() -> None:
    buffer = ReplayBuffer(
        capacity=3,
        seed=42,
    )

    for index in range(5):
        buffer.add(make_example(index))

    assert len(buffer) == 3
    assert buffer.is_full
    assert buffer.total_added == 5

    examples = buffer.examples()

    stored_indices = [
        int(example.state[0, 0, 0])
        for example in examples
    ]

    # 最早的 0、1 已被覆盖。
    assert stored_indices == [2, 3, 4]


def test_uniform_sample_shapes() -> None:
    buffer = ReplayBuffer(
        capacity=100,
        seed=42,
    )

    buffer.extend(
        make_example(index)
        for index in range(20)
    )

    batch = buffer.sample(
        batch_size=8
    )

    assert isinstance(batch, ReplayBatch)
    assert batch.batch_size == 8

    assert batch.states.shape == (
        8,
        3,
        9,
        9,
    )

    assert batch.policies.shape == (
        8,
        81,
    )

    assert batch.values.shape == (
        8,
        1,
    )

    assert batch.states.dtype == np.float32
    assert batch.policies.dtype == np.float32
    assert batch.values.dtype == np.float32

    assert np.allclose(
        batch.policies.sum(axis=1),
        np.ones(8),
    )

    assert np.all(
        np.isin(
            batch.values,
            [-1.0, 0.0, 1.0],
        )
    )


def test_sample_without_replacement() -> None:
    buffer = ReplayBuffer(
        capacity=10,
        seed=123,
    )

    buffer.extend(
        make_example(index)
        for index in range(10)
    )

    batch = buffer.sample(
        batch_size=10,
        replace=False,
    )

    sampled_indices = (
        batch.states[:, 0, 0, 0]
        .astype(int)
        .tolist()
    )

    assert len(set(sampled_indices)) == 10
    assert sorted(sampled_indices) == list(
        range(10)
    )


def test_sample_with_replacement() -> None:
    buffer = ReplayBuffer(
        capacity=3,
        seed=123,
    )

    buffer.extend(
        make_example(index)
        for index in range(3)
    )

    batch = buffer.sample(
        batch_size=10,
        replace=True,
    )

    assert batch.batch_size == 10


def test_cannot_sample_empty_buffer() -> None:
    buffer = ReplayBuffer(
        capacity=10
    )

    with pytest.raises(
        ValueError,
        match="为空",
    ):
        buffer.sample(batch_size=1)


def test_large_batch_without_replacement_fails() -> None:
    buffer = ReplayBuffer(
        capacity=10
    )

    buffer.add(make_example(0))

    with pytest.raises(
        ValueError,
        match="不能超过",
    ):
        buffer.sample(
            batch_size=2,
            replace=False,
        )


def test_invalid_batch_size() -> None:
    buffer = ReplayBuffer(
        capacity=10
    )

    buffer.add(make_example(0))

    with pytest.raises(ValueError):
        buffer.sample(batch_size=0)

    with pytest.raises(ValueError):
        buffer.sample(batch_size=-1)


def test_reproducible_sampling() -> None:
    buffer1 = ReplayBuffer(
        capacity=20,
        seed=999,
    )

    buffer2 = ReplayBuffer(
        capacity=20,
        seed=999,
    )

    examples = [
        make_example(index)
        for index in range(20)
    ]

    buffer1.extend(examples)
    buffer2.extend(examples)

    batch1 = buffer1.sample(
        batch_size=8
    )

    batch2 = buffer2.sample(
        batch_size=8
    )

    assert np.array_equal(
        batch1.states,
        batch2.states,
    )

    assert np.array_equal(
        batch1.policies,
        batch2.policies,
    )

    assert np.array_equal(
        batch1.values,
        batch2.values,
    )


def test_shape_mismatch_is_rejected() -> None:
    buffer = ReplayBuffer(
        capacity=10
    )

    buffer.add(
        make_example(
            index=0,
            board_size=9,
        )
    )

    with pytest.raises(
        ValueError,
        match="state 形状",
    ):
        buffer.add(
            make_example(
                index=1,
                board_size=5,
            )
        )


def test_clear() -> None:
    buffer = ReplayBuffer(
        capacity=10
    )

    buffer.extend(
        make_example(index)
        for index in range(5)
    )

    assert buffer.total_added == 5

    buffer.clear()

    assert len(buffer) == 0
    assert buffer.is_empty
    assert buffer.state_shape is None
    assert buffer.policy_shape is None

    # clear 不重置累计写入计数。
    assert buffer.total_added == 5


def test_numpy_batch_to_cpu_torch() -> None:
    buffer = ReplayBuffer(
        capacity=10,
        seed=42,
    )

    buffer.extend(
        make_example(index)
        for index in range(10)
    )

    batch = buffer.sample(
        batch_size=4
    )

    torch_batch = batch.to_torch(
        device="cpu"
    )

    assert torch_batch.batch_size == 4

    assert torch_batch.states.shape == (
        4,
        3,
        9,
        9,
    )

    assert torch_batch.policies.shape == (
        4,
        81,
    )

    assert torch_batch.values.shape == (
        4,
        1,
    )

    assert torch_batch.states.dtype == torch.float32
    assert torch_batch.policies.dtype == torch.float32
    assert torch_batch.values.dtype == torch.float32

    assert torch_batch.states.device.type == "cpu"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA 不可用",
)
def test_sample_torch_to_cuda() -> None:
    buffer = ReplayBuffer(
        capacity=20,
        seed=42,
    )

    buffer.extend(
        make_example(index)
        for index in range(20)
    )

    batch = buffer.sample_torch(
        batch_size=8,
        device="cuda",
    )

    assert batch.states.device.type == "cuda"
    assert batch.policies.device.type == "cuda"
    assert batch.values.device.type == "cuda"

    assert batch.states.shape == (
        8,
        3,
        9,
        9,
    )

    assert torch.isfinite(
        batch.states
    ).all()

    assert torch.isfinite(
        batch.policies
    ).all()

    assert torch.isfinite(
        batch.values
    ).all()


def test_save_and_load(
    tmp_path: Path,
) -> None:
    buffer = ReplayBuffer(
        capacity=10,
        seed=42,
    )

    buffer.extend(
        make_example(index)
        for index in range(7)
    )

    path = tmp_path / "replay.npz"
    buffer.save(path)

    assert path.exists()

    loaded = ReplayBuffer.load(
        path,
        seed=42,
    )

    assert loaded.capacity == 10
    assert len(loaded) == 7
    assert loaded.total_added == 7
    assert loaded.state_shape == (3, 9, 9)
    assert loaded.policy_shape == (81,)

    original_examples = buffer.examples()
    loaded_examples = loaded.examples()

    for original, restored in zip(
        original_examples,
        loaded_examples,
    ):
        assert np.array_equal(
            original.state,
            restored.state,
        )

        assert np.array_equal(
            original.policy,
            restored.policy,
        )

        assert original.value == restored.value


def test_load_with_smaller_capacity_keeps_latest(
    tmp_path: Path,
) -> None:
    buffer = ReplayBuffer(
        capacity=10
    )

    buffer.extend(
        make_example(index)
        for index in range(8)
    )

    path = tmp_path / "replay.npz"
    buffer.save(path)

    loaded = ReplayBuffer.load(
        path,
        capacity=3,
    )

    assert loaded.capacity == 3
    assert len(loaded) == 3
    assert loaded.total_added == 8

    indices = [
        int(example.state[0, 0, 0])
        for example in loaded.examples()
    ]

    assert indices == [5, 6, 7]


def test_cannot_save_empty_buffer(
    tmp_path: Path,
) -> None:
    buffer = ReplayBuffer(
        capacity=10
    )

    with pytest.raises(
        ValueError,
        match="为空",
    ):
        buffer.save(
            tmp_path / "empty.npz"
        )


def test_load_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
    ):
        ReplayBuffer.load(
            tmp_path / "missing.npz"
        )
