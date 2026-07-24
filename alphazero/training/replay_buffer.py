from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch import Tensor

from alphazero.selfplay.episode import TrainingExample


@dataclass(frozen=True)
class ReplayBatch:
    """
    从 Replay Buffer 中采样得到的 NumPy batch。

    states:
        shape = [batch, channels, board_size, board_size]

    policies:
        shape = [batch, action_size]

    values:
        shape = [batch, 1]
    """

    states: np.ndarray
    policies: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        states = np.ascontiguousarray(
            self.states,
            dtype=np.float32,
        )

        policies = np.ascontiguousarray(
            self.policies,
            dtype=np.float32,
        )

        values = np.ascontiguousarray(
            self.values,
            dtype=np.float32,
        )

        if states.ndim != 4:
            raise ValueError(
                "states 必须是四维数组 "
                "[batch, channels, height, width]"
            )

        if policies.ndim != 2:
            raise ValueError(
                "policies 必须是二维数组 "
                "[batch, action_size]"
            )

        if values.ndim != 2:
            raise ValueError(
                "values 必须是二维数组 [batch, 1]"
            )

        batch_size = states.shape[0]

        if policies.shape[0] != batch_size:
            raise ValueError(
                "states 和 policies 的 batch 大小不一致"
            )

        if values.shape != (batch_size, 1):
            raise ValueError(
                f"values 形状必须是 {(batch_size, 1)}，"
                f"实际为 {values.shape}"
            )

        if not np.all(np.isfinite(states)):
            raise ValueError(
                "states 包含 NaN 或 Inf"
            )

        if not np.all(np.isfinite(policies)):
            raise ValueError(
                "policies 包含 NaN 或 Inf"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "values 包含 NaN 或 Inf"
            )

        object.__setattr__(self, "states", states)
        object.__setattr__(self, "policies", policies)
        object.__setattr__(self, "values", values)

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0])

    def to_torch(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ) -> "TorchReplayBatch":
        """
        将 NumPy batch 转换为 PyTorch Tensor。
        """
        device = torch.device(device)

        states = torch.from_numpy(
            self.states
        ).to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )

        policies = torch.from_numpy(
            self.policies
        ).to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )

        values = torch.from_numpy(
            self.values
        ).to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )

        return TorchReplayBatch(
            states=states,
            policies=policies,
            values=values,
        )


@dataclass(frozen=True)
class TorchReplayBatch:
    """
    PyTorch 版本的训练 batch。
    """

    states: Tensor
    policies: Tensor
    values: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0])


class ReplayBuffer:
    """
    固定容量的均匀采样循环经验池。

    当样本数量超过 capacity 时，最旧的样本会被自动删除。

    第一版 AlphaZero 使用均匀采样，不实现 prioritized replay。
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        capacity: int,
        seed: Optional[int] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(
                "capacity 必须大于 0"
            )

        self.capacity = int(capacity)

        self._examples: deque[TrainingExample] = deque(
            maxlen=self.capacity
        )

        self._rng = np.random.default_rng(seed)

        self._state_shape: Optional[
            tuple[int, ...]
        ] = None

        self._policy_shape: Optional[
            tuple[int, ...]
        ] = None

        # 记录从创建以来总共写入多少样本，
        # 即使旧样本被循环覆盖也继续增长。
        self.total_added = 0

    def __len__(self) -> int:
        return len(self._examples)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def is_full(self) -> bool:
        return len(self) == self.capacity

    @property
    def state_shape(
        self,
    ) -> Optional[tuple[int, ...]]:
        return self._state_shape

    @property
    def policy_shape(
        self,
    ) -> Optional[tuple[int, ...]]:
        return self._policy_shape

    def clear(self) -> None:
        """
        清空经验池和形状信息。

        total_added 保留累计值，用于训练统计。
        """
        self._examples.clear()
        self._state_shape = None
        self._policy_shape = None

    def add(
        self,
        example: TrainingExample,
    ) -> None:
        """
        添加一条训练样本。
        """
        if not isinstance(
            example,
            TrainingExample,
        ):
            raise TypeError(
                "example 必须是 TrainingExample"
            )

        self._validate_example_shape(example)

        # 创建新的 TrainingExample 副本，
        # 避免调用方对象后续发生变化。
        stored_example = TrainingExample(
            state=example.state,
            policy=example.policy,
            value=example.value,
        )

        self._examples.append(stored_example)
        self.total_added += 1

    def extend(
        self,
        examples: Iterable[TrainingExample],
    ) -> int:
        """
        批量添加训练样本。

        返回本次添加的样本数量。
        """
        count = 0

        for example in examples:
            self.add(example)
            count += 1

        return count

    def sample(
        self,
        batch_size: int,
        *,
        replace: bool = False,
    ) -> ReplayBatch:
        """
        均匀随机采样一个 batch。

        replace=False:
            无放回采样，batch_size 不能超过当前样本数。

        replace=True:
            有放回采样，允许 batch_size 超过当前样本数。
        """
        if batch_size <= 0:
            raise ValueError(
                "batch_size 必须大于 0"
            )

        current_size = len(self)

        if current_size == 0:
            raise ValueError(
                "Replay Buffer 为空，不能采样"
            )

        if (
            not replace
            and batch_size > current_size
        ):
            raise ValueError(
                f"无放回采样时 batch_size={batch_size} "
                f"不能超过当前样本数 {current_size}"
            )

        indices = self._rng.choice(
            current_size,
            size=batch_size,
            replace=replace,
        )

        examples_list = list(self._examples)

        selected = [
            examples_list[int(index)]
            for index in indices
        ]

        states = np.stack(
            [
                example.state
                for example in selected
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        policies = np.stack(
            [
                example.policy
                for example in selected
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        values = np.asarray(
            [
                example.value
                for example in selected
            ],
            dtype=np.float32,
        ).reshape(-1, 1)

        return ReplayBatch(
            states=states,
            policies=policies,
            values=values,
        )

    def sample_torch(
        self,
        batch_size: int,
        device: torch.device | str,
        *,
        replace: bool = False,
        non_blocking: bool = False,
    ) -> TorchReplayBatch:
        """
        采样并直接转换为 PyTorch Tensor。
        """
        batch = self.sample(
            batch_size=batch_size,
            replace=replace,
        )

        return batch.to_torch(
            device=device,
            non_blocking=non_blocking,
        )

    def examples(self) -> tuple[TrainingExample, ...]:
        """
        按从旧到新的顺序返回当前样本快照。
        """
        return tuple(self._examples)

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        将 Replay Buffer 保存为压缩 NPZ 文件。

        注意：
            大规模正式训练时，Replay Buffer 可能很大。
            不应每个训练 step 都保存，只在固定 iteration
            或正常退出时保存。
        """
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if len(self) == 0:
            raise ValueError(
                "Replay Buffer 为空，不能保存"
            )

        states = np.stack(
            [
                example.state
                for example in self._examples
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        policies = np.stack(
            [
                example.policy
                for example in self._examples
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        values = np.asarray(
            [
                example.value
                for example in self._examples
            ],
            dtype=np.float32,
        )

        # 先写临时文件，成功后再原子替换目标文件，
        # 减少中途中断造成文件损坏的风险。
        temporary_path = path.with_name(
            path.name + ".tmp.npz"
        )

        np.savez_compressed(
            temporary_path,
            format_version=np.asarray(
                [self.FORMAT_VERSION],
                dtype=np.int64,
            ),
            capacity=np.asarray(
                [self.capacity],
                dtype=np.int64,
            ),
            total_added=np.asarray(
                [self.total_added],
                dtype=np.int64,
            ),
            states=states,
            policies=policies,
            values=values,
        )

        temporary_path.replace(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        seed: Optional[int] = None,
        capacity: Optional[int] = None,
    ) -> "ReplayBuffer":
        """
        从 NPZ 文件恢复 Replay Buffer。

        capacity:
            None 时使用文件中保存的容量。
            指定时使用新容量；如果新容量更小，只保留最新样本。
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Replay Buffer 文件不存在：{path}"
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            required_keys = {
                "format_version",
                "capacity",
                "total_added",
                "states",
                "policies",
                "values",
            }

            missing_keys = (
                required_keys - set(data.files)
            )

            if missing_keys:
                raise ValueError(
                    "Replay Buffer 文件缺少字段："
                    f"{sorted(missing_keys)}"
                )

            format_version = int(
                data["format_version"][0]
            )

            if format_version != cls.FORMAT_VERSION:
                raise ValueError(
                    "不支持的 Replay Buffer 格式版本："
                    f"{format_version}"
                )

            saved_capacity = int(
                data["capacity"][0]
            )

            total_added = int(
                data["total_added"][0]
            )

            states = np.asarray(
                data["states"],
                dtype=np.float32,
            )

            policies = np.asarray(
                data["policies"],
                dtype=np.float32,
            )

            values = np.asarray(
                data["values"],
                dtype=np.float32,
            )

        if states.ndim != 4:
            raise ValueError(
                "保存文件中的 states 形状错误"
            )

        if policies.ndim != 2:
            raise ValueError(
                "保存文件中的 policies 形状错误"
            )

        sample_count = states.shape[0]

        if policies.shape[0] != sample_count:
            raise ValueError(
                "states 和 policies 样本数不一致"
            )

        if values.shape != (sample_count,):
            raise ValueError(
                "values 形状错误"
            )

        target_capacity = (
            saved_capacity
            if capacity is None
            else int(capacity)
        )

        if target_capacity <= 0:
            raise ValueError(
                "capacity 必须大于 0"
            )

        buffer = cls(
            capacity=target_capacity,
            seed=seed,
        )

        # 若新容量更小，仅保留最新的 target_capacity 条。
        start_index = max(
            0,
            sample_count - target_capacity,
        )

        for index in range(
            start_index,
            sample_count,
        ):
            buffer.add(
                TrainingExample(
                    state=states[index],
                    policy=policies[index],
                    value=float(values[index]),
                )
            )

        # add 会修改 total_added，因此恢复为文件记录的累计值。
        buffer.total_added = total_added

        return buffer

    def _validate_example_shape(
        self,
        example: TrainingExample,
    ) -> None:
        if example.state.ndim != 3:
            raise ValueError(
                "example.state 必须是三维数组"
            )

        if example.policy.ndim != 1:
            raise ValueError(
                "example.policy 必须是一维数组"
            )

        state_shape = tuple(
            int(value)
            for value in example.state.shape
        )

        policy_shape = tuple(
            int(value)
            for value in example.policy.shape
        )

        if self._state_shape is None:
            self._state_shape = state_shape
        elif state_shape != self._state_shape:
            raise ValueError(
                "state 形状与 Replay Buffer 不一致："
                f"期望 {self._state_shape}，"
                f"实际 {state_shape}"
            )

        if self._policy_shape is None:
            self._policy_shape = policy_shape
        elif policy_shape != self._policy_shape:
            raise ValueError(
                "policy 形状与 Replay Buffer 不一致："
                f"期望 {self._policy_shape}，"
                f"实际 {policy_shape}"
            )

        policy_sum = float(
            example.policy.sum()
        )

        if not np.isclose(
            policy_sum,
            1.0,
            atol=1e-5,
        ):
            raise ValueError(
                "策略目标的概率和必须为 1，"
                f"实际为 {policy_sum}"
            )

        if np.any(example.policy < 0):
            raise ValueError(
                "策略目标不能包含负数"
            )
