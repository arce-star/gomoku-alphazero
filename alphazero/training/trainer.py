from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from alphazero.training.replay_buffer import (
    ReplayBuffer,
    TorchReplayBatch,
)


@dataclass(frozen=True)
class TrainerConfig:
    """
    Policy-Value 网络训练参数。
    """

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    max_grad_norm: Optional[float] = 5.0
    use_amp: bool = True

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate 必须大于 0"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay 不能小于 0"
            )

        if self.value_loss_weight < 0:
            raise ValueError(
                "value_loss_weight 不能小于 0"
            )

        if (
            self.max_grad_norm is not None
            and self.max_grad_norm <= 0
        ):
            raise ValueError(
                "max_grad_norm 必须大于 0 或为 None"
            )


@dataclass(frozen=True)
class TrainMetrics:
    """
    单个训练 step 的指标。
    """

    total_loss: float
    policy_loss: float
    value_loss: float
    policy_entropy: float
    predicted_value_mean: float
    target_value_mean: float
    grad_norm: float
    learning_rate: float
    batch_size: int

    def as_dict(self) -> dict[str, float]:
        return {
            "loss/total": self.total_loss,
            "loss/policy": self.policy_loss,
            "loss/value": self.value_loss,
            "policy/entropy": self.policy_entropy,
            "value/predicted_mean": self.predicted_value_mean,
            "value/target_mean": self.target_value_mean,
            "optimization/grad_norm": self.grad_norm,
            "optimization/learning_rate": self.learning_rate,
            "batch_size": float(self.batch_size),
        }


@dataclass(frozen=True)
class AveragedTrainMetrics:
    """
    多个训练 step 的平均指标。
    """

    total_loss: float
    policy_loss: float
    value_loss: float
    policy_entropy: float
    predicted_value_mean: float
    target_value_mean: float
    grad_norm: float
    learning_rate: float
    batch_size: float
    steps: int

    def as_dict(self) -> dict[str, float]:
        return {
            "loss/total": self.total_loss,
            "loss/policy": self.policy_loss,
            "loss/value": self.value_loss,
            "policy/entropy": self.policy_entropy,
            "value/predicted_mean": self.predicted_value_mean,
            "value/target_mean": self.target_value_mean,
            "optimization/grad_norm": self.grad_norm,
            "optimization/learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "steps": float(self.steps),
        }


class Trainer:
    """
    AlphaZero Policy-Value 网络训练器。
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,
        config: TrainerConfig | None = None,
        optimizer: Optional[
            torch.optim.Optimizer
        ] = None,
    ) -> None:
        if config is None:
            config = TrainerConfig()

        config.validate()

        self.model = model

        requested_device = torch.device(device)

        if requested_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "请求使用 CUDA，但当前 CUDA 不可用"
                )

            # 将未指定编号的 cuda 规范化为 cuda:0，
            # 避免 torch.device("cuda") 与 Tensor 的
            # torch.device("cuda:0") 比较时被误判为不同设备。
            if requested_device.index is None:
                requested_device = torch.device(
                    "cuda",
                    torch.cuda.current_device(),
                )

        self.device = requested_device
        self.config = config

        self.model.to(self.device)

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        self.optimizer = optimizer

        # AMP 仅在 CUDA 上启用。
        self.use_amp = (
            config.use_amp
            and self.device.type == "cuda"
        )

        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
            # 默认 65536 对随机初始化的小型策略价值网络
            # 有时过大，首个 step 可能出现 FP16 梯度溢出。
            init_scale=1024.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )

        self.training_steps = 0

    @staticmethod
    def policy_loss(
        policy_logits: torch.Tensor,
        target_policy: torch.Tensor,
    ) -> torch.Tensor:
        """
        软标签策略交叉熵。

        target_policy 是 MCTS visit policy，
        不一定是 one-hot。
        """
        if policy_logits.ndim != 2:
            raise ValueError(
                "policy_logits 必须是二维张量"
            )

        if target_policy.shape != policy_logits.shape:
            raise ValueError(
                "target_policy 与 policy_logits 形状不一致："
                f"{tuple(target_policy.shape)} != "
                f"{tuple(policy_logits.shape)}"
            )

        log_probabilities = torch.log_softmax(
            policy_logits,
            dim=1,
        )

        return -(
            target_policy * log_probabilities
        ).sum(dim=1).mean()

    def train_step(
        self,
        batch: TorchReplayBatch,
    ) -> TrainMetrics:
        """
        使用一个 batch 更新模型一次。

        batch 必须已经位于 Trainer 的目标设备上。
        """
        self._validate_batch(batch)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=self.device.type,
            enabled=self.use_amp,
        ):
            policy_logits, predicted_values = (
                self.model(batch.states)
            )

            policy_loss = self.policy_loss(
                policy_logits=policy_logits,
                target_policy=batch.policies,
            )

            value_loss = nn.functional.mse_loss(
                predicted_values,
                batch.values,
            )

            total_loss = (
                policy_loss
                + self.config.value_loss_weight
                * value_loss
            )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "训练损失出现 NaN 或 Inf"
            )

        self.grad_scaler.scale(
            total_loss
        ).backward()

        # 在检查和裁剪梯度前解除 AMP 缩放。
        self.grad_scaler.unscale_(
            self.optimizer
        )

        grad_norm = self._gradient_norm()

        if not np.isfinite(grad_norm):
            self.optimizer.zero_grad(
                set_to_none=True
            )
            raise FloatingPointError(
                "梯度范数出现 NaN 或 Inf"
            )

        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.max_grad_norm,
            )

        self.grad_scaler.step(
            self.optimizer
        )
        self.grad_scaler.update()

        self.training_steps += 1

        with torch.no_grad():
            probabilities = torch.softmax(
                policy_logits.float(),
                dim=1,
            )

            log_probabilities = torch.log_softmax(
                policy_logits.float(),
                dim=1,
            )

            entropy = -(
                probabilities * log_probabilities
            ).sum(dim=1).mean()

            predicted_value_mean = (
                predicted_values.float().mean()
            )

            target_value_mean = (
                batch.values.float().mean()
            )

        learning_rate = float(
            self.optimizer.param_groups[0]["lr"]
        )

        return TrainMetrics(
            total_loss=float(
                total_loss.detach().float().item()
            ),
            policy_loss=float(
                policy_loss.detach().float().item()
            ),
            value_loss=float(
                value_loss.detach().float().item()
            ),
            policy_entropy=float(
                entropy.detach().float().item()
            ),
            predicted_value_mean=float(
                predicted_value_mean.item()
            ),
            target_value_mean=float(
                target_value_mean.item()
            ),
            grad_norm=float(grad_norm),
            learning_rate=learning_rate,
            batch_size=batch.batch_size,
        )

    def train_from_buffer(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        steps: int,
        *,
        replace: bool = False,
    ) -> AveragedTrainMetrics:
        """
        从 Replay Buffer 重复采样并执行多个训练 step。
        """
        if steps <= 0:
            raise ValueError(
                "steps 必须大于 0"
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size 必须大于 0"
            )

        metrics_list: list[TrainMetrics] = []

        for _ in range(steps):
            batch = replay_buffer.sample_torch(
                batch_size=batch_size,
                device=self.device,
                replace=replace,
            )

            metrics = self.train_step(batch)
            metrics_list.append(metrics)

        return self._average_metrics(
            metrics_list
        )

    def optimizer_state_dict(self) -> dict:
        """
        返回优化器状态，后续用于 Checkpoint。
        """
        return self.optimizer.state_dict()

    def scaler_state_dict(self) -> dict:
        """
        返回 AMP GradScaler 状态。
        """
        return self.grad_scaler.state_dict()

    def load_optimizer_state_dict(
        self,
        state_dict: dict,
    ) -> None:
        self.optimizer.load_state_dict(
            state_dict
        )
        self._move_optimizer_state_to_device()

    def load_scaler_state_dict(
        self,
        state_dict: dict,
    ) -> None:
        self.grad_scaler.load_state_dict(
            state_dict
        )

    def _validate_batch(
        self,
        batch: TorchReplayBatch,
    ) -> None:
        if not isinstance(
            batch,
            TorchReplayBatch,
        ):
            raise TypeError(
                "batch 必须是 TorchReplayBatch"
            )

        if batch.states.ndim != 4:
            raise ValueError(
                "batch.states 必须是四维张量"
            )

        if batch.policies.ndim != 2:
            raise ValueError(
                "batch.policies 必须是二维张量"
            )

        if batch.values.ndim != 2:
            raise ValueError(
                "batch.values 必须是二维张量"
            )

        batch_size = batch.states.shape[0]

        if batch_size <= 0:
            raise ValueError(
                "batch 不能为空"
            )

        if batch.policies.shape[0] != batch_size:
            raise ValueError(
                "states 和 policies 的 batch 大小不一致"
            )

        if batch.values.shape != (
            batch_size,
            1,
        ):
            raise ValueError(
                "values 形状必须是 [batch, 1]"
            )

        tensors = (
            batch.states,
            batch.policies,
            batch.values,
        )

        for tensor in tensors:
            if tensor.device != self.device:
                raise ValueError(
                    "batch 所在设备与 Trainer 不一致："
                    f"{tensor.device} != {self.device}"
                )

            if not tensor.is_floating_point():
                raise TypeError(
                    "训练 batch 必须是浮点张量"
                )

            if not torch.isfinite(tensor).all():
                raise ValueError(
                    "训练 batch 包含 NaN 或 Inf"
                )

        policy_sums = batch.policies.sum(dim=1)

        if not torch.allclose(
            policy_sums,
            torch.ones_like(policy_sums),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError(
                "每条策略目标的概率和必须为 1"
            )

        if torch.any(batch.policies < 0):
            raise ValueError(
                "策略目标不能包含负数"
            )

        if torch.any(batch.values < -1.0):
            raise ValueError(
                "价值目标不能小于 -1"
            )

        if torch.any(batch.values > 1.0):
            raise ValueError(
                "价值目标不能大于 1"
            )

    def _gradient_norm(self) -> float:
        """
        计算裁剪前的全局 L2 梯度范数。
        """
        squared_norm = 0.0

        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue

            gradient = parameter.grad.detach()

            parameter_norm = float(
                torch.linalg.vector_norm(
                    gradient.float()
                ).item()
            )

            squared_norm += (
                parameter_norm * parameter_norm
            )

        return float(
            squared_norm ** 0.5
        )

    @staticmethod
    def _average_metrics(
        metrics_list: list[TrainMetrics],
    ) -> AveragedTrainMetrics:
        if not metrics_list:
            raise ValueError(
                "metrics_list 不能为空"
            )

        def average(
            attribute: str,
        ) -> float:
            return float(
                np.mean(
                    [
                        getattr(metric, attribute)
                        for metric in metrics_list
                    ]
                )
            )

        return AveragedTrainMetrics(
            total_loss=average("total_loss"),
            policy_loss=average("policy_loss"),
            value_loss=average("value_loss"),
            policy_entropy=average(
                "policy_entropy"
            ),
            predicted_value_mean=average(
                "predicted_value_mean"
            ),
            target_value_mean=average(
                "target_value_mean"
            ),
            grad_norm=average("grad_norm"),
            learning_rate=average(
                "learning_rate"
            ),
            batch_size=average("batch_size"),
            steps=len(metrics_list),
        )

    def _move_optimizer_state_to_device(
        self,
    ) -> None:
        """
        加载 Checkpoint 后，将优化器状态迁移到目标设备。
        """
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(
                        self.device
                    )
