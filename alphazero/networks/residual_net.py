from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class NetworkConfig:
    """
    Policy-Value 网络配置。
    """

    board_size: int = 9
    input_channels: int = 3
    channels: int = 64
    residual_blocks: int = 5
    value_hidden_channels: int = 32
    value_hidden_size: int = 64

    @property
    def action_size(self) -> int:
        return self.board_size * self.board_size

    def validate(self) -> None:
        if self.board_size <= 0:
            raise ValueError("board_size 必须大于 0")

        if self.input_channels <= 0:
            raise ValueError("input_channels 必须大于 0")

        if self.channels <= 0:
            raise ValueError("channels 必须大于 0")

        if self.residual_blocks < 0:
            raise ValueError("residual_blocks 不能小于 0")

        if self.value_hidden_channels <= 0:
            raise ValueError("value_hidden_channels 必须大于 0")

        if self.value_hidden_size <= 0:
            raise ValueError("value_hidden_size 必须大于 0")


class ConvBlock(nn.Module):
    """
    Conv2d + BatchNorm2d + ReLU。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """
    AlphaZero 风格的基础残差块。

    输入和输出形状相同：
        [batch, channels, height, width]
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(channels)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)

        out = out + residual
        out = self.activation(out)

        return out


class PolicyHead(nn.Module):
    """
    策略头。

    每个棋盘位置输出一个 logit，因此：
        [batch, channels, H, W]
        ->
        [batch, H * W]

    这里不执行 softmax。训练损失和 MCTS 会根据各自需求处理
    policy logits。
    """

    def __init__(
        self,
        in_channels: int,
        board_size: int,
    ) -> None:
        super().__init__()

        self.board_size = board_size
        self.action_size = board_size * board_size

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        logits = self.conv(x)
        logits = logits.flatten(start_dim=1)
        return logits


class ValueHead(nn.Module):
    """
    价值头。

    输出当前待行动玩家视角下的局面价值，范围为 [-1, 1]。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(hidden_channels)
        self.activation = nn.ReLU(inplace=True)

        # 全局平均池化让价值头不依赖固定的空间展开尺寸，
        # 同时便于后续导出 ONNX。
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc1 = nn.Linear(hidden_channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        # AlphaZero 的价值目标位于 [-1, 1]。
        value = torch.tanh(x)
        return value


class PolicyValueNet(nn.Module):
    """
    五子棋 Policy-Value 残差网络。

    输入：
        states: float tensor
        shape = [batch, input_channels, board_size, board_size]

    输出：
        policy_logits:
            shape = [batch, board_size * board_size]

        value:
            shape = [batch, 1]
            range = [-1, 1]
    """

    def __init__(
        self,
        config: NetworkConfig | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = NetworkConfig()

        config.validate()
        self.config = config

        self.stem = ConvBlock(
            in_channels=config.input_channels,
            out_channels=config.channels,
            kernel_size=3,
        )

        self.residual_tower = nn.Sequential(
            *[
                ResidualBlock(config.channels)
                for _ in range(config.residual_blocks)
            ]
        )

        self.policy_head = PolicyHead(
            in_channels=config.channels,
            board_size=config.board_size,
        )

        self.value_head = ValueHead(
            in_channels=config.channels,
            hidden_channels=config.value_hidden_channels,
            hidden_size=config.value_hidden_size,
        )

        self._initialize_weights()

    @property
    def action_size(self) -> int:
        return self.config.action_size

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_input(states)

        features = self.stem(states)
        features = self.residual_tower(features)

        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return policy_logits, value

    @torch.no_grad()
    def predict(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """
        推理辅助接口。

        返回：
            policy_probs: 已执行 softmax 的策略概率
            value: [-1, 1] 的价值

        注意：
            这里只进行 softmax，不会屏蔽五子棋中的非法动作。
            非法动作掩码由 MCTS 处理。
        """
        was_training = self.training
        self.eval()

        policy_logits, value = self(states)
        policy_probs = torch.softmax(policy_logits, dim=1)

        if was_training:
            self.train()

        return policy_probs, value

    def parameter_count(self) -> int:
        """返回可训练参数数量。"""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _validate_input(self, states: Tensor) -> None:
        if not isinstance(states, Tensor):
            raise TypeError("states 必须是 torch.Tensor")

        if states.ndim != 4:
            raise ValueError(
                "states 必须是四维张量 "
                "[batch, channels, height, width]，"
                f"实际形状为 {tuple(states.shape)}"
            )

        expected_channels = self.config.input_channels
        expected_size = self.config.board_size

        if states.shape[1] != expected_channels:
            raise ValueError(
                f"输入通道数必须是 {expected_channels}，"
                f"实际得到 {states.shape[1]}"
            )

        if states.shape[2] != expected_size:
            raise ValueError(
                f"输入高度必须是 {expected_size}，"
                f"实际得到 {states.shape[2]}"
            )

        if states.shape[3] != expected_size:
            raise ValueError(
                f"输入宽度必须是 {expected_size}，"
                f"实际得到 {states.shape[3]}"
            )

        if not states.is_floating_point():
            raise TypeError(
                "states 必须是浮点张量，例如 torch.float32"
            )

    def _initialize_weights(self) -> None:
        """
        初始化卷积层、BatchNorm 和全连接层。
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # 策略头和价值头的最终输出层使用较小初始化。
        #
        # 如果对只有一个输出通道的 policy 1x1 Conv 使用
        # Kaiming fan_out 初始化，初始 logits 容易过大，导致
        # 随机网络的策略分布过于尖锐，并增加 FP16 梯度溢出风险。
        nn.init.normal_(
            self.policy_head.conv.weight,
            mean=0.0,
            std=0.01,
        )

        if self.policy_head.conv.bias is not None:
            nn.init.zeros_(
                self.policy_head.conv.bias
            )

        nn.init.normal_(
            self.value_head.fc2.weight,
            mean=0.0,
            std=0.01,
        )

        if self.value_head.fc2.bias is not None:
            nn.init.zeros_(
                self.value_head.fc2.bias
            )
