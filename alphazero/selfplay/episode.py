from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from alphazero.games.base import Game, GameState
from alphazero.games.symmetry import generate_symmetries
from alphazero.mcts.search import MCTS


@dataclass(frozen=True)
class SelfPlayConfig:
    """
    单局自对弈配置。

    temperature_moves:
        前多少步使用 sampling_temperature。

        达到该步数后，使用 temperature=0，
        即选择访问次数最高的动作。

    sampling_temperature:
        开局阶段访问次数分布的温度。

    add_root_noise:
        是否给每一步搜索的根节点加入 Dirichlet 噪声。

    augment_symmetries:
        是否将每个原始样本扩充为 8 个对称样本。

    max_moves:
        最大落子数。None 表示使用动作空间大小。
        主要用于安全防护和测试。
    """

    temperature_moves: int = 10
    sampling_temperature: float = 1.0
    add_root_noise: bool = True
    augment_symmetries: bool = True
    max_moves: Optional[int] = None

    def validate(self) -> None:
        if self.temperature_moves < 0:
            raise ValueError(
                "temperature_moves 不能小于 0"
            )

        if self.sampling_temperature <= 0:
            raise ValueError(
                "sampling_temperature 必须大于 0"
            )

        if self.max_moves is not None:
            if self.max_moves <= 0:
                raise ValueError(
                    "max_moves 必须大于 0"
                )


@dataclass(frozen=True)
class TrainingExample:
    """
    一条 AlphaZero 训练样本。

    state:
        当前玩家视角下的局面编码。
        shape = [channels, board_size, board_size]

    policy:
        MCTS 根节点访问次数生成的策略目标。
        shape = [action_size]

    value:
        最终比赛结果，从该 state 的当前玩家视角计算。
        取值为 -1.0、0.0、1.0。
    """

    state: np.ndarray
    policy: np.ndarray
    value: float

    def __post_init__(self) -> None:
        state = np.ascontiguousarray(
            self.state,
            dtype=np.float32,
        )

        policy = np.ascontiguousarray(
            self.policy,
            dtype=np.float32,
        )

        value = float(self.value)

        if not np.all(np.isfinite(state)):
            raise ValueError(
                "state 包含 NaN 或 Inf"
            )

        if not np.all(np.isfinite(policy)):
            raise ValueError(
                "policy 包含 NaN 或 Inf"
            )

        if not np.isfinite(value):
            raise ValueError(
                "value 必须是有限数值"
            )

        if value < -1.0 or value > 1.0:
            raise ValueError(
                "value 必须位于 [-1, 1]"
            )

        # 保存副本，避免外部修改训练样本。
        state = state.copy()
        policy = policy.copy()

        state.setflags(write=False)
        policy.setflags(write=False)

        object.__setattr__(self, "state", state)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class EpisodeResult:
    """
    一局自对弈结果。
    """

    examples: list[TrainingExample]
    winner: int
    move_count: int
    final_state: GameState

    @property
    def black_won(self) -> bool:
        return self.winner == 1

    @property
    def white_won(self) -> bool:
        return self.winner == -1

    @property
    def is_draw(self) -> bool:
        return self.winner == 0


@dataclass
class _PendingExample:
    """
    比赛尚未结束时暂存的样本。

    player 保存该局面下的当前行动方，
    用于比赛结束后生成正确的 value target。
    """

    state: np.ndarray
    policy: np.ndarray
    player: int


def play_episode(
    game: Game,
    mcts: MCTS,
    config: SelfPlayConfig | None = None,
) -> EpisodeResult:
    """
    运行一局完整的 AlphaZero 自对弈。

    每一步执行：
        1. 编码当前状态
        2. 运行 MCTS
        3. 保存 MCTS visit policy
        4. 根据温度选择动作
        5. 执行动作

    比赛结束后，将最终胜负转换成每个历史状态当前玩家
    视角下的 value target。
    """
    if config is None:
        config = SelfPlayConfig()

    config.validate()

    if mcts.game is not game:
        raise ValueError(
            "mcts.game 必须与传入的 game 是同一个对象"
        )

    state = game.initial_state()

    pending_examples: list[
        _PendingExample
    ] = []

    maximum_moves = (
        game.action_size
        if config.max_moves is None
        else min(
            config.max_moves,
            game.action_size,
        )
    )

    for move_index in range(maximum_moves):
        terminal_value = game.terminal_value(
            state
        )

        if terminal_value is not None:
            break

        if move_index < config.temperature_moves:
            temperature = (
                config.sampling_temperature
            )
        else:
            temperature = 0.0

        result = mcts.search(
            state,
            add_root_noise=config.add_root_noise,
            temperature=temperature,
        )

        if result.action is None:
            raise RuntimeError(
                "非终局状态下 MCTS 没有返回动作"
            )

        legal_mask = game.legal_actions(state)

        if legal_mask[result.action] <= 0:
            raise RuntimeError(
                f"MCTS 返回了非法动作 {result.action}"
            )

        policy_sum = float(
            result.visit_policy.sum()
        )

        if not np.isclose(
            policy_sum,
            1.0,
            atol=1e-5,
        ):
            raise RuntimeError(
                "MCTS visit policy 概率和不为 1，"
                f"实际为 {policy_sum}"
            )

        if np.any(
            result.visit_policy[
                legal_mask <= 0
            ] > 0
        ):
            raise RuntimeError(
                "MCTS visit policy 给非法动作分配了概率"
            )

        encoded_state = game.encode_state(
            state
        )

        pending_examples.append(
            _PendingExample(
                state=encoded_state.copy(),
                policy=result.visit_policy.copy(),
                player=state.current_player,
            )
        )

        state = game.next_state(
            state,
            result.action,
        )

    terminal_value = game.terminal_value(state)

    if terminal_value is None:
        raise RuntimeError(
            "自对弈在非终局状态下达到 max_moves。"
            "正常训练时 max_moves 应至少等于 action_size。"
        )

    winning_player = game.winner(state)

    examples = _finalize_examples(
        pending_examples=pending_examples,
        winner=winning_player,
        board_size=game.board_size,
        augment_symmetries=(
            config.augment_symmetries
        ),
    )

    return EpisodeResult(
        examples=examples,
        winner=winning_player,
        move_count=state.move_count,
        final_state=state,
    )


def _finalize_examples(
    pending_examples: list[_PendingExample],
    winner: int,
    board_size: int,
    augment_symmetries: bool,
) -> list[TrainingExample]:
    """
    为一局比赛的历史状态填充最终 value target。
    """
    if winner not in (-1, 0, 1):
        raise ValueError(
            "winner 必须是 -1、0 或 1"
        )

    finalized: list[TrainingExample] = []

    for pending in pending_examples:
        if winner == 0:
            value = 0.0
        elif winner == pending.player:
            value = 1.0
        else:
            value = -1.0

        if augment_symmetries:
            transformed_samples = (
                generate_symmetries(
                    encoded_state=pending.state,
                    policy=pending.policy,
                    board_size=board_size,
                )
            )
        else:
            transformed_samples = [
                (
                    np.ascontiguousarray(
                        pending.state,
                        dtype=np.float32,
                    ),
                    np.ascontiguousarray(
                        pending.policy,
                        dtype=np.float32,
                    ),
                )
            ]

        for state, policy in transformed_samples:
            finalized.append(
                TrainingExample(
                    state=state,
                    policy=policy,
                    value=value,
                )
            )

    return finalized
