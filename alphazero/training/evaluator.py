from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from alphazero.games.base import Game, GameState
from alphazero.mcts.search import (
    MCTS,
    MCTSConfig,
    PositionEvaluator,
    TorchNetworkEvaluator,
)


@dataclass(frozen=True)
class ArenaConfig:
    """Candidate 与 best 模型的评估参数。"""

    games: int = 50
    num_simulations: int = 64
    c_puct: float = 1.5
    promotion_threshold: float = 0.55
    max_moves: Optional[int] = None

    def validate(self) -> None:
        if self.games <= 0:
            raise ValueError("games 必须大于 0")

        if self.num_simulations <= 0:
            raise ValueError(
                "num_simulations 必须大于 0"
            )

        if self.c_puct <= 0:
            raise ValueError(
                "c_puct 必须大于 0"
            )

        if not 0.0 <= self.promotion_threshold <= 1.0:
            raise ValueError(
                "promotion_threshold 必须位于 [0, 1]"
            )

        if (
            self.max_moves is not None
            and self.max_moves <= 0
        ):
            raise ValueError(
                "max_moves 必须大于 0 或为 None"
            )


@dataclass(frozen=True)
class ArenaGameResult:
    """单局 Arena 对局结果。"""

    winner: int
    candidate_player: int
    move_count: int
    final_state: GameState

    @property
    def candidate_result(self) -> int:
        """
        Candidate 视角结果：

        1  candidate 获胜
        0  和棋
        -1 candidate 失败
        """
        if self.winner == 0:
            return 0

        if self.winner == self.candidate_player:
            return 1

        return -1


@dataclass(frozen=True)
class ArenaResult:
    """完整 Arena 评估结果。"""

    candidate_wins: int
    best_wins: int
    draws: int
    games: int
    promotion_threshold: float
    game_results: tuple[ArenaGameResult, ...]

    @property
    def candidate_score(self) -> float:
        if self.games == 0:
            return 0.0

        return (
            self.candidate_wins
            + 0.5 * self.draws
        ) / self.games

    @property
    def candidate_decisive_win_rate(self) -> float:
        decisive_games = (
            self.candidate_wins + self.best_wins
        )

        if decisive_games == 0:
            return 0.0

        return self.candidate_wins / decisive_games

    @property
    def should_promote(self) -> bool:
        return (
            self.candidate_score
            >= self.promotion_threshold
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "arena/games": float(self.games),
            "arena/candidate_wins": float(
                self.candidate_wins
            ),
            "arena/best_wins": float(
                self.best_wins
            ),
            "arena/draws": float(self.draws),
            "arena/candidate_score": (
                self.candidate_score
            ),
            "arena/candidate_decisive_win_rate": (
                self.candidate_decisive_win_rate
            ),
            "arena/promotion_threshold": (
                self.promotion_threshold
            ),
            "arena/should_promote": float(
                self.should_promote
            ),
        }


def play_arena_game(
    *,
    game: Game,
    candidate_evaluator: PositionEvaluator,
    best_evaluator: PositionEvaluator,
    candidate_player: int,
    config: ArenaConfig,
    seed: Optional[int] = None,
) -> ArenaGameResult:
    """
    进行一局确定性 Arena 对局。

    Arena 搜索固定使用：
        add_root_noise=False
        temperature=0
    """
    config.validate()

    if candidate_player not in (1, -1):
        raise ValueError(
            "candidate_player 必须是 1 或 -1"
        )

    state = game.initial_state()

    maximum_moves = (
        game.action_size
        if config.max_moves is None
        else min(config.max_moves, game.action_size)
    )

    candidate_mcts = MCTS(
        game=game,
        evaluator=candidate_evaluator,
        config=MCTSConfig(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
        ),
        seed=seed,
    )

    best_mcts = MCTS(
        game=game,
        evaluator=best_evaluator,
        config=MCTSConfig(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
        ),
        seed=None if seed is None else seed + 1,
    )

    for _ in range(maximum_moves):
        if game.terminal_value(state) is not None:
            break

        if state.current_player == candidate_player:
            current_mcts = candidate_mcts
        else:
            current_mcts = best_mcts

        search_result = current_mcts.search(
            state,
            add_root_noise=False,
            temperature=0,
        )

        if search_result.action is None:
            raise RuntimeError(
                "非终局 Arena 状态没有返回动作"
            )

        legal_mask = game.legal_actions(state)

        if legal_mask[search_result.action] <= 0:
            raise RuntimeError(
                "Arena MCTS 返回非法动作："
                f"{search_result.action}"
            )

        state = game.next_state(
            state,
            search_result.action,
        )

    if game.terminal_value(state) is None:
        raise RuntimeError(
            "Arena 对局在非终局状态达到 max_moves"
        )

    return ArenaGameResult(
        winner=game.winner(state),
        candidate_player=candidate_player,
        move_count=state.move_count,
        final_state=state,
    )


def evaluate_agents(
    *,
    game: Game,
    candidate_evaluator: PositionEvaluator,
    best_evaluator: PositionEvaluator,
    config: ArenaConfig,
    seed: Optional[int] = None,
) -> ArenaResult:
    """
    评估两个 PositionEvaluator。

    偶数局 candidate 执黑，奇数局 candidate 执白。
    如果 games 是奇数，candidate 会多执黑一局。
    """
    config.validate()

    candidate_wins = 0
    best_wins = 0
    draws = 0
    game_results: list[ArenaGameResult] = []

    for game_index in range(config.games):
        candidate_player = (
            1 if game_index % 2 == 0 else -1
        )

        game_seed = (
            None
            if seed is None
            else seed + game_index * 2
        )

        result = play_arena_game(
            game=game,
            candidate_evaluator=candidate_evaluator,
            best_evaluator=best_evaluator,
            candidate_player=candidate_player,
            config=config,
            seed=game_seed,
        )

        game_results.append(result)

        if result.candidate_result == 1:
            candidate_wins += 1
        elif result.candidate_result == -1:
            best_wins += 1
        else:
            draws += 1

    return ArenaResult(
        candidate_wins=candidate_wins,
        best_wins=best_wins,
        draws=draws,
        games=config.games,
        promotion_threshold=(
            config.promotion_threshold
        ),
        game_results=tuple(game_results),
    )


def evaluate_models(
    *,
    game: Game,
    candidate_model: nn.Module,
    best_model: nn.Module,
    device: torch.device | str,
    config: ArenaConfig,
    seed: Optional[int] = None,
) -> ArenaResult:
    """使用 PyTorch 模型执行 Arena 评估。"""
    device = torch.device(device)

    candidate_evaluator = TorchNetworkEvaluator(
        game=game,
        model=candidate_model,
        device=device,
    )

    best_evaluator = TorchNetworkEvaluator(
        game=game,
        model=best_model,
        device=device,
    )

    return evaluate_agents(
        game=game,
        candidate_evaluator=candidate_evaluator,
        best_evaluator=best_evaluator,
        config=config,
        seed=seed,
    )
