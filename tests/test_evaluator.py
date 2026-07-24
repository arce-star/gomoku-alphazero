from __future__ import annotations

import numpy as np
import pytest
import torch

from alphazero.games.gomoku import GomokuGame
from alphazero.networks.residual_net import (
    NetworkConfig,
    PolicyValueNet,
)
from alphazero.training.evaluator import (
    ArenaConfig,
    ArenaResult,
    evaluate_agents,
    evaluate_models,
    play_arena_game,
)


class UniformEvaluator:
    """均匀策略、零价值测试评估器。"""

    def __init__(self, action_size: int) -> None:
        self.action_size = action_size
        self.call_count = 0

    def evaluate(self, state):
        self.call_count += 1

        return (
            np.zeros(
                self.action_size,
                dtype=np.float64,
            ),
            0.0,
        )


def make_game() -> GomokuGame:
    return GomokuGame(
        board_size=5,
        connect=4,
    )


def make_model() -> PolicyValueNet:
    return PolicyValueNet(
        NetworkConfig(
            board_size=5,
            input_channels=3,
            channels=8,
            residual_blocks=1,
            value_hidden_channels=4,
            value_hidden_size=8,
        )
    )


def test_invalid_arena_config() -> None:
    with pytest.raises(ValueError):
        ArenaConfig(games=0).validate()

    with pytest.raises(ValueError):
        ArenaConfig(
            num_simulations=0
        ).validate()

    with pytest.raises(ValueError):
        ArenaConfig(c_puct=0).validate()

    with pytest.raises(ValueError):
        ArenaConfig(
            promotion_threshold=1.1
        ).validate()

    with pytest.raises(ValueError):
        ArenaConfig(max_moves=0).validate()


def test_arena_result_score() -> None:
    result = ArenaResult(
        candidate_wins=3,
        best_wins=1,
        draws=2,
        games=6,
        promotion_threshold=0.6,
        game_results=(),
    )

    assert result.candidate_score == pytest.approx(
        4.0 / 6.0
    )

    assert (
        result.candidate_decisive_win_rate
        == pytest.approx(0.75)
    )

    assert result.should_promote


def test_draw_score() -> None:
    result = ArenaResult(
        candidate_wins=0,
        best_wins=0,
        draws=4,
        games=4,
        promotion_threshold=0.55,
        game_results=(),
    )

    assert result.candidate_score == 0.5
    assert result.candidate_decisive_win_rate == 0.0
    assert not result.should_promote


def test_play_single_arena_game() -> None:
    game = make_game()

    candidate = UniformEvaluator(
        game.action_size
    )
    best = UniformEvaluator(
        game.action_size
    )

    result = play_arena_game(
        game=game,
        candidate_evaluator=candidate,
        best_evaluator=best,
        candidate_player=1,
        config=ArenaConfig(
            games=1,
            num_simulations=2,
        ),
        seed=42,
    )

    assert result.winner in (-1, 0, 1)
    assert result.candidate_player == 1
    assert result.candidate_result in (-1, 0, 1)
    assert 1 <= result.move_count <= 25

    assert game.terminal_value(
        result.final_state
    ) is not None

    assert candidate.call_count > 0
    assert best.call_count > 0


def test_evaluate_agents_alternates_colors() -> None:
    game = make_game()

    candidate = UniformEvaluator(
        game.action_size
    )
    best = UniformEvaluator(
        game.action_size
    )

    result = evaluate_agents(
        game=game,
        candidate_evaluator=candidate,
        best_evaluator=best,
        config=ArenaConfig(
            games=4,
            num_simulations=2,
            promotion_threshold=0.55,
        ),
        seed=123,
    )

    assert result.games == 4

    assert (
        result.candidate_wins
        + result.best_wins
        + result.draws
        == 4
    )

    candidate_colors = [
        game_result.candidate_player
        for game_result in result.game_results
    ]

    assert candidate_colors == [1, -1, 1, -1]

    assert 0.0 <= result.candidate_score <= 1.0

    metrics = result.as_dict()

    assert metrics["arena/games"] == 4.0
    assert "arena/candidate_score" in metrics
    assert "arena/should_promote" in metrics


def test_arena_is_reproducible() -> None:
    game1 = make_game()
    game2 = make_game()

    result1 = evaluate_agents(
        game=game1,
        candidate_evaluator=UniformEvaluator(
            game1.action_size
        ),
        best_evaluator=UniformEvaluator(
            game1.action_size
        ),
        config=ArenaConfig(
            games=4,
            num_simulations=2,
        ),
        seed=999,
    )

    result2 = evaluate_agents(
        game=game2,
        candidate_evaluator=UniformEvaluator(
            game2.action_size
        ),
        best_evaluator=UniformEvaluator(
            game2.action_size
        ),
        config=ArenaConfig(
            games=4,
            num_simulations=2,
        ),
        seed=999,
    )

    assert result1.candidate_wins == (
        result2.candidate_wins
    )
    assert result1.best_wins == result2.best_wins
    assert result1.draws == result2.draws

    for game_result1, game_result2 in zip(
        result1.game_results,
        result2.game_results,
    ):
        assert game_result1.winner == (
            game_result2.winner
        )
        assert game_result1.move_count == (
            game_result2.move_count
        )
        assert np.array_equal(
            game_result1.final_state.board,
            game_result2.final_state.board,
        )


def test_invalid_candidate_player() -> None:
    game = make_game()

    evaluator = UniformEvaluator(
        game.action_size
    )

    with pytest.raises(
        ValueError,
        match="candidate_player",
    ):
        play_arena_game(
            game=game,
            candidate_evaluator=evaluator,
            best_evaluator=evaluator,
            candidate_player=0,
            config=ArenaConfig(
                games=1,
                num_simulations=2,
            ),
        )


def test_evaluate_cpu_models() -> None:
    game = make_game()

    result = evaluate_models(
        game=game,
        candidate_model=make_model(),
        best_model=make_model(),
        device="cpu",
        config=ArenaConfig(
            games=2,
            num_simulations=2,
        ),
        seed=7,
    )

    assert result.games == 2

    assert (
        result.candidate_wins
        + result.best_wins
        + result.draws
        == 2
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_evaluate_cuda_models() -> None:
    game = make_game()

    candidate = make_model()
    best = make_model()

    result = evaluate_models(
        game=game,
        candidate_model=candidate,
        best_model=best,
        device="cuda",
        config=ArenaConfig(
            games=2,
            num_simulations=2,
        ),
        seed=8,
    )

    assert result.games == 2

    assert next(
        candidate.parameters()
    ).device.type == "cuda"

    assert next(
        best.parameters()
    ).device.type == "cuda"
