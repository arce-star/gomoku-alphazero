from __future__ import annotations

import numpy as np
import pytest

from alphazero.games.gomoku import GomokuGame
from alphazero.mcts.search import (
    MCTS,
    MCTSConfig,
)
from alphazero.selfplay.episode import (
    SelfPlayConfig,
    TrainingExample,
    play_episode,
)


class UniformEvaluator:
    """
    测试用均匀策略、零价值评估器。
    """

    def __init__(
        self,
        action_size: int,
    ) -> None:
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


def make_test_components(
    seed: int = 42,
):
    game = GomokuGame(
        board_size=5,
        connect=4,
    )

    evaluator = UniformEvaluator(
        game.action_size
    )

    mcts = MCTS(
        game=game,
        evaluator=evaluator,
        config=MCTSConfig(
            num_simulations=4,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        ),
        seed=seed,
    )

    return game, evaluator, mcts


def test_training_example_converts_dtype() -> None:
    example = TrainingExample(
        state=np.zeros(
            (3, 5, 5),
            dtype=np.float64,
        ),
        policy=np.full(
            25,
            1.0 / 25.0,
            dtype=np.float64,
        ),
        value=1,
    )

    assert example.state.dtype == np.float32
    assert example.policy.dtype == np.float32
    assert example.value == 1.0

    assert example.state.flags.c_contiguous
    assert example.policy.flags.c_contiguous

    assert example.state.flags.writeable is False
    assert example.policy.flags.writeable is False


def test_complete_episode_without_augmentation() -> None:
    game, evaluator, mcts = (
        make_test_components(seed=1)
    )

    result = play_episode(
        game=game,
        mcts=mcts,
        config=SelfPlayConfig(
            temperature_moves=4,
            sampling_temperature=1.0,
            add_root_noise=True,
            augment_symmetries=False,
        ),
    )

    assert result.winner in (-1, 0, 1)

    assert 1 <= result.move_count <= game.action_size

    assert game.terminal_value(
        result.final_state
    ) is not None

    # 不使用增强时，每一步生成一条样本。
    assert len(result.examples) == result.move_count

    assert evaluator.call_count > 0

    for example in result.examples:
        assert example.state.shape == (
            3,
            5,
            5,
        )

        assert example.policy.shape == (
            25,
        )

        assert example.policy.sum() == pytest.approx(
            1.0,
            abs=1e-5,
        )

        assert np.all(example.policy >= 0)

        assert example.value in (
            -1.0,
            0.0,
            1.0,
        )


def test_episode_with_symmetry_augmentation() -> None:
    game, _, mcts = make_test_components(
        seed=2
    )

    result = play_episode(
        game=game,
        mcts=mcts,
        config=SelfPlayConfig(
            temperature_moves=4,
            sampling_temperature=1.0,
            add_root_noise=True,
            augment_symmetries=True,
        ),
    )

    # 每个原始局面扩充为 8 条。
    assert len(result.examples) == (
        result.move_count * 8
    )

    for example in result.examples:
        assert example.state.shape == (
            3,
            5,
            5,
        )

        assert example.policy.shape == (
            25,
        )

        assert example.policy.sum() == pytest.approx(
            1.0,
            abs=1e-5,
        )


def test_value_targets_follow_player_view() -> None:
    game, _, mcts = make_test_components(
        seed=3
    )

    result = play_episode(
        game=game,
        mcts=mcts,
        config=SelfPlayConfig(
            temperature_moves=5,
            add_root_noise=True,
            augment_symmetries=False,
        ),
    )

    if result.winner == 0:
        assert all(
            example.value == 0.0
            for example in result.examples
        )
        return

    # 初始状态始终是黑棋视角。
    expected_first_value = (
        1.0
        if result.winner == 1
        else -1.0
    )

    assert result.examples[0].value == (
        expected_first_value
    )

    # 当前玩家逐步交替，因此非和棋时 value 也应交替。
    for index, example in enumerate(
        result.examples
    ):
        player = 1 if index % 2 == 0 else -1

        expected_value = (
            1.0
            if player == result.winner
            else -1.0
        )

        assert example.value == expected_value


def test_selfplay_is_reproducible_without_noise() -> None:
    game1, _, mcts1 = make_test_components(
        seed=100
    )

    game2, _, mcts2 = make_test_components(
        seed=100
    )

    config = SelfPlayConfig(
        temperature_moves=0,
        add_root_noise=False,
        augment_symmetries=False,
    )

    result1 = play_episode(
        game=game1,
        mcts=mcts1,
        config=config,
    )

    result2 = play_episode(
        game=game2,
        mcts=mcts2,
        config=config,
    )

    assert result1.winner == result2.winner
    assert result1.move_count == result2.move_count

    assert np.array_equal(
        result1.final_state.board,
        result2.final_state.board,
    )

    assert len(result1.examples) == len(
        result2.examples
    )

    for example1, example2 in zip(
        result1.examples,
        result2.examples,
    ):
        assert np.array_equal(
            example1.state,
            example2.state,
        )

        assert np.array_equal(
            example1.policy,
            example2.policy,
        )

        assert example1.value == example2.value


def test_max_moves_guard() -> None:
    game, _, mcts = make_test_components(
        seed=5
    )

    with pytest.raises(
        RuntimeError,
        match="max_moves",
    ):
        play_episode(
            game=game,
            mcts=mcts,
            config=SelfPlayConfig(
                temperature_moves=1,
                add_root_noise=False,
                augment_symmetries=False,
                max_moves=1,
            ),
        )


def test_mcts_and_game_must_match() -> None:
    game1, _, mcts = make_test_components(
        seed=6
    )

    game2 = GomokuGame(
        board_size=5,
        connect=4,
    )

    assert game1 is not game2

    with pytest.raises(
        ValueError,
        match="同一个对象",
    ):
        play_episode(
            game=game2,
            mcts=mcts,
            config=SelfPlayConfig(
                augment_symmetries=False,
            ),
        )
