from __future__ import annotations

import numpy as np
import pytest
import time

from alphazero.baselines.heuristic import (
    HeuristicConfig,
    HeuristicPlayer,
)
from alphazero.games.gomoku import GomokuGame


def make_game(
    board_size: int = 9,
    connect: int = 5,
) -> GomokuGame:
    return GomokuGame(board_size=board_size, connect=connect)


# ----------------------------------------------------------------
# helper: apply a sequence of actions and return the final state
# ----------------------------------------------------------------
def play_actions(game: GomokuGame, *actions: int):
    """Return the state after applying *actions* (using next_state)."""
    state = game.initial_state()
    for action in actions:
        state = game.next_state(state, int(action))
    return state


# ================================================================
# Tests
# ================================================================

def test_empty_board_plays_center() -> None:
    game = make_game()
    state = game.initial_state()
    player = HeuristicPlayer()
    action = player.select_action(game, state)
    # 9×9 centre = (4, 4) → action 40
    assert action == 40


def test_takes_immediate_horizontal_win() -> None:
    game = make_game()
    # Black moves: 36, 37, 38, 39 (row 4, cols 0-3)
    # White moves: 0, 1, 2, 3   (row 0, cols 0-3)
    # Black to play at (4, 4) = action 40 wins
    state = play_actions(
        game,
        36, 0, 37, 1, 38, 2, 39, 3,
    )
    # Black to move
    assert state.current_player == 1

    player = HeuristicPlayer()
    action = player.select_action(game, state)
    assert action == 40


def test_blocks_opponents_immediate_horizontal_win() -> None:
    game = make_game()
    # White has 36, 37, 38, 39 (row 4, cols 0-3), one more for win at 40
    # Black to move must block at 40
    state = play_actions(
        game,
        0, 36, 1, 37, 2, 38, 10, 39,
    )
    # Black to move
    assert state.current_player == 1

    player = HeuristicPlayer()
    action = player.select_action(game, state)
    assert action == 40


def test_returns_legal_action() -> None:
    game = make_game()
    state = play_actions(game, 40, 39, 31, 49)

    player = HeuristicPlayer()
    action = player.select_action(game, state)

    legal_mask = game.legal_actions(state)
    assert legal_mask[action] > 0.0


def test_rejects_terminal_position() -> None:
    game = make_game()
    # 9 moves: Black gets 5 in a row at 36,37,38,39,40
    state = play_actions(game, 36, 0, 37, 1, 38, 2, 39, 3, 40)

    assert game.terminal_value(state) is not None

    player = HeuristicPlayer()
    with pytest.raises(ValueError, match="terminal"):
        player.select_action(game, state)


def test_original_state_not_mutated() -> None:
    game = make_game()
    state = play_actions(game, 40, 39, 31, 49)

    original_board = state.board.copy()

    player = HeuristicPlayer()
    _ = player.select_action(game, state)

    # state.board must be unchanged
    assert np.array_equal(state.board, original_board)


def test_5x5_connect_4_immediate_win() -> None:
    game = make_game(board_size=5, connect=4)
    # Black: 0, 1, 2  → row 0 cols 0,1,2
    # White: 5, 6, 7 → row 1 cols 0,1,2
    # After 6 moves it is Black's turn.
    # Black has 3 in a row and can win immediately at action 3.
    state = play_actions(game, 0, 5, 1, 6, 2, 7)

    assert state.current_player == 1
    player = HeuristicPlayer()
    action = player.select_action(game, state)
    assert action == 3


def test_can_play_complete_game_against_itself() -> None:
    game = make_game(board_size=5, connect=4)
    player = HeuristicPlayer(
        HeuristicConfig(
            candidate_radius=2,
            candidate_limit=10,
            search_depth=1,
        )
    )

    state = game.initial_state()

    while game.terminal_value(state) is None:
        action = player.select_action(game, state)

        # action must be legal
        legal_mask = game.legal_actions(state)
        assert legal_mask[action] > 0.0

        state = game.next_state(state, int(action))

    assert game.winner(state) in (-1, 0, 1)
    assert np.count_nonzero(state.board) > 0
    # Should complete within a reasonable time on a 5×5 board
    assert state.move_count <= 25


def test_self_play_completes_quickly() -> None:
    """Self-play on a small board must be fast enough for CI."""
    game = make_game(board_size=5, connect=4)
    player = HeuristicPlayer(
        HeuristicConfig(
            candidate_radius=1,
            candidate_limit=6,
            search_depth=1,
        )
    )

    state = game.initial_state()
    started = time.perf_counter()

    while game.terminal_value(state) is None:
        action = player.select_action(game, state)
        state = game.next_state(state, int(action))

    elapsed = time.perf_counter() - started
    # A 5×5 game with depth-1 search should finish in well under 5 seconds.
    assert elapsed < 5.0


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        HeuristicConfig(candidate_radius=0).validate()
    with pytest.raises(ValueError):
        HeuristicConfig(candidate_limit=0).validate()
    with pytest.raises(ValueError):
        HeuristicConfig(search_depth=0).validate()
