import numpy as np
import pytest

from alphazero.games.base import GameState
from alphazero.games.gomoku import GomokuGame


@pytest.fixture
def game() -> GomokuGame:
    return GomokuGame(board_size=9, connect=5)


def make_state(
    board: np.ndarray,
    current_player: int = 1,
    last_action=None,
) -> GameState:
    return GameState(
        board=board,
        current_player=current_player,
        move_count=int(np.count_nonzero(board)),
        last_action=last_action,
    )


def test_initial_state(game: GomokuGame) -> None:
    state = game.initial_state()

    assert state.board.shape == (9, 9)
    assert state.board.dtype == np.int8
    assert np.all(state.board == 0)

    assert state.current_player == 1
    assert state.move_count == 0
    assert state.last_action is None

    assert game.board_size == 9
    assert game.action_size == 81
    assert game.input_channels == 3

    assert game.winner(state) == 0
    assert game.terminal_value(state) is None


def test_action_coordinate_conversion(game: GomokuGame) -> None:
    assert game.coord_to_action(0, 0) == 0
    assert game.coord_to_action(0, 8) == 8
    assert game.coord_to_action(1, 0) == 9
    assert game.coord_to_action(8, 8) == 80

    assert game.action_to_coord(0) == (0, 0)
    assert game.action_to_coord(8) == (0, 8)
    assert game.action_to_coord(9) == (1, 0)
    assert game.action_to_coord(80) == (8, 8)


def test_initial_legal_actions(game: GomokuGame) -> None:
    state = game.initial_state()
    legal = game.legal_actions(state)

    assert legal.shape == (81,)
    assert legal.dtype == np.float32
    assert np.sum(legal) == 81
    assert np.all(legal == 1.0)


def test_next_state_and_player_switch(game: GomokuGame) -> None:
    initial = game.initial_state()
    action = game.coord_to_action(4, 4)

    next_state = game.next_state(initial, action)

    # 原状态不能被修改。
    assert initial.board[4, 4] == 0
    assert initial.move_count == 0

    # 新状态正确落下黑棋。
    assert next_state.board[4, 4] == 1
    assert next_state.current_player == -1
    assert next_state.move_count == 1
    assert next_state.last_action == action

    legal = game.legal_actions(next_state)

    assert legal[action] == 0.0
    assert np.sum(legal) == 80


def test_white_move(game: GomokuGame) -> None:
    state = game.initial_state()

    state = game.next_state(
        state,
        game.coord_to_action(4, 4),
    )

    state = game.next_state(
        state,
        game.coord_to_action(3, 3),
    )

    assert state.board[4, 4] == 1
    assert state.board[3, 3] == -1
    assert state.current_player == 1
    assert state.move_count == 2


def test_cannot_play_on_occupied_position(
    game: GomokuGame,
) -> None:
    state = game.initial_state()
    action = game.coord_to_action(4, 4)

    state = game.next_state(state, action)

    with pytest.raises(ValueError, match="已经有棋子"):
        game.next_state(state, action)


def test_invalid_action(game: GomokuGame) -> None:
    state = game.initial_state()

    with pytest.raises(ValueError):
        game.next_state(state, -1)

    with pytest.raises(ValueError):
        game.next_state(state, 81)

    with pytest.raises(TypeError):
        game.next_state(state, "0")


@pytest.mark.parametrize(
    "coordinates",
    [
        # 水平五连
        [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)],

        # 垂直五连
        [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6)],

        # 左上到右下
        [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],

        # 右上到左下
        [(1, 7), (2, 6), (3, 5), (4, 4), (5, 3)],
    ],
)
def test_black_wins_in_all_directions(
    game: GomokuGame,
    coordinates,
) -> None:
    board = np.zeros((9, 9), dtype=np.int8)

    for row, col in coordinates:
        board[row, col] = 1

    state = make_state(
        board=board,
        current_player=-1,
    )

    assert game.winner(state) == 1
    assert game.terminal_value(state) == -1.0


def test_white_wins(game: GomokuGame) -> None:
    board = np.zeros((9, 9), dtype=np.int8)

    for col in range(2, 7):
        board[5, col] = -1

    state = make_state(
        board=board,
        current_player=1,
    )

    assert game.winner(state) == -1
    assert game.terminal_value(state) == -1.0


def test_four_stones_is_not_a_win(game: GomokuGame) -> None:
    board = np.zeros((9, 9), dtype=np.int8)

    for col in range(4):
        board[0, col] = 1

    state = make_state(board)

    assert game.winner(state) == 0
    assert game.terminal_value(state) is None


def test_complete_game_and_terminal_value(
    game: GomokuGame,
) -> None:
    state = game.initial_state()

    # 黑棋在第 0 行下出五连。
    # 白棋在第 8 行放置四颗棋子，不形成五连。
    moves = [
        (0, 0), (8, 0),
        (0, 1), (8, 1),
        (0, 2), (8, 2),
        (0, 3), (8, 3),
        (0, 4),
    ]

    for row, col in moves:
        state = game.next_state(
            state,
            game.coord_to_action(row, col),
        )

    assert game.winner(state) == 1

    # 黑棋落子后已经切换到白棋，因此白棋视角价值是 -1。
    assert state.current_player == -1
    assert game.terminal_value(state) == -1.0

    # 终局没有合法动作。
    assert np.sum(game.legal_actions(state)) == 0

    with pytest.raises(ValueError, match="棋局已经结束"):
        game.next_state(
            state,
            game.coord_to_action(4, 4),
        )


def test_draw_board(game: GomokuGame) -> None:
    """
    构造一个填满且没有五连的棋盘。

    每一行每两格切换颜色，相邻行反色：
    XXOOXXOOX
    OOXXOOXXO
    ...
    """
    board = np.zeros((9, 9), dtype=np.int8)

    for row in range(9):
        for col in range(9):
            board[row, col] = (
                1 if (row + col // 2) % 2 == 0 else -1
            )

    state = make_state(
        board=board,
        current_player=-1,
    )

    assert state.move_count == 81
    assert game.winner(state) == 0
    assert game.terminal_value(state) == 0.0
    assert np.sum(game.legal_actions(state)) == 0


def test_encode_initial_state(game: GomokuGame) -> None:
    state = game.initial_state()
    encoded = game.encode_state(state)

    assert encoded.shape == (3, 9, 9)
    assert encoded.dtype == np.float32

    # 初始棋盘没有棋子。
    assert np.all(encoded[0] == 0)
    assert np.all(encoded[1] == 0)

    # 初始当前玩家是黑棋。
    assert np.all(encoded[2] == 1)


def test_encode_state_from_current_player_view(
    game: GomokuGame,
) -> None:
    state = game.initial_state()

    # 黑棋落在 (4, 4)，之后当前玩家变成白棋。
    state = game.next_state(
        state,
        game.coord_to_action(4, 4),
    )

    encoded = game.encode_state(state)

    assert state.current_player == -1

    # 当前玩家白棋还没有棋子。
    assert encoded[0, 4, 4] == 0.0

    # (4, 4) 是对手黑棋。
    assert encoded[1, 4, 4] == 1.0

    # 当前玩家是白棋，所以颜色通道全为 0。
    assert np.all(encoded[2] == 0.0)


def test_state_board_is_read_only(game: GomokuGame) -> None:
    state = game.initial_state()

    with pytest.raises(ValueError):
        state.board[0, 0] = 1


def test_render(game: GomokuGame) -> None:
    state = game.initial_state()

    state = game.next_state(
        state,
        game.coord_to_action(0, 0),
    )

    state = game.next_state(
        state,
        game.coord_to_action(1, 1),
    )

    output = game.render(state)

    assert "X" in output
    assert "O" in output
    assert "." in output
