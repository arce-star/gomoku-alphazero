from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from alphazero.games.base import Game, GameState


class GomokuGame(Game):
    """
    无禁手五子棋。

    默认规则：
    - 9×9 棋盘
    - 黑棋先行
    - 连续 5 子获胜
    - 无禁手
    - 落子后轮到另一方行动
    """

    def __init__(self, board_size: int = 9, connect: int = 5) -> None:
        if board_size <= 0:
            raise ValueError("board_size 必须大于 0")

        if connect <= 1:
            raise ValueError("connect 必须大于 1")

        if connect > board_size:
            raise ValueError("connect 不能大于 board_size")

        self._board_size = int(board_size)
        self.connect = int(connect)

    @property
    def board_size(self) -> int:
        return self._board_size

    @property
    def action_size(self) -> int:
        return self.board_size * self.board_size

    @property
    def input_channels(self) -> int:
        # 通道 0：当前玩家的棋子
        # 通道 1：对手的棋子
        # 通道 2：当前玩家是否为黑棋
        return 3

    def initial_state(self) -> GameState:
        board = np.zeros(
            (self.board_size, self.board_size),
            dtype=np.int8,
        )

        return GameState(
            board=board,
            current_player=1,
            move_count=0,
            last_action=None,
        )

    def action_to_coord(self, action: int) -> Tuple[int, int]:
        """
        将动作编号转换成棋盘坐标。

        action = row * board_size + col
        """
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action 必须是整数")

        action = int(action)

        if action < 0 or action >= self.action_size:
            raise ValueError(
                f"action 必须位于 [0, {self.action_size - 1}]，"
                f"实际得到 {action}"
            )

        row, col = divmod(action, self.board_size)
        return row, col

    def coord_to_action(self, row: int, col: int) -> int:
        """将棋盘坐标转换成动作编号。"""
        if not isinstance(row, (int, np.integer)):
            raise TypeError("row 必须是整数")

        if not isinstance(col, (int, np.integer)):
            raise TypeError("col 必须是整数")

        row = int(row)
        col = int(col)

        if not (0 <= row < self.board_size):
            raise ValueError(
                f"row 必须位于 [0, {self.board_size - 1}]"
            )

        if not (0 <= col < self.board_size):
            raise ValueError(
                f"col 必须位于 [0, {self.board_size - 1}]"
            )

        return row * self.board_size + col

    def legal_actions(self, state: GameState) -> np.ndarray:
        """
        返回长度为 action_size 的合法动作掩码。
        """
        self._validate_state(state)

        # 终局之后不允许继续落子。
        if self.terminal_value(state) is not None:
            return np.zeros(self.action_size, dtype=np.float32)

        return (state.board.reshape(-1) == 0).astype(np.float32)

    def next_state(self, state: GameState, action: int) -> GameState:
        """
        执行一步落子。

        本方法不会修改原状态，而是返回新的 GameState。
        """
        self._validate_state(state)

        if self.terminal_value(state) is not None:
            raise ValueError("棋局已经结束，不能继续落子")

        row, col = self.action_to_coord(action)

        if state.board[row, col] != 0:
            raise ValueError(
                f"位置 ({row}, {col}) 已经有棋子，不能重复落子"
            )

        new_board = state.board.copy()
        new_board[row, col] = state.current_player

        return GameState(
            board=new_board,
            current_player=-state.current_player,
            move_count=state.move_count + 1,
            last_action=int(action),
        )

    def winner(self, state: GameState) -> int:
        """
        扫描整个棋盘并返回获胜者。
        """
        self._validate_state(state)

        board = state.board
        size = self.board_size

        # 四个检查方向：
        # 水平、垂直、主对角线、副对角线
        directions = (
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1),
        )

        for row in range(size):
            for col in range(size):
                player = int(board[row, col])

                if player == 0:
                    continue

                for delta_row, delta_col in directions:
                    end_row = row + (self.connect - 1) * delta_row
                    end_col = col + (self.connect - 1) * delta_col

                    # 如果终点超出棋盘，则这个方向不可能连成五子。
                    if not (
                        0 <= end_row < size
                        and 0 <= end_col < size
                    ):
                        continue

                    connected = True

                    for step in range(1, self.connect):
                        check_row = row + step * delta_row
                        check_col = col + step * delta_col

                        if board[check_row, check_col] != player:
                            connected = False
                            break

                    if connected:
                        return player

        return 0

    def terminal_value(self, state: GameState) -> Optional[float]:
        """
        返回当前玩家视角下的终局价值。

        注意：
        next_state 落子后会立即切换 current_player。
        因此正常情况下，如果上一个玩家刚刚获胜，
        当前待行动玩家得到的价值就是 -1.0。
        """
        self._validate_state(state)

        winning_player = self.winner(state)

        if winning_player != 0:
            if winning_player == state.current_player:
                return 1.0
            return -1.0

        if state.move_count == self.action_size:
            return 0.0

        return None

    def encode_state(self, state: GameState) -> np.ndarray:
        """
        将状态编码成 3 个通道。

        channel 0:
            当前玩家的棋子位置。

        channel 1:
            对手的棋子位置。

        channel 2:
            当前玩家颜色。
            当前玩家为黑棋时全为 1，白棋时全为 0。
        """
        self._validate_state(state)

        current_stones = (
            state.board == state.current_player
        ).astype(np.float32)

        opponent_stones = (
            state.board == -state.current_player
        ).astype(np.float32)

        current_color = np.full(
            (self.board_size, self.board_size),
            1.0 if state.current_player == 1 else 0.0,
            dtype=np.float32,
        )

        encoded = np.stack(
            [
                current_stones,
                opponent_stones,
                current_color,
            ],
            axis=0,
        )

        return encoded.astype(np.float32, copy=False)

    def render(self, state: GameState) -> str:
        """
        返回便于终端显示的棋盘字符串。

        X：黑棋
        O：白棋
        .：空位
        """
        self._validate_state(state)

        symbols = {
            0: ".",
            1: "X",
            -1: "O",
        }

        header = "   " + " ".join(
            str(col) for col in range(self.board_size)
        )

        lines = [header]

        for row in range(self.board_size):
            cells = " ".join(
                symbols[int(value)]
                for value in state.board[row]
            )
            lines.append(f"{row:2d} {cells}")

        return "\n".join(lines)

    def _validate_state(self, state: GameState) -> None:
        """检查状态结构，尽早发现规则层错误。"""
        if not isinstance(state, GameState):
            raise TypeError("state 必须是 GameState")

        expected_shape = (self.board_size, self.board_size)

        if state.board.shape != expected_shape:
            raise ValueError(
                f"棋盘形状必须是 {expected_shape}，"
                f"实际得到 {state.board.shape}"
            )

        if state.current_player not in (1, -1):
            raise ValueError("current_player 必须是 1 或 -1")

        if np.any(
            (state.board != 0)
            & (state.board != 1)
            & (state.board != -1)
        ):
            raise ValueError("棋盘只能包含 -1、0、1")

        actual_move_count = int(np.count_nonzero(state.board))

        if state.move_count != actual_move_count:
            raise ValueError(
                f"move_count={state.move_count}，"
                f"但棋盘实际有 {actual_move_count} 个棋子"
            )

        if not 0 <= state.move_count <= self.action_size:
            raise ValueError("move_count 超出有效范围")

        if state.last_action is not None:
            self.action_to_coord(state.last_action)
