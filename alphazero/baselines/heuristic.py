from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from alphazero.games.base import GameState
from alphazero.games.gomoku import GomokuGame


DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
)

PATTERN_SCORES: dict[str, int] = {
    "five": 1_000_000,
    "open_four": 100_000,
    "closed_four": 12_000,
    "open_three": 8_000,
    "closed_three": 1_000,
    "open_two": 300,
    "closed_two": 80,
    "single": 10,
}


@dataclass(frozen=True)
class HeuristicConfig:
    """Configuration for the heuristic Gomoku baseline."""

    candidate_radius: int = 2
    candidate_limit: int = 12
    search_depth: int = 2
    center_bonus: int = 8

    def validate(self) -> None:
        if self.candidate_radius < 1:
            raise ValueError("candidate_radius must be >= 1")
        if self.candidate_limit < 1:
            raise ValueError("candidate_limit must be >= 1")
        if self.search_depth < 1:
            raise ValueError("search_depth must be >= 1")


class HeuristicPlayer:
    """A deterministic, non-neural Gomoku baseline.

    The public API expects a GomokuGame and a GameState.  The returned action
    follows the project convention: ``action = row * board_size + col``.

    The evaluator uses only board occupancy and pattern-based tactical rules.
    It never calls a neural network or MCTS.
    """

    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.config = config or HeuristicConfig()
        self.config.validate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(
        self,
        game: GomokuGame,
        state: GameState,
    ) -> int:
        """Return the best action for *state.current_player*.

        Raises ``ValueError`` if the position is terminal.
        Does **not** modify *state* or *state.board*.
        """
        if game.terminal_value(state) is not None:
            raise ValueError(
                "cannot select an action from a terminal game"
            )

        board_size = game.board_size
        connect = game.connect
        legal_mask = game.legal_actions(state)
        legal_actions = np.flatnonzero(legal_mask > 0.0).astype(np.int64)

        if legal_actions.size == 0:
            raise ValueError("no legal actions available")

        player = state.current_player

        # Work on a mutable 2-D copy so the original GameState is untouched.
        board = state.board.copy()
        # Convert to a writable int8 array (copy() returns writable).
        board = np.asarray(board, dtype=np.int8)

        # --- empty board: centre ---
        if not np.any(board):
            centre = board_size // 2
            return int(centre * board_size + centre)

        # --- immediate win ---
        winning_action = _find_immediate_win(
            board=board,
            player=player,
            board_size=board_size,
            connect=connect,
            actions=legal_actions,
        )
        if winning_action is not None:
            return int(winning_action)

        # --- block opponent's immediate win ---
        blocking_action = _find_immediate_win(
            board=board,
            player=-player,
            board_size=board_size,
            connect=connect,
            actions=legal_actions,
        )
        if blocking_action is not None:
            return int(blocking_action)

        # --- candidate generation + ordered search ---
        candidates = _candidate_actions(
            board=board,
            legal_actions=legal_actions,
            board_size=board_size,
            radius=self.config.candidate_radius,
        )

        ordered = _ordered_actions(
            board=board,
            player=player,
            board_size=board_size,
            connect=connect,
            actions=candidates,
            limit=self.config.candidate_limit,
            center_bonus=self.config.center_bonus,
        )

        depth = self.config.search_depth

        best_action = int(ordered[0])
        best_score = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        for action in ordered:
            action = int(action)
            row, col = divmod(action, board_size)
            board[row, col] = player

            score = -_negamax(
                board=board,
                player=-player,
                board_size=board_size,
                connect=connect,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
                candidate_radius=self.config.candidate_radius,
                candidate_limit=self.config.candidate_limit,
                center_bonus=self.config.center_bonus,
            )

            board[row, col] = 0

            if score > best_score:
                best_score = score
                best_action = action

            alpha = max(alpha, best_score)

        return int(best_action)


# ======================================================================
# Internal helpers (module-level functions, 2-D mutable board throughout)
# ======================================================================


def _find_immediate_win(
    *,
    board: np.ndarray,
    player: int,
    board_size: int,
    connect: int,
    actions: np.ndarray,
) -> int | None:
    """Return an action that immediately gives *player* a connect-in-a-row.

    ``board`` must be a **writable** 2-D int8 array.
    """
    for action in actions:
        action = int(action)
        row, col = divmod(action, board_size)
        if board[row, col] != 0:
            continue

        board[row, col] = player
        won = _is_winning_move(
            board=board,
            row=row,
            col=col,
            player=player,
            board_size=board_size,
            connect=connect,
        )
        board[row, col] = 0

        if won:
            return action

    return None


def _is_winning_move(
    *,
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    board_size: int,
    connect: int,
) -> bool:
    for dr, dc in DIRECTIONS:
        count = 1
        for sign in (-1, 1):
            r = row + sign * dr
            c = col + sign * dc
            while (
                0 <= r < board_size
                and 0 <= c < board_size
                and board[r, c] == player
            ):
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= connect:
            return True
    return False


def _candidate_actions(
    *,
    board: np.ndarray,
    legal_actions: np.ndarray,
    board_size: int,
    radius: int,
) -> np.ndarray:
    """Return legal actions near occupied cells (Chebyshev distance)."""
    occupied = np.argwhere(board != 0)

    if occupied.size == 0:
        centre = board_size // 2
        return np.array([centre * board_size + centre], dtype=np.int64)

    legal_set: set[int] = set(int(a) for a in legal_actions)
    candidates: set[int] = set()

    for row, col in occupied:
        row, col = int(row), int(col)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    action = nr * board_size + nc
                    if action in legal_set:
                        candidates.add(action)

    if candidates:
        return np.array(sorted(candidates), dtype=np.int64)

    return np.asarray(legal_actions, dtype=np.int64)


def _ordered_actions(
    *,
    board: np.ndarray,
    player: int,
    board_size: int,
    connect: int,
    actions: np.ndarray,
    limit: int,
    center_bonus: int,
) -> list[int]:
    scored: list[tuple[int, int]] = []

    for action in actions:
        action = int(action)
        off_score = _move_score(
            board=board,
            action=action,
            player=player,
            board_size=board_size,
            connect=connect,
            center_bonus=center_bonus,
        )
        def_score = _move_score(
            board=board,
            action=action,
            player=-player,
            board_size=board_size,
            connect=connect,
            center_bonus=center_bonus,
        )
        total = off_score + int(def_score * 0.9)
        scored.append((total, action))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [action for _, action in scored[:limit]]


def _move_score(
    *,
    board: np.ndarray,
    action: int,
    player: int,
    board_size: int,
    connect: int,
    center_bonus: int,
) -> int:
    row, col = divmod(action, board_size)

    if board[row, col] != 0:
        return -PATTERN_SCORES["five"]

    board[row, col] = player

    if _is_winning_move(
        board=board,
        row=row,
        col=col,
        player=player,
        board_size=board_size,
        connect=connect,
    ):
        board[row, col] = 0
        return PATTERN_SCORES["five"]

    score = _point_pattern_score(
        board=board,
        row=row,
        col=col,
        player=player,
        board_size=board_size,
        connect=connect,
    )

    centre = (board_size - 1) / 2.0
    distance = abs(row - centre) + abs(col - centre)
    score += int(max(0.0, board_size - distance) * center_bonus)

    board[row, col] = 0
    return score


def _point_pattern_score(
    *,
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    board_size: int,
    connect: int,
) -> int:
    total = 0
    for dr, dc in DIRECTIONS:
        length, open_ends = _line_info(
            board=board,
            row=row,
            col=col,
            player=player,
            dr=dr,
            dc=dc,
            board_size=board_size,
        )
        total += _line_score(length=length, open_ends=open_ends, connect=connect)
    return total


def _line_info(
    *,
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    dr: int,
    dc: int,
    board_size: int,
) -> tuple[int, int]:
    length = 1
    open_ends = 0

    for sign in (-1, 1):
        r = row + sign * dr
        c = col + sign * dc
        while (
            0 <= r < board_size
            and 0 <= c < board_size
            and board[r, c] == player
        ):
            length += 1
            r += sign * dr
            c += sign * dc

        if 0 <= r < board_size and 0 <= c < board_size and board[r, c] == 0:
            open_ends += 1

    return length, open_ends


def _line_score(
    *,
    length: int,
    open_ends: int,
    connect: int,
) -> int:
    if length >= connect:
        return PATTERN_SCORES["five"]

    if length == connect - 1:
        if open_ends == 2:
            return PATTERN_SCORES["open_four"]
        if open_ends == 1:
            return PATTERN_SCORES["closed_four"]
        return 0

    if length == connect - 2:
        if open_ends == 2:
            return PATTERN_SCORES["open_three"]
        if open_ends == 1:
            return PATTERN_SCORES["closed_three"]
        return 0

    if length == connect - 3:
        if open_ends == 2:
            return PATTERN_SCORES["open_two"]
        if open_ends == 1:
            return PATTERN_SCORES["closed_two"]
        return 0

    if length == 1:
        return PATTERN_SCORES["single"]

    return 0


def _negamax(
    *,
    board: np.ndarray,
    player: int,
    board_size: int,
    connect: int,
    depth: int,
    alpha: float,
    beta: float,
    candidate_radius: int,
    candidate_limit: int,
    center_bonus: int,
) -> float:
    # Collect legal actions from the mutable board.
    legal_actions = np.flatnonzero(board == 0).astype(np.int64)

    if legal_actions.size == 0:
        return 0.0

    # Immediate win / block checks on the mutable board.
    own_win = _find_immediate_win(
        board=board,
        player=player,
        board_size=board_size,
        connect=connect,
        actions=legal_actions,
    )
    if own_win is not None:
        return float(PATTERN_SCORES["five"])

    opp_win = _find_immediate_win(
        board=board,
        player=-player,
        board_size=board_size,
        connect=connect,
        actions=legal_actions,
    )
    if opp_win is not None:
        return -float(PATTERN_SCORES["five"])

    if depth <= 0:
        return float(
            _evaluate_board(
                board=board,
                player=player,
                board_size=board_size,
                connect=connect,
            )
        )

    candidates = _candidate_actions(
        board=board,
        legal_actions=legal_actions,
        board_size=board_size,
        radius=candidate_radius,
    )
    ordered = _ordered_actions(
        board=board,
        player=player,
        board_size=board_size,
        connect=connect,
        actions=candidates,
        limit=candidate_limit,
        center_bonus=center_bonus,
    )

    best_score = -float("inf")

    for action in ordered:
        action = int(action)
        row, col = divmod(action, board_size)
        board[row, col] = player

        score = -_negamax(
            board=board,
            player=-player,
            board_size=board_size,
            connect=connect,
            depth=depth - 1,
            alpha=-beta,
            beta=-alpha,
            candidate_radius=candidate_radius,
            candidate_limit=candidate_limit,
            center_bonus=center_bonus,
        )

        board[row, col] = 0

        best_score = max(best_score, score)
        alpha = max(alpha, score)

        if alpha >= beta:
            break

    return best_score


def _evaluate_board(
    *,
    board: np.ndarray,
    player: int,
    board_size: int,
    connect: int,
) -> int:
    own = _score_player(
        board=board,
        player=player,
        board_size=board_size,
        connect=connect,
    )
    opp = _score_player(
        board=board,
        player=-player,
        board_size=board_size,
        connect=connect,
    )
    return own - opp


def _score_player(
    *,
    board: np.ndarray,
    player: int,
    board_size: int,
    connect: int,
) -> int:
    score = 0
    for row in range(board_size):
        for col in range(board_size):
            if board[row, col] != player:
                continue
            for dr, dc in DIRECTIONS:
                # Only count each line segment from its first stone.
                pr, pc = row - dr, col - dc
                if (
                    0 <= pr < board_size
                    and 0 <= pc < board_size
                    and board[pr, pc] == player
                ):
                    continue

                length, open_ends = _line_info(
                    board=board,
                    row=row,
                    col=col,
                    player=player,
                    dr=dr,
                    dc=dc,
                    board_size=board_size,
                )
                score += _line_score(
                    length=length,
                    open_ends=open_ends,
                    connect=connect,
                )
    return score
