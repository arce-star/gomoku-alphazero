from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class GameState:
    """
    通用棋局状态。

    board:
        0  表示空位
        1  表示黑棋
        -1 表示白棋

    current_player:
        当前准备行动的玩家，取值只能是 1 或 -1。

    move_count:
        已经落下的棋子总数。

    last_action:
        上一步动作编号。初始状态为 None。
    """

    board: np.ndarray
    current_player: int
    move_count: int
    last_action: Optional[int] = None

    def __post_init__(self) -> None:
        # 复制数组，防止外部代码修改 GameState 内部棋盘。
        board = np.asarray(self.board, dtype=np.int8).copy()
        board.setflags(write=False)
        object.__setattr__(self, "board", board)


class Game(ABC):
    """
    AlphaZero 使用的游戏规则基础接口。
    """

    @property
    @abstractmethod
    def board_size(self) -> int:
        """棋盘边长。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def action_size(self) -> int:
        """动作空间大小。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def input_channels(self) -> int:
        """神经网络输入通道数。"""
        raise NotImplementedError

    @abstractmethod
    def initial_state(self) -> GameState:
        """创建初始棋局。"""
        raise NotImplementedError

    @abstractmethod
    def legal_actions(self, state: GameState) -> np.ndarray:
        """
        返回合法动作掩码。

        返回形状为 [action_size] 的 float32 数组：
        1.0 表示合法动作，0.0 表示非法动作。
        """
        raise NotImplementedError

    @abstractmethod
    def next_state(self, state: GameState, action: int) -> GameState:
        """执行动作并返回新的棋局状态。"""
        raise NotImplementedError

    @abstractmethod
    def winner(self, state: GameState) -> int:
        """
        返回获胜者：

        1  表示黑棋获胜
        -1 表示白棋获胜
        0  表示当前没有获胜者
        """
        raise NotImplementedError

    @abstractmethod
    def terminal_value(self, state: GameState) -> Optional[float]:
        """
        判断棋局是否结束。

        返回值始终从 state.current_player 的视角计算：

        1.0  当前玩家获胜
        -1.0 当前玩家失败
        0.0  平局
        None 棋局尚未结束
        """
        raise NotImplementedError

    @abstractmethod
    def encode_state(self, state: GameState) -> np.ndarray:
        """
        将棋局编码为神经网络输入。

        返回形状：
            [input_channels, board_size, board_size]
        """
        raise NotImplementedError
