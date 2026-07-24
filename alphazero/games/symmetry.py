from __future__ import annotations

import numpy as np


def apply_symmetry(
    encoded_state: np.ndarray,
    policy: np.ndarray,
    board_size: int,
    rotation: int,
    flip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对神经网络状态和策略目标执行相同的棋盘对称变换。

    参数：
        encoded_state:
            shape = [channels, board_size, board_size]

        policy:
            shape = [board_size * board_size]

        board_size:
            棋盘边长。

        rotation:
            逆时针旋转次数，必须为 0、1、2、3。
            每次旋转 90 度。

        flip:
            是否在旋转后沿水平方向翻转。

    返回：
        transformed_state:
            shape 与 encoded_state 相同。

        transformed_policy:
            shape 与 policy 相同。
    """
    encoded_state = np.asarray(encoded_state)
    policy = np.asarray(policy)

    if board_size <= 0:
        raise ValueError("board_size 必须大于 0")

    if encoded_state.ndim != 3:
        raise ValueError(
            "encoded_state 必须是三维数组 "
            "[channels, height, width]"
        )

    expected_state_shape = (
        encoded_state.shape[0],
        board_size,
        board_size,
    )

    if encoded_state.shape != expected_state_shape:
        raise ValueError(
            f"encoded_state 形状必须是 {expected_state_shape}，"
            f"实际得到 {encoded_state.shape}"
        )

    expected_policy_shape = (
        board_size * board_size,
    )

    if policy.shape != expected_policy_shape:
        raise ValueError(
            f"policy 形状必须是 {expected_policy_shape}，"
            f"实际得到 {policy.shape}"
        )

    if rotation not in (0, 1, 2, 3):
        raise ValueError(
            "rotation 必须是 0、1、2、3"
        )

    if not isinstance(flip, (bool, np.bool_)):
        raise TypeError("flip 必须是布尔值")

    policy_board = policy.reshape(
        board_size,
        board_size,
    )

    # np.rot90 默认逆时针旋转。
    transformed_state = np.rot90(
        encoded_state,
        k=rotation,
        axes=(1, 2),
    )

    transformed_policy_board = np.rot90(
        policy_board,
        k=rotation,
        axes=(0, 1),
    )

    if flip:
        # 沿棋盘左右方向翻转。
        transformed_state = np.flip(
            transformed_state,
            axis=2,
        )

        transformed_policy_board = np.flip(
            transformed_policy_board,
            axis=1,
        )

    # rot90 和 flip 可能返回负 stride 的视图。
    # torch.from_numpy 不支持负 stride，因此必须 contiguous copy。
    transformed_state = np.ascontiguousarray(
        transformed_state,
        dtype=np.float32,
    )

    transformed_policy = np.ascontiguousarray(
        transformed_policy_board.reshape(-1),
        dtype=np.float32,
    )

    return transformed_state, transformed_policy


def generate_symmetries(
    encoded_state: np.ndarray,
    policy: np.ndarray,
    board_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    生成正方形棋盘的 8 个二面体对称样本。

    即：
        4 个旋转
        ×
        是否水平翻转
    """
    samples: list[
        tuple[np.ndarray, np.ndarray]
    ] = []

    for rotation in range(4):
        for flip in (False, True):
            transformed = apply_symmetry(
                encoded_state=encoded_state,
                policy=policy,
                board_size=board_size,
                rotation=rotation,
                flip=flip,
            )

            samples.append(transformed)

    return samples
