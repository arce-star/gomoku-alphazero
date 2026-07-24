import numpy as np
import pytest

from alphazero.games.symmetry import (
    apply_symmetry,
    generate_symmetries,
)


def test_identity_symmetry() -> None:
    state = np.zeros(
        (3, 9, 9),
        dtype=np.float32,
    )

    state[0, 2, 3] = 1.0

    policy = np.zeros(
        81,
        dtype=np.float32,
    )

    policy[2 * 9 + 3] = 1.0

    transformed_state, transformed_policy = (
        apply_symmetry(
            encoded_state=state,
            policy=policy,
            board_size=9,
            rotation=0,
            flip=False,
        )
    )

    assert np.array_equal(
        transformed_state,
        state,
    )

    assert np.array_equal(
        transformed_policy,
        policy,
    )


@pytest.mark.parametrize(
    "rotation",
    [0, 1, 2, 3],
)
@pytest.mark.parametrize(
    "flip",
    [False, True],
)
def test_state_and_policy_transform_together(
    rotation: int,
    flip: bool,
) -> None:
    state = np.zeros(
        (3, 9, 9),
        dtype=np.float32,
    )

    policy_board = np.zeros(
        (9, 9),
        dtype=np.float32,
    )

    # 在状态和策略的相同位置放置唯一标记。
    state[0, 1, 3] = 1.0
    policy_board[1, 3] = 1.0

    transformed_state, transformed_policy = (
        apply_symmetry(
            encoded_state=state,
            policy=policy_board.reshape(-1),
            board_size=9,
            rotation=rotation,
            flip=flip,
        )
    )

    state_position = np.argwhere(
        transformed_state[0] == 1.0
    )

    policy_position = np.argwhere(
        transformed_policy.reshape(9, 9) == 1.0
    )

    assert state_position.shape == (1, 2)
    assert policy_position.shape == (1, 2)

    assert np.array_equal(
        state_position,
        policy_position,
    )


def test_generate_eight_symmetries() -> None:
    state = np.zeros(
        (3, 9, 9),
        dtype=np.float32,
    )

    state[0, 1, 2] = 1.0
    state[1, 6, 7] = 1.0

    policy = np.zeros(
        81,
        dtype=np.float32,
    )

    policy[1 * 9 + 2] = 0.7
    policy[6 * 9 + 7] = 0.3

    samples = generate_symmetries(
        encoded_state=state,
        policy=policy,
        board_size=9,
    )

    assert len(samples) == 8

    for transformed_state, transformed_policy in samples:
        assert transformed_state.shape == (3, 9, 9)
        assert transformed_policy.shape == (81,)

        assert transformed_state.dtype == np.float32
        assert transformed_policy.dtype == np.float32

        assert transformed_state.flags.c_contiguous
        assert transformed_policy.flags.c_contiguous

        assert transformed_policy.sum() == pytest.approx(
            1.0
        )


def test_four_rotations_restore_original() -> None:
    state = np.arange(
        3 * 9 * 9,
        dtype=np.float32,
    ).reshape(3, 9, 9)

    policy = np.arange(
        81,
        dtype=np.float32,
    )

    transformed_state, transformed_policy = (
        apply_symmetry(
            encoded_state=state,
            policy=policy,
            board_size=9,
            rotation=0,
            flip=False,
        )
    )

    for _ in range(4):
        transformed_state, transformed_policy = (
            apply_symmetry(
                encoded_state=transformed_state,
                policy=transformed_policy,
                board_size=9,
                rotation=1,
                flip=False,
            )
        )

    assert np.array_equal(
        transformed_state,
        state,
    )

    assert np.array_equal(
        transformed_policy,
        policy,
    )


def test_double_flip_restores_original() -> None:
    state = np.arange(
        3 * 9 * 9,
        dtype=np.float32,
    ).reshape(3, 9, 9)

    policy = np.arange(
        81,
        dtype=np.float32,
    )

    state_once, policy_once = apply_symmetry(
        encoded_state=state,
        policy=policy,
        board_size=9,
        rotation=0,
        flip=True,
    )

    state_twice, policy_twice = apply_symmetry(
        encoded_state=state_once,
        policy=policy_once,
        board_size=9,
        rotation=0,
        flip=True,
    )

    assert np.array_equal(
        state_twice,
        state,
    )

    assert np.array_equal(
        policy_twice,
        policy,
    )


def test_invalid_shapes() -> None:
    state = np.zeros(
        (3, 9, 9),
        dtype=np.float32,
    )

    policy = np.zeros(
        81,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        apply_symmetry(
            encoded_state=np.zeros(
                (9, 9),
                dtype=np.float32,
            ),
            policy=policy,
            board_size=9,
            rotation=0,
            flip=False,
        )

    with pytest.raises(ValueError):
        apply_symmetry(
            encoded_state=state,
            policy=np.zeros(
                80,
                dtype=np.float32,
            ),
            board_size=9,
            rotation=0,
            flip=False,
        )

    with pytest.raises(ValueError):
        apply_symmetry(
            encoded_state=state,
            policy=policy,
            board_size=9,
            rotation=4,
            flip=False,
        )
