from __future__ import annotations

import numpy as np
import pytest
import torch

from alphazero.games.gomoku import GomokuGame
from alphazero.mcts.node import MCTSNode
from alphazero.mcts.search import (
    MCTS,
    MCTSConfig,
    TorchNetworkEvaluator,
)
from alphazero.networks.residual_net import (
    NetworkConfig,
    PolicyValueNet,
)


class UniformEvaluator:
    """
    测试使用的固定评估器。

    所有动作 logit 相同，value 固定。
    """

    def __init__(
        self,
        action_size: int,
        value: float = 0.0,
    ) -> None:
        self.action_size = action_size
        self.value = value
        self.call_count = 0

    def evaluate(self, state):
        self.call_count += 1

        return (
            np.zeros(
                self.action_size,
                dtype=np.float64,
            ),
            self.value,
        )


class PreferredActionEvaluator:
    """
    对指定动作给出更高策略先验的评估器。
    """

    def __init__(
        self,
        action_size: int,
        preferred_action: int,
    ) -> None:
        self.action_size = action_size
        self.preferred_action = preferred_action
        self.call_count = 0

    def evaluate(self, state):
        self.call_count += 1

        logits = np.zeros(
            self.action_size,
            dtype=np.float64,
        )

        logits[self.preferred_action] = 8.0

        return logits, 0.0


@pytest.fixture
def game() -> GomokuGame:
    return GomokuGame(
        board_size=9,
        connect=5,
    )


def test_node_defaults() -> None:
    node = MCTSNode(prior=0.5)

    assert node.prior == 0.5
    assert node.visit_count == 0
    assert node.value_sum == 0.0
    assert node.q_value == 0.0
    assert node.is_expanded is False


def test_node_update() -> None:
    node = MCTSNode(prior=1.0)

    node.update(1.0)
    node.update(-0.5)

    assert node.visit_count == 2
    assert node.value_sum == pytest.approx(0.5)
    assert node.q_value == pytest.approx(0.25)


def test_node_expand() -> None:
    node = MCTSNode(prior=1.0)

    node.expand(
        {
            0: 0.25,
            3: 0.75,
        }
    )

    assert node.is_expanded
    assert set(node.children) == {0, 3}
    assert node.children[0].prior == 0.25
    assert node.children[3].prior == 0.75


def test_backpropagation_changes_sign() -> None:
    root = MCTSNode(prior=1.0)
    child = MCTSNode(prior=0.5)
    leaf = MCTSNode(prior=0.5)

    MCTS._backpropagate(
        path=[root, child, leaf],
        leaf_value=1.0,
    )

    # 叶子当前玩家视角：+1
    assert leaf.q_value == pytest.approx(1.0)

    # 向上一层玩家切换：-1
    assert child.q_value == pytest.approx(-1.0)

    # 再向上一层玩家再次切换：+1
    assert root.q_value == pytest.approx(1.0)


def test_masked_softmax_excludes_illegal_actions() -> None:
    logits = np.array(
        [100.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    legal_mask = np.array(
        [0.0, 1.0, 1.0, 0.0],
        dtype=np.float32,
    )

    probabilities = MCTS._masked_softmax(
        logits,
        legal_mask,
    )

    assert probabilities[0] == 0.0
    assert probabilities[3] == 0.0

    assert probabilities[1] == pytest.approx(0.5)
    assert probabilities[2] == pytest.approx(0.5)

    assert probabilities.sum() == pytest.approx(1.0)


def test_search_returns_legal_normalized_policy(
    game: GomokuGame,
) -> None:
    evaluator = UniformEvaluator(
        game.action_size
    )

    mcts = MCTS(
        game=game,
        evaluator=evaluator,
        config=MCTSConfig(
            num_simulations=32,
            c_puct=1.5,
        ),
        seed=42,
    )

    state = game.initial_state()

    result = mcts.search(
        state,
        add_root_noise=False,
        temperature=1.0,
    )

    assert result.action is not None
    assert 0 <= result.action < game.action_size

    assert result.visit_policy.shape == (81,)
    assert result.visit_policy.dtype == np.float32

    assert np.all(result.visit_policy >= 0)
    assert result.visit_policy.sum() == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert result.root.visit_count == 32

    child_visits = sum(
        child.visit_count
        for child in result.root.children.values()
    )

    assert child_visits == 32
    assert evaluator.call_count >= 1


def test_search_never_selects_occupied_action(
    game: GomokuGame,
) -> None:
    state = game.initial_state()

    occupied_action = game.coord_to_action(4, 4)
    state = game.next_state(
        state,
        occupied_action,
    )

    evaluator = PreferredActionEvaluator(
        action_size=game.action_size,
        preferred_action=occupied_action,
    )

    mcts = MCTS(
        game=game,
        evaluator=evaluator,
        config=MCTSConfig(
            num_simulations=32,
        ),
        seed=123,
    )

    result = mcts.search(
        state,
        add_root_noise=False,
        temperature=0,
    )

    assert result.action != occupied_action
    assert result.visit_policy[occupied_action] == 0.0
    assert occupied_action not in result.root.children


def test_search_finds_immediate_win(
    game: GomokuGame,
) -> None:
    state = game.initial_state()

    # 8 步后轮到黑棋。
    # 黑棋在第 0 行已经有 4 连，(0, 4) 是立即获胜动作。
    moves = [
        (0, 0), (8, 0),
        (0, 1), (8, 1),
        (0, 2), (8, 2),
        (0, 3), (8, 3),
    ]

    for row, col in moves:
        state = game.next_state(
            state,
            game.coord_to_action(row, col),
        )

    winning_action = game.coord_to_action(
        0,
        4,
    )

    evaluator = PreferredActionEvaluator(
        action_size=game.action_size,
        preferred_action=winning_action,
    )

    mcts = MCTS(
        game=game,
        evaluator=evaluator,
        config=MCTSConfig(
            num_simulations=32,
            c_puct=1.5,
        ),
        seed=7,
    )

    result = mcts.search(
        state,
        add_root_noise=False,
        temperature=0,
    )

    assert result.action == winning_action

    winning_child = result.root.children[
        winning_action
    ]

    # 落下获胜动作后轮到白棋，
    # 因此获胜子节点从白棋视角为 -1。
    assert winning_child.q_value == pytest.approx(-1.0)

    next_state = game.next_state(
        state,
        result.action,
    )

    assert game.winner(next_state) == 1
    assert game.terminal_value(next_state) == -1.0


def test_terminal_state_does_not_call_evaluator(
    game: GomokuGame,
) -> None:
    state = game.initial_state()

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

    evaluator = UniformEvaluator(
        game.action_size
    )

    mcts = MCTS(
        game=game,
        evaluator=evaluator,
        config=MCTSConfig(
            num_simulations=8,
        ),
        seed=0,
    )

    result = mcts.search(state)

    assert result.action is None
    assert result.root_value == -1.0
    assert result.visit_policy.sum() == 0.0
    assert evaluator.call_count == 0


def test_temperature_zero_returns_one_hot_policy(
    game: GomokuGame,
) -> None:
    evaluator = UniformEvaluator(
        game.action_size
    )

    mcts = MCTS(
        game=game,
        evaluator=evaluator,
        config=MCTSConfig(
            num_simulations=16,
        ),
        seed=11,
    )

    result = mcts.search(
        game.initial_state(),
        temperature=0,
    )

    assert result.action is not None
    assert result.visit_policy.sum() == pytest.approx(1.0)
    assert np.count_nonzero(result.visit_policy) == 1
    assert result.visit_policy[result.action] == 1.0


def test_root_dirichlet_noise_preserves_prior_sum(
    game: GomokuGame,
) -> None:
    evaluator = UniformEvaluator(
        game.action_size
    )

    mcts = MCTS(
        game=game,
        evaluator=evaluator,
        config=MCTSConfig(
            num_simulations=8,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        ),
        seed=99,
    )

    result = mcts.search(
        game.initial_state(),
        add_root_noise=True,
        temperature=1.0,
    )

    priors = np.array(
        [
            child.prior
            for child in result.root.children.values()
        ]
    )

    assert np.all(priors >= 0)
    assert priors.sum() == pytest.approx(
        1.0,
        abs=1e-6,
    )

    # 原始均匀先验每个动作都是 1/81。
    # 加入噪声后应该不再全部相同。
    assert np.std(priors) > 0


def test_torch_network_evaluator(
    game: GomokuGame,
) -> None:
    model = PolicyValueNet(
        NetworkConfig(
            board_size=9,
            input_channels=3,
            channels=16,
            residual_blocks=1,
            value_hidden_channels=8,
            value_hidden_size=16,
        )
    )

    evaluator = TorchNetworkEvaluator(
        game=game,
        model=model,
        device="cpu",
    )

    logits, value = evaluator.evaluate(
        game.initial_state()
    )

    assert logits.shape == (81,)
    assert np.all(np.isfinite(logits))
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA 不可用",
)
def test_mcts_with_cuda_network(
    game: GomokuGame,
) -> None:
    model = PolicyValueNet(
        NetworkConfig(
            board_size=9,
            input_channels=3,
            channels=16,
            residual_blocks=1,
            value_hidden_channels=8,
            value_hidden_size=16,
        )
    )

    evaluator = TorchNetworkEvaluator(
        game=game,
        model=model,
        device="cuda",
    )

    mcts = MCTS(
        game=game,
        evaluator=evaluator,
        config=MCTSConfig(
            num_simulations=8,
        ),
        seed=21,
    )

    result = mcts.search(
        game.initial_state(),
        add_root_noise=False,
        temperature=1.0,
    )

    assert result.action is not None
    assert result.visit_policy.sum() == pytest.approx(
        1.0,
        abs=1e-6,
    )
