from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import torch
from torch import nn

from alphazero.games.base import Game, GameState
from alphazero.mcts.node import MCTSNode


class PositionEvaluator(Protocol):
    """
    MCTS 使用的局面评估接口。

    evaluate 返回：
        policy_logits: shape [action_size]
        value: 当前待行动玩家视角，范围通常为 [-1, 1]
    """

    def evaluate(
        self,
        state: GameState,
    ) -> tuple[np.ndarray, float]:
        ...


class TorchNetworkEvaluator:
    """
    使用 PyTorch Policy-Value 网络进行局面评估。

    MCTS 只依赖 PositionEvaluator，而不直接依赖具体网络类，
    方便后续替换为批量推理或 ONNX Runtime。
    """

    def __init__(
        self,
        game: Game,
        model: nn.Module,
        device: torch.device | str,
    ) -> None:
        self.game = game
        self.model = model
        self.device = torch.device(device)

        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(
        self,
        state: GameState,
    ) -> tuple[np.ndarray, float]:
        encoded = self.game.encode_state(state)

        states = (
            torch.from_numpy(encoded)
            .unsqueeze(0)
            .to(
                device=self.device,
                dtype=torch.float32,
            )
        )

        was_training = self.model.training
        self.model.eval()

        policy_logits, value = self.model(states)

        if was_training:
            self.model.train()

        policy_logits_np = (
            policy_logits[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        value_float = float(
            value[0, 0].detach().cpu().item()
        )

        if policy_logits_np.shape != (
            self.game.action_size,
        ):
            raise ValueError(
                "网络策略输出形状错误："
                f"期望 {(self.game.action_size,)}，"
                f"实际 {policy_logits_np.shape}"
            )

        if not np.all(np.isfinite(policy_logits_np)):
            raise ValueError(
                "网络 policy logits 包含 NaN 或 Inf"
            )

        if not np.isfinite(value_float):
            raise ValueError(
                "网络 value 包含 NaN 或 Inf"
            )

        # 防止随机初始化或数值误差导致越界。
        value_float = float(
            np.clip(value_float, -1.0, 1.0)
        )

        return policy_logits_np, value_float


@dataclass(frozen=True)
class MCTSConfig:
    """
    标准 PUCT MCTS 配置。
    """

    num_simulations: int = 64
    c_puct: float = 1.5

    # 自对弈时使用根节点 Dirichlet 噪声。
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # 给未访问子节点使用的初始 Q。
    # 第一版设为 0，不先实现复杂 FPU。
    initial_q: float = 0.0

    def validate(self) -> None:
        if self.num_simulations <= 0:
            raise ValueError(
                "num_simulations 必须大于 0"
            )

        if self.c_puct <= 0:
            raise ValueError("c_puct 必须大于 0")

        if self.dirichlet_alpha <= 0:
            raise ValueError(
                "dirichlet_alpha 必须大于 0"
            )

        if not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError(
                "dirichlet_epsilon 必须位于 [0, 1]"
            )

        if not np.isfinite(self.initial_q):
            raise ValueError(
                "initial_q 必须是有限数值"
            )


@dataclass
class SearchResult:
    """
    一次根节点搜索结果。
    """

    action: Optional[int]
    visit_policy: np.ndarray
    root_value: float
    root: MCTSNode


class MCTS:
    """
    标准 AlphaZero PUCT MCTS。

    搜索流程：
        1. 展开根节点
        2. PUCT 选择
        3. 到达叶子节点
        4. 网络评估或读取终局价值
        5. 价值逐层变号并回传
        6. 使用根节点访问次数生成策略目标
    """

    def __init__(
        self,
        game: Game,
        evaluator: PositionEvaluator,
        config: MCTSConfig | None = None,
        seed: Optional[int] = None,
    ) -> None:
        if config is None:
            config = MCTSConfig()

        config.validate()

        self.game = game
        self.evaluator = evaluator
        self.config = config
        self.rng = np.random.default_rng(seed)

    def search(
        self,
        state: GameState,
        *,
        add_root_noise: bool = False,
        temperature: float = 1.0,
    ) -> SearchResult:
        """
        从 state 开始搜索。

        add_root_noise:
            自对弈时设为 True。
            Arena 和人机对战时设为 False。

        temperature:
            仅影响最终动作选择和 visit_policy。
            不改变树内部搜索过程。

            temperature = 0:
                选择访问次数最大的动作。

            temperature > 0:
                policy ∝ visit_count ** (1 / temperature)
        """
        if temperature < 0:
            raise ValueError(
                "temperature 不能小于 0"
            )

        terminal_value = self.game.terminal_value(state)

        if terminal_value is not None:
            return SearchResult(
                action=None,
                visit_policy=np.zeros(
                    self.game.action_size,
                    dtype=np.float32,
                ),
                root_value=float(terminal_value),
                root=MCTSNode(prior=1.0),
            )

        root = MCTSNode(prior=1.0)

        # 搜索开始前先展开根节点。
        root_value = self._expand_and_evaluate(
            root,
            state,
        )

        if add_root_noise:
            self._add_dirichlet_noise(root)

        for _ in range(
            self.config.num_simulations
        ):
            self._run_simulation(
                root=root,
                root_state=state,
            )

        visit_policy = self._visit_policy(
            root=root,
            temperature=temperature,
        )

        action = self._sample_action(
            visit_policy=visit_policy,
            temperature=temperature,
        )

        # 搜索后的根节点 Q 比首次网络评估通常更可信。
        if root.visit_count > 0:
            root_value = root.q_value

        return SearchResult(
            action=action,
            visit_policy=visit_policy,
            root_value=float(root_value),
            root=root,
        )

    def _run_simulation(
        self,
        root: MCTSNode,
        root_state: GameState,
    ) -> None:
        node = root
        state = root_state

        # path 中每个节点的价值视角都交替变化。
        path = [node]

        while node.is_expanded:
            action, child = self._select_child(node)

            state = self.game.next_state(
                state,
                action,
            )

            node = child
            path.append(node)

        terminal_value = self.game.terminal_value(
            state
        )

        if terminal_value is not None:
            leaf_value = float(terminal_value)
        else:
            leaf_value = self._expand_and_evaluate(
                node,
                state,
            )

        self._backpropagate(
            path=path,
            leaf_value=leaf_value,
        )

    def _select_child(
        self,
        node: MCTSNode,
    ) -> tuple[int, MCTSNode]:
        """
        使用 PUCT 选择一个子节点。

        score =
            parent-perspective Q + exploration U

        child.q_value 是子节点玩家视角，因此父节点视角需要取负。
        """
        if not node.children:
            raise ValueError(
                "不能从未展开节点选择子节点"
            )

        parent_visits = max(
            1,
            node.visit_count,
        )

        best_score = -float("inf")
        best_actions: list[int] = []

        for action, child in node.children.items():
            if child.visit_count > 0:
                parent_q = -child.q_value
            else:
                parent_q = self.config.initial_q

            exploration = (
                self.config.c_puct
                * child.prior
                * np.sqrt(parent_visits)
                / (1 + child.visit_count)
            )

            score = parent_q + exploration

            if score > best_score + 1e-12:
                best_score = score
                best_actions = [action]

            elif abs(score - best_score) <= 1e-12:
                best_actions.append(action)

        selected_action = int(
            self.rng.choice(best_actions)
        )

        return (
            selected_action,
            node.children[selected_action],
        )

    def _expand_and_evaluate(
        self,
        node: MCTSNode,
        state: GameState,
    ) -> float:
        """
        使用网络评估非终局节点并展开合法动作。
        """
        policy_logits, value = (
            self.evaluator.evaluate(state)
        )

        policy_logits = np.asarray(
            policy_logits,
            dtype=np.float64,
        )

        if policy_logits.shape != (
            self.game.action_size,
        ):
            raise ValueError(
                "评估器 policy logits 形状错误："
                f"期望 {(self.game.action_size,)}，"
                f"实际 {policy_logits.shape}"
            )

        if not np.all(np.isfinite(policy_logits)):
            raise ValueError(
                "评估器 policy logits 包含 NaN 或 Inf"
            )

        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                "评估器 value 包含 NaN 或 Inf"
            )

        legal_mask = self.game.legal_actions(state)

        priors = self._masked_softmax(
            logits=policy_logits,
            legal_mask=legal_mask,
        )

        action_priors = {
            int(action): float(priors[action])
            for action in np.flatnonzero(legal_mask)
        }

        node.expand(action_priors)

        return float(np.clip(value, -1.0, 1.0))

    @staticmethod
    def _masked_softmax(
        logits: np.ndarray,
        legal_mask: np.ndarray,
    ) -> np.ndarray:
        """
        只在合法动作上计算稳定 Softmax。
        """
        logits = np.asarray(
            logits,
            dtype=np.float64,
        )

        legal_mask = np.asarray(
            legal_mask,
            dtype=np.float64,
        )

        if logits.shape != legal_mask.shape:
            raise ValueError(
                "logits 和 legal_mask 形状必须相同"
            )

        legal_indices = np.flatnonzero(
            legal_mask > 0
        )

        if legal_indices.size == 0:
            raise ValueError(
                "非终局状态没有合法动作"
            )

        legal_logits = logits[legal_indices]
        max_logit = np.max(legal_logits)

        exp_logits = np.exp(
            legal_logits - max_logit
        )

        denominator = exp_logits.sum()

        probabilities = np.zeros_like(
            logits,
            dtype=np.float64,
        )

        if (
            not np.isfinite(denominator)
            or denominator <= 0
        ):
            probabilities[legal_indices] = (
                1.0 / legal_indices.size
            )
        else:
            probabilities[legal_indices] = (
                exp_logits / denominator
            )

        return probabilities

    def _add_dirichlet_noise(
        self,
        root: MCTSNode,
    ) -> None:
        """
        给根节点合法动作的先验概率加入 Dirichlet 噪声。
        """
        if not root.children:
            return

        actions = list(root.children.keys())

        noise = self.rng.dirichlet(
            np.full(
                len(actions),
                self.config.dirichlet_alpha,
                dtype=np.float64,
            )
        )

        epsilon = (
            self.config.dirichlet_epsilon
        )

        for action, noise_value in zip(
            actions,
            noise,
        ):
            child = root.children[action]

            child.prior = float(
                (1.0 - epsilon) * child.prior
                + epsilon * noise_value
            )

    @staticmethod
    def _backpropagate(
        path: list[MCTSNode],
        leaf_value: float,
    ) -> None:
        """
        从叶子向根节点回传价值。

        leaf_value 是叶子节点当前玩家视角。
        每向上一层，当前玩家发生切换，因此价值取反。
        """
        value = float(leaf_value)

        for node in reversed(path):
            node.update(value)
            value = -value

    def _visit_policy(
        self,
        root: MCTSNode,
        temperature: float,
    ) -> np.ndarray:
        """
        根据根节点子节点访问次数生成策略目标。
        """
        counts = root.child_visit_counts(
            self.game.action_size
        ).astype(np.float64)

        legal_actions = np.array(
            list(root.children.keys()),
            dtype=np.int64,
        )

        policy = np.zeros(
            self.game.action_size,
            dtype=np.float64,
        )

        if legal_actions.size == 0:
            return policy.astype(np.float32)

        if temperature == 0:
            maximum = np.max(counts[legal_actions])

            best_actions = legal_actions[
                counts[legal_actions] == maximum
            ]

            # temperature=0 时仍可能出现并列，
            # 在并列动作中随机选择一个。
            selected = int(
                self.rng.choice(best_actions)
            )

            policy[selected] = 1.0
            return policy.astype(np.float32)

        exponent = 1.0 / temperature

        positive_counts = counts[
            legal_actions
        ]

        # 当前配置 num_simulations > 0，正常情况下访问次数和大于 0。
        if positive_counts.sum() <= 0:
            priors = root.child_priors(
                self.game.action_size
            ).astype(np.float64)

            prior_sum = priors[
                legal_actions
            ].sum()

            if prior_sum <= 0:
                policy[legal_actions] = (
                    1.0 / legal_actions.size
                )
            else:
                policy[legal_actions] = (
                    priors[legal_actions]
                    / prior_sum
                )

            return policy.astype(np.float32)

        # 使用 log-space，避免极低温度下 counts ** exponent 溢出。
        scaled_logits = np.full_like(
            counts,
            -np.inf,
            dtype=np.float64,
        )

        visited = counts > 0

        scaled_logits[visited] = (
            np.log(counts[visited])
            * exponent
        )

        legal_scaled = scaled_logits[
            legal_actions
        ]

        max_scaled = np.max(legal_scaled)
        weights = np.exp(
            legal_scaled - max_scaled
        )

        weight_sum = weights.sum()

        if (
            not np.isfinite(weight_sum)
            or weight_sum <= 0
        ):
            policy[legal_actions] = (
                1.0 / legal_actions.size
            )
        else:
            policy[legal_actions] = (
                weights / weight_sum
            )

        return policy.astype(np.float32)

    def _sample_action(
        self,
        visit_policy: np.ndarray,
        temperature: float,
    ) -> Optional[int]:
        """
        从 visit policy 选择实际动作。
        """
        probability_sum = float(
            visit_policy.sum()
        )

        if probability_sum <= 0:
            return None

        probabilities = (
            visit_policy.astype(np.float64)
            / probability_sum
        )

        if temperature == 0:
            return int(np.argmax(probabilities))

        return int(
            self.rng.choice(
                self.game.action_size,
                p=probabilities,
            )
        )
