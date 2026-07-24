from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class MCTSNode:
    """
    MCTS 树节点。

    prior:
        父节点提供的先验概率 P(s, a)。

    visit_count:
        节点访问次数 N。

    value_sum:
        从“本节点当前待行动玩家视角”累计的价值。

    children:
        action -> child node。

    价值视角约定：
        child.q_value 是子节点当前玩家的视角。
        父节点选择 child 时，应使用 -child.q_value。
    """

    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "MCTSNode"] = field(
        default_factory=dict
    )

    @property
    def is_expanded(self) -> bool:
        """节点是否已经展开。"""
        return len(self.children) > 0

    @property
    def q_value(self) -> float:
        """
        本节点当前待行动玩家视角下的平均价值。
        """
        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count

    def expand(
        self,
        action_priors: dict[int, float],
    ) -> None:
        """
        根据合法动作及其先验概率创建子节点。

        已存在的子节点不会被覆盖。
        """
        for action, prior in action_priors.items():
            action = int(action)
            prior = float(prior)

            if action < 0:
                raise ValueError("action 不能小于 0")

            if not np.isfinite(prior):
                raise ValueError("prior 必须是有限数值")

            if prior < 0.0:
                raise ValueError("prior 不能小于 0")

            if action not in self.children:
                self.children[action] = MCTSNode(
                    prior=prior
                )

    def update(self, value: float) -> None:
        """
        使用本节点当前玩家视角的 value 更新节点。
        """
        value = float(value)

        if not np.isfinite(value):
            raise ValueError("value 必须是有限数值")

        self.visit_count += 1
        self.value_sum += value

    def child_visit_counts(
        self,
        action_size: int,
    ) -> np.ndarray:
        """
        返回长度为 action_size 的子节点访问次数数组。
        """
        if action_size <= 0:
            raise ValueError("action_size 必须大于 0")

        counts = np.zeros(
            action_size,
            dtype=np.float32,
        )

        for action, child in self.children.items():
            if action >= action_size:
                raise ValueError(
                    f"子节点动作 {action} 超出动作空间"
                )

            counts[action] = float(child.visit_count)

        return counts

    def child_priors(
        self,
        action_size: int,
    ) -> np.ndarray:
        """
        返回长度为 action_size 的子节点先验概率数组。
        """
        if action_size <= 0:
            raise ValueError("action_size 必须大于 0")

        priors = np.zeros(
            action_size,
            dtype=np.float32,
        )

        for action, child in self.children.items():
            if action >= action_size:
                raise ValueError(
                    f"子节点动作 {action} 超出动作空间"
                )

            priors[action] = float(child.prior)

        return priors
