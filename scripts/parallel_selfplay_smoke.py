from __future__ import annotations

import torch

from alphazero.mcts.search import MCTSConfig
from alphazero.networks.residual_net import NetworkConfig, PolicyValueNet
from alphazero.selfplay.episode import SelfPlayConfig
from alphazero.selfplay.worker import play_episodes_parallel


def main() -> None:
    model = PolicyValueNet(
        NetworkConfig(
            board_size=5,
            input_channels=3,
            channels=16,
            residual_blocks=1,
            value_hidden_channels=8,
            value_hidden_size=16,
        )
    )

    episodes = play_episodes_parallel(
        board_size=5,
        connect=4,
        model=model,
        device="cuda",
        games=4,
        workers=2,
        mcts_config=MCTSConfig(num_simulations=4),
        self_play_config=SelfPlayConfig(
            temperature_moves=5,
            augment_symmetries=False,
        ),
        base_seed=42,
        iteration=1,
        inference_batch_size=8,
    )

    assert len(episodes) == 4
    assert all(episode.move_count > 0 for episode in episodes)
    assert all(episode.examples for episode in episodes)

    print("games:", len(episodes))
    print("moves:", [episode.move_count for episode in episodes])
    print("examples:", sum(len(x.examples) for x in episodes))
    print("parallel self-play passed")


if __name__ == "__main__":
    main()
