from __future__ import annotations

import numpy as np
import torch

from alphazero.mcts.search import MCTSConfig
from alphazero.networks.residual_net import NetworkConfig, PolicyValueNet
from alphazero.selfplay.episode import SelfPlayConfig
from alphazero.selfplay.worker import play_episodes_parallel


def build_model() -> PolicyValueNet:
    return PolicyValueNet(
        NetworkConfig(
            board_size=5,
            input_channels=3,
            channels=8,
            residual_blocks=1,
            value_hidden_channels=4,
            value_hidden_size=8,
        )
    )


def test_parallel_selfplay_returns_valid_episodes() -> None:
    torch.manual_seed(42)

    episodes = play_episodes_parallel(
        board_size=5,
        connect=4,
        model=build_model(),
        device="cpu",
        games=2,
        workers=2,
        mcts_config=MCTSConfig(
            num_simulations=2,
            c_puct=1.5,
        ),
        self_play_config=SelfPlayConfig(
            temperature_moves=5,
            sampling_temperature=1.0,
            add_root_noise=True,
            augment_symmetries=False,
        ),
        base_seed=42,
        iteration=1,
        inference_batch_size=2,
        batch_wait_ms=1.0,
    )

    assert len(episodes) == 2

    for episode in episodes:
        assert episode.winner in (-1, 0, 1)
        assert 1 <= episode.move_count <= 25
        assert len(episode.examples) == episode.move_count

        for example in episode.examples:
            assert example.state.shape == (3, 5, 5)
            assert example.policy.shape == (25,)
            assert np.isclose(example.policy.sum(), 1.0)
            assert example.value in (-1.0, 0.0, 1.0)


def test_parallel_selfplay_limits_workers_to_games() -> None:
    episodes = play_episodes_parallel(
        board_size=5,
        connect=4,
        model=build_model(),
        device="cpu",
        games=1,
        workers=4,
        mcts_config=MCTSConfig(num_simulations=1),
        self_play_config=SelfPlayConfig(
            augment_symmetries=False,
        ),
        base_seed=7,
        iteration=1,
        inference_batch_size=4,
        batch_wait_ms=1.0,
    )

    assert len(episodes) == 1
    assert episodes[0].move_count > 0
