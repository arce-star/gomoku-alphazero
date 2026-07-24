from __future__ import annotations

import argparse
import time

import torch

from alphazero.mcts.search import MCTSConfig
from alphazero.networks.residual_net import NetworkConfig, PolicyValueNet
from alphazero.selfplay.episode import SelfPlayConfig
from alphazero.selfplay.worker import play_episodes_parallel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(42)

    model = PolicyValueNet(
        NetworkConfig(
            board_size=9,
            input_channels=3,
            channels=64,
            residual_blocks=5,
            value_hidden_channels=32,
            value_hidden_size=128,
        )
    )

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    started = time.perf_counter()

    episodes = play_episodes_parallel(
        board_size=9,
        connect=5,
        model=model,
        device=args.device,
        games=args.games,
        workers=args.workers,
        mcts_config=MCTSConfig(
            num_simulations=args.simulations,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        ),
        self_play_config=SelfPlayConfig(
            temperature_moves=15,
            sampling_temperature=1.0,
            add_root_noise=True,
            augment_symmetries=False,
        ),
        base_seed=42,
        iteration=1,
        inference_batch_size=args.workers,
        batch_wait_ms=2.0,
    )

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    moves = sum(episode.move_count for episode in episodes)
    evaluations = sum(
        episode.move_count * (args.simulations + 1)
        for episode in episodes
    )

    print(f"workers: {args.workers}")
    print(f"games: {len(episodes)}")
    print(f"moves: {moves}")
    print(f"seconds: {elapsed:.3f}")
    print(f"games/hour: {len(episodes) * 3600 / elapsed:.1f}")
    print(f"moves/second: {moves / elapsed:.1f}")
    print(f"estimated evaluations/second: {evaluations / elapsed:.1f}")


if __name__ == "__main__":
    main()
