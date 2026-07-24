from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alphazero.games.gomoku import GomokuGame
from alphazero.mcts.search import (
    MCTS,
    MCTSConfig,
    TorchNetworkEvaluator,
)
from alphazero.networks.residual_net import (
    NetworkConfig,
    PolicyValueNet,
)
from alphazero.selfplay.episode import (
    SelfPlayConfig,
    play_episode,
)
from alphazero.training.replay_buffer import ReplayBuffer
from alphazero.training.trainer import (
    Trainer,
    TrainerConfig,
)
from alphazero.utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from alphazero.utils.config import (
    apply_overrides,
    load_config,
)
from alphazero.utils.seed import seed_everything


def build_model(
    config: dict[str, Any],
) -> PolicyValueNet:
    network = config["network"]
    game = config["game"]

    return PolicyValueNet(
        NetworkConfig(
            board_size=int(game["board_size"]),
            input_channels=int(
                network["input_channels"]
            ),
            channels=int(network["channels"]),
            residual_blocks=int(
                network["residual_blocks"]
            ),
            value_hidden_channels=int(
                network["value_hidden_channels"]
            ),
            value_hidden_size=int(
                network["value_hidden_size"]
            ),
        )
    )


def build_trainer(
    model: PolicyValueNet,
    config: dict[str, Any],
    device: torch.device,
) -> Trainer:
    training = config["training"]

    max_grad_norm = training.get(
        "max_grad_norm"
    )

    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)

    return Trainer(
        model=model,
        device=device,
        config=TrainerConfig(
            learning_rate=float(
                training["learning_rate"]
            ),
            weight_decay=float(
                training["weight_decay"]
            ),
            value_loss_weight=float(
                training["value_loss_weight"]
            ),
            max_grad_norm=max_grad_norm,
            use_amp=bool(training["use_amp"]),
        ),
    )


def run_smoke_test(
    config: dict[str, Any],
    *,
    device: torch.device,
) -> Path:
    started_at = time.perf_counter()

    experiment = config["experiment"]
    game_config = config["game"]
    mcts_config = config["mcts"]
    self_play_config = config["self_play"]
    training_config = config["training"]

    seed = int(experiment["seed"])
    seed_everything(seed)

    game = GomokuGame(
        board_size=int(
            game_config["board_size"]
        ),
        connect=int(game_config["connect"]),
    )

    model = build_model(config)
    trainer = build_trainer(
        model=model,
        config=config,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        capacity=int(
            training_config["replay_capacity"]
        ),
        seed=seed,
    )

    print("=== Smoke Test ===")
    print("Device:", trainer.device)
    print("AMP enabled:", trainer.use_amp)
    print(
        "Game:",
        f"{game.board_size}x{game.board_size}",
        f"connect={game.connect}",
    )
    print(
        "Model parameters:",
        f"{model.parameter_count():,}",
    )

    self_play_started = time.perf_counter()

    winner_counts = {
        1: 0,
        -1: 0,
        0: 0,
    }

    total_moves = 0
    games_per_iteration = int(
        self_play_config[
            "games_per_iteration"
        ]
    )

    for game_index in range(
        games_per_iteration
    ):
        # 第一版每局创建新树，不复用上一局搜索树。
        evaluator = TorchNetworkEvaluator(
            game=game,
            model=model,
            device=trainer.device,
        )

        mcts = MCTS(
            game=game,
            evaluator=evaluator,
            config=MCTSConfig(
                num_simulations=int(
                    mcts_config[
                        "num_simulations"
                    ]
                ),
                c_puct=float(
                    mcts_config["c_puct"]
                ),
                dirichlet_alpha=float(
                    mcts_config[
                        "dirichlet_alpha"
                    ]
                ),
                dirichlet_epsilon=float(
                    mcts_config[
                        "dirichlet_epsilon"
                    ]
                ),
            ),
            seed=seed + game_index,
        )

        episode = play_episode(
            game=game,
            mcts=mcts,
            config=SelfPlayConfig(
                temperature_moves=int(
                    self_play_config[
                        "temperature_moves"
                    ]
                ),
                sampling_temperature=float(
                    self_play_config[
                        "sampling_temperature"
                    ]
                ),
                add_root_noise=bool(
                    self_play_config[
                        "add_root_noise"
                    ]
                ),
                augment_symmetries=bool(
                    self_play_config[
                        "augment_symmetries"
                    ]
                ),
            ),
        )

        replay_buffer.extend(
            episode.examples
        )

        winner_counts[episode.winner] += 1
        total_moves += episode.move_count

        print(
            f"Self-play game "
            f"{game_index + 1}/"
            f"{games_per_iteration}: "
            f"winner={episode.winner:+d}, "
            f"moves={episode.move_count}, "
            f"examples={len(episode.examples)}"
        )

    self_play_seconds = (
        time.perf_counter()
        - self_play_started
    )

    batch_size = int(
        training_config["batch_size"]
    )

    if len(replay_buffer) < batch_size:
        raise RuntimeError(
            "Smoke Test 生成的样本不足一个 batch："
            f"{len(replay_buffer)} < {batch_size}"
        )

    training_started = time.perf_counter()

    train_metrics = trainer.train_from_buffer(
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        steps=int(
            training_config[
                "steps_per_iteration"
            ]
        ),
    )

    training_seconds = (
        time.perf_counter()
        - training_started
    )

    checkpoint_dir = Path(
        experiment["checkpoint_dir"]
    )

    checkpoint_path = (
        checkpoint_dir / "smoke.ckpt"
    )

    metrics = train_metrics.as_dict()
    metrics.update(
        {
            "self_play/games": float(
                games_per_iteration
            ),
            "self_play/moves": float(
                total_moves
            ),
            "self_play/examples": float(
                len(replay_buffer)
            ),
            "self_play/black_wins": float(
                winner_counts[1]
            ),
            "self_play/white_wins": float(
                winner_counts[-1]
            ),
            "self_play/draws": float(
                winner_counts[0]
            ),
            "time/self_play_seconds": (
                self_play_seconds
            ),
            "time/training_seconds": (
                training_seconds
            ),
        }
    )

    save_checkpoint(
        checkpoint_path,
        model=model,
        trainer=trainer,
        iteration=1,
        config=config,
        metrics=metrics,
    )

    # 使用固定输入比较加载前后的输出。
    model.eval()

    verification_input = torch.zeros(
        2,
        game.input_channels,
        game.board_size,
        game.board_size,
        dtype=torch.float32,
        device=trainer.device,
    )

    verification_input[:, 2] = 1.0

    with torch.no_grad():
        expected_policy, expected_value = (
            model(verification_input)
        )

    restored_model = build_model(config)

    restored_trainer = build_trainer(
        model=restored_model,
        config=config,
        device=device,
    )

    metadata = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        trainer=restored_trainer,
        # CPU 加载更节省 GPU 峰值显存；
        # Trainer 会将 optimizer state 移回目标设备。
        map_location="cpu",
        restore_rng=True,
    )

    restored_model.eval()

    with torch.no_grad():
        actual_policy, actual_value = (
            restored_model(
                verification_input
            )
        )

    policy_matches = torch.allclose(
        expected_policy,
        actual_policy,
        atol=1e-6,
        rtol=1e-6,
    )

    value_matches = torch.allclose(
        expected_value,
        actual_value,
        atol=1e-6,
        rtol=1e-6,
    )

    if not policy_matches:
        raise RuntimeError(
            "Checkpoint 恢复后 policy 输出不一致"
        )

    if not value_matches:
        raise RuntimeError(
            "Checkpoint 恢复后 value 输出不一致"
        )

    if metadata.iteration != 1:
        raise RuntimeError(
            "Checkpoint iteration 恢复错误"
        )

    if (
        metadata.training_steps
        != trainer.training_steps
    ):
        raise RuntimeError(
            "Checkpoint training_steps 恢复错误"
        )

    total_seconds = (
        time.perf_counter() - started_at
    )

    print()
    print("=== Self-play Summary ===")
    print("Games:", games_per_iteration)
    print("Moves:", total_moves)
    print("Replay size:", len(replay_buffer))
    print("Black wins:", winner_counts[1])
    print("White wins:", winner_counts[-1])
    print("Draws:", winner_counts[0])
    print(
        "Self-play seconds:",
        round(self_play_seconds, 3),
    )

    print()
    print("=== Training Summary ===")

    for name, value in (
        train_metrics.as_dict().items()
    ):
        print(f"{name}: {value:.6f}")

    print(
        "Training seconds:",
        round(training_seconds, 3),
    )

    print()
    print("=== Checkpoint Verification ===")
    print("Path:", checkpoint_path)
    print(
        "Size MB:",
        round(
            checkpoint_path.stat().st_size
            / 1024**2,
            3,
        ),
    )
    print(
        "Restored iteration:",
        metadata.iteration,
    )
    print(
        "Restored training steps:",
        metadata.training_steps,
    )
    print(
        "Policy output matches:",
        policy_matches,
    )
    print(
        "Value output matches:",
        value_matches,
    )
    print(
        "Total seconds:",
        round(total_seconds, 3),
    )
    print("Smoke test passed")

    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the end-to-end AlphaZero "
            "smoke training pipeline."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/smoke.yaml",
        help="Path to YAML config.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device.",
    )

    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override an existing config value. "
            "May be specified multiple times."
        ),
    )

    return parser.parse_args()


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        requested == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable"
        )

    return torch.device(requested)


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    if args.overrides:
        config = apply_overrides(
            config,
            args.overrides,
        )

    device = resolve_device(args.device)

    run_smoke_test(
        config=config,
        device=device,
    )


if __name__ == "__main__":
    main()
