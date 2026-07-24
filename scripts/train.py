from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import torch

from alphazero.games.gomoku import GomokuGame
from alphazero.mcts.search import MCTS, MCTSConfig, TorchNetworkEvaluator
from alphazero.networks.residual_net import NetworkConfig, PolicyValueNet
from alphazero.selfplay.episode import SelfPlayConfig, play_episode
from alphazero.selfplay.worker import play_episodes_parallel
from alphazero.training.evaluator import ArenaConfig, evaluate_models
from alphazero.training.replay_buffer import ReplayBuffer
from alphazero.training.trainer import Trainer, TrainerConfig
from alphazero.utils.checkpoint import load_checkpoint, save_checkpoint
from alphazero.utils.config import apply_overrides, load_config
from alphazero.utils.seed import seed_everything


def build_model(config: dict[str, Any]) -> PolicyValueNet:
    game = config["game"]
    network = config["network"]

    return PolicyValueNet(
        NetworkConfig(
            board_size=int(game["board_size"]),
            input_channels=int(network["input_channels"]),
            channels=int(network["channels"]),
            residual_blocks=int(network["residual_blocks"]),
            value_hidden_channels=int(network["value_hidden_channels"]),
            value_hidden_size=int(network["value_hidden_size"]),
        )
    )


def build_trainer(
    model: PolicyValueNet,
    config: dict[str, Any],
    device: torch.device,
) -> Trainer:
    training = config["training"]
    max_grad_norm = training.get("max_grad_norm")

    return Trainer(
        model=model,
        device=device,
        config=TrainerConfig(
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            value_loss_weight=float(training["value_loss_weight"]),
            max_grad_norm=(
                None
                if max_grad_norm is None
                else float(max_grad_norm)
            ),
            use_amp=bool(training["use_amp"]),
        ),
    )


def generate_self_play(
    *,
    game: GomokuGame,
    model: PolicyValueNet,
    replay: ReplayBuffer,
    config: dict[str, Any],
    device: torch.device,
    iteration: int,
) -> dict[str, float]:
    mcts_config = config["mcts"]
    self_play_config = config["self_play"]
    base_seed = int(config["experiment"]["seed"])

    games = int(self_play_config["games_per_iteration"])
    workers = int(self_play_config.get("workers", 1))
    wins = {1: 0, -1: 0, 0: 0}
    total_moves = 0
    total_examples = 0
    started = time.perf_counter()

    search_config = MCTSConfig(
        num_simulations=int(mcts_config["num_simulations"]),
        c_puct=float(mcts_config["c_puct"]),
        dirichlet_alpha=float(mcts_config["dirichlet_alpha"]),
        dirichlet_epsilon=float(mcts_config["dirichlet_epsilon"]),
    )
    episode_config = SelfPlayConfig(
        temperature_moves=int(
            self_play_config["temperature_moves"]
        ),
        sampling_temperature=float(
            self_play_config["sampling_temperature"]
        ),
        add_root_noise=bool(
            self_play_config["add_root_noise"]
        ),
        augment_symmetries=bool(
            self_play_config["augment_symmetries"]
        ),
    )

    if workers > 1:
        episodes = play_episodes_parallel(
            board_size=game.board_size,
            connect=game.connect,
            model=model,
            device=device,
            games=games,
            workers=workers,
            mcts_config=search_config,
            self_play_config=episode_config,
            base_seed=base_seed,
            iteration=iteration,
            inference_batch_size=int(
                self_play_config.get(
                    "inference_batch_size",
                    workers,
                )
            ),
            batch_wait_ms=float(
                self_play_config.get("batch_wait_ms", 2.0)
            ),
        )
    else:
        episodes = []

        for game_index in range(games):
            evaluator = TorchNetworkEvaluator(
                game=game,
                model=model,
                device=device,
            )
            mcts = MCTS(
                game=game,
                evaluator=evaluator,
                config=search_config,
                seed=(
                    base_seed
                    + iteration * 100_000
                    + game_index
                ),
            )
            episodes.append(
                play_episode(
                    game=game,
                    mcts=mcts,
                    config=episode_config,
                )
            )

    for game_index, episode in enumerate(episodes):
        replay.extend(episode.examples)
        wins[episode.winner] += 1
        total_moves += episode.move_count
        total_examples += len(episode.examples)

        print(
            f"  self-play {game_index + 1}/{games}: "
            f"winner={episode.winner:+d}, "
            f"moves={episode.move_count}, "
            f"examples={len(episode.examples)}"
        )

    elapsed = time.perf_counter() - started

    return {
        "self_play/games": float(games),
        "self_play/workers": float(min(workers, games)),
        "self_play/moves": float(total_moves),
        "self_play/examples": float(total_examples),
        "self_play/black_wins": float(wins[1]),
        "self_play/white_wins": float(wins[-1]),
        "self_play/draws": float(wins[0]),
        "self_play/seconds": elapsed,
        "self_play/games_per_second": games / max(elapsed, 1e-9),
    }


def run_training(
    config: dict[str, Any],
    *,
    device: torch.device,
    resume: bool,
) -> None:
    experiment = config["experiment"]
    training_config = config["training"]
    arena_config = config["arena"]

    seed = int(experiment["seed"])
    seed_everything(seed)

    checkpoint_dir = Path(experiment["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_path = checkpoint_dir / "latest.ckpt"
    best_path = checkpoint_dir / "best.ckpt"
    replay_path = checkpoint_dir / "replay.npz"

    game = GomokuGame(
        board_size=int(config["game"]["board_size"]),
        connect=int(config["game"]["connect"]),
    )

    best_model = build_model(config)
    best_trainer = build_trainer(
        best_model,
        config,
        device,
    )

    replay = ReplayBuffer(
        capacity=int(training_config["replay_capacity"]),
        seed=seed,
    )

    start_iteration = 1

    if resume:
        if not latest_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {latest_path}"
            )

        metadata = load_checkpoint(
            latest_path,
            model=best_model,
            trainer=best_trainer,
            map_location="cpu",
            restore_rng=True,
        )

        start_iteration = metadata.iteration + 1

        if replay_path.exists():
            replay = ReplayBuffer.load(
                replay_path,
                seed=seed,
                capacity=int(
                    training_config["replay_capacity"]
                ),
            )

        print(
            f"Resumed from iteration {metadata.iteration}, "
            f"replay size={len(replay)}"
        )

    total_iterations = int(experiment["iterations"])

    print("=== AlphaZero Training ===")
    print("Device:", best_trainer.device)
    print("Board:", f"{game.board_size}x{game.board_size}")
    print("Parameters:", f"{best_model.parameter_count():,}")
    print("Iterations:", total_iterations)

    for iteration in range(
        start_iteration,
        total_iterations + 1,
    ):
        iteration_started = time.perf_counter()
        print(f"\n=== Iteration {iteration}/{total_iterations} ===")

        self_play_metrics = generate_self_play(
            game=game,
            model=best_model,
            replay=replay,
            config=config,
            device=device,
            iteration=iteration,
        )

        batch_size = int(training_config["batch_size"])

        if len(replay) < batch_size:
            raise RuntimeError(
                f"Replay size {len(replay)} is smaller "
                f"than batch size {batch_size}"
            )

        candidate_model = copy.deepcopy(best_model)
        candidate_trainer = build_trainer(
            candidate_model,
            config,
            device,
        )
        # Count optimization steps globally, including rejected candidates.
        candidate_trainer.training_steps = best_trainer.training_steps

        train_metrics = candidate_trainer.train_from_buffer(
            replay_buffer=replay,
            batch_size=batch_size,
            steps=int(
                training_config["steps_per_iteration"]
            ),
        )

        arena = evaluate_models(
            game=game,
            candidate_model=candidate_model,
            best_model=best_model,
            device=device,
            config=ArenaConfig(
                games=int(arena_config["games"]),
                num_simulations=int(
                    arena_config["num_simulations"]
                ),
                c_puct=float(
                    arena_config.get(
                        "c_puct",
                        config["mcts"]["c_puct"],
                    )
                ),
                promotion_threshold=float(
                    arena_config["promotion_threshold"]
                ),
            ),
            seed=seed + iteration * 1_000_000,
        )

        promoted = arena.should_promote

        if promoted:
            best_model = candidate_model
            best_trainer = candidate_trainer
            print("Candidate promoted")
        else:
            # Candidate 被拒绝后，best 保持不变。
            # Keep best weights while preserving the global step count.
            best_trainer.training_steps = candidate_trainer.training_steps
            print("Candidate rejected")

        metrics = {}
        metrics.update(self_play_metrics)
        metrics.update(train_metrics.as_dict())
        metrics.update(arena.as_dict())
        metrics["iteration/seconds"] = (
            time.perf_counter() - iteration_started
        )

        save_checkpoint(
            latest_path,
            model=best_model,
            trainer=best_trainer,
            iteration=iteration,
            config=config,
            metrics=metrics,
        )

        if promoted or not best_path.exists():
            save_checkpoint(
                best_path,
                model=best_model,
                trainer=best_trainer,
                iteration=iteration,
                config=config,
                metrics=metrics,
            )

        replay.save(replay_path)

        print(
            "Arena:",
            f"W={arena.candidate_wins}",
            f"L={arena.best_wins}",
            f"D={arena.draws}",
            f"score={arena.candidate_score:.3f}",
        )
        print("Replay size:", len(replay))
        print(
            "Loss:",
            f"{train_metrics.total_loss:.4f}",
        )
        print(
            "Iteration seconds:",
            f"{metrics['iteration/seconds']:.2f}",
        )

    print("\nTraining completed")
    print("Latest:", latest_path)
    print("Best:", best_path)
    print("Replay:", replay_path)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable"
        )

    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AlphaZero for Gomoku."
    )

    parser.add_argument(
        "--config",
        default="configs/smoke.yaml",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.overrides:
        config = apply_overrides(
            config,
            args.overrides,
        )

    run_training(
        config,
        device=resolve_device(args.device),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
