from __future__ import annotations

import multiprocessing as mp
import time
from queue import Empty
from typing import Sequence

import numpy as np
import torch
from torch import nn

from alphazero.games.base import GameState
from alphazero.games.gomoku import GomokuGame
from alphazero.mcts.search import MCTS, MCTSConfig
from alphazero.selfplay.episode import EpisodeResult, SelfPlayConfig, play_episode


class QueueEvaluator:
    """Worker-side synchronous evaluator backed by the main GPU process."""

    def __init__(
        self,
        game: GomokuGame,
        worker_id: int,
        request_queue,
        response_queue,
    ) -> None:
        self.game = game
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.request_id = 0

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        request_id = self.request_id
        self.request_id += 1

        encoded = self.game.encode_state(state)
        self.request_queue.put(
            (self.worker_id, request_id, encoded),
        )

        response_id, logits, value, error = self.response_queue.get()

        if response_id != request_id:
            raise RuntimeError("Inference response ID mismatch")
        if error is not None:
            raise RuntimeError(error)

        return np.asarray(logits, dtype=np.float64), float(value)


def _worker_main(
    worker_id: int,
    game_indices: Sequence[int],
    board_size: int,
    connect: int,
    mcts_config: MCTSConfig,
    self_play_config: SelfPlayConfig,
    base_seed: int,
    iteration: int,
    request_queue,
    response_queue,
    result_queue,
) -> None:
    try:
        game = GomokuGame(board_size=board_size, connect=connect)
        evaluator = QueueEvaluator(
            game,
            worker_id,
            request_queue,
            response_queue,
        )

        for game_index in game_indices:
            mcts = MCTS(
                game=game,
                evaluator=evaluator,
                config=mcts_config,
                seed=base_seed + iteration * 100_000 + game_index,
            )
            episode = play_episode(
                game=game,
                mcts=mcts,
                config=self_play_config,
            )
            result_queue.put(("result", game_index, episode))

        result_queue.put(("done", worker_id, None))
    except BaseException as exc:
        result_queue.put(
            ("error", worker_id, f"{type(exc).__name__}: {exc}")
        )


@torch.no_grad()
def play_episodes_parallel(
    *,
    board_size: int,
    connect: int,
    model: nn.Module,
    device: torch.device | str,
    games: int,
    workers: int,
    mcts_config: MCTSConfig,
    self_play_config: SelfPlayConfig,
    base_seed: int,
    iteration: int,
    inference_batch_size: int = 32,
    batch_wait_ms: float = 2.0,
) -> list[EpisodeResult]:
    if games <= 0:
        raise ValueError("games must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")

    worker_count = min(workers, games)
    context = mp.get_context("spawn")

    request_queue = context.Queue(maxsize=worker_count * 2)
    result_queue = context.Queue()
    response_queues = [
        context.Queue(maxsize=2)
        for _ in range(worker_count)
    ]

    assignments = [
        list(range(worker_id, games, worker_count))
        for worker_id in range(worker_count)
    ]

    processes = [
        context.Process(
            target=_worker_main,
            args=(
                worker_id,
                assignments[worker_id],
                board_size,
                connect,
                mcts_config,
                self_play_config,
                base_seed,
                iteration,
                request_queue,
                response_queues[worker_id],
                result_queue,
            ),
            name=f"selfplay-{worker_id}",
        )
        for worker_id in range(worker_count)
    ]

    device = torch.device(device)
    model.to(device)
    was_training = model.training
    model.eval()

    episodes: dict[int, EpisodeResult] = {}
    finished_workers = 0

    for process in processes:
        process.start()

    try:
        while len(episodes) < games:
            while True:
                try:
                    kind, index, payload = result_queue.get_nowait()
                except Empty:
                    break

                if kind == "result":
                    episodes[index] = payload
                elif kind == "done":
                    finished_workers += 1
                elif kind == "error":
                    raise RuntimeError(
                        f"Self-play worker {index} failed: {payload}"
                    )

            if len(episodes) >= games:
                break

            try:
                first = request_queue.get(timeout=0.05)
            except Empty:
                if finished_workers == worker_count:
                    raise RuntimeError("Workers ended before all games completed")
                continue

            requests = [first]
            deadline = time.perf_counter() + batch_wait_ms / 1000.0

            while len(requests) < inference_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    requests.append(request_queue.get(timeout=remaining))
                except Empty:
                    break

            states = np.stack(
                [request[2] for request in requests],
                axis=0,
            )
            tensor = torch.from_numpy(states).to(
                device=device,
                dtype=torch.float32,
            )
            logits, values = model(tensor)
            logits = logits.float().cpu().numpy()
            values = values[:, 0].float().cpu().numpy()

            for batch_index, (worker_id, request_id, _) in enumerate(requests):
                response_queues[worker_id].put(
                    (
                        request_id,
                        logits[batch_index],
                        float(np.clip(values[batch_index], -1.0, 1.0)),
                        None,
                    )
                )
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()

        if was_training:
            model.train()

        request_queue.close()
        result_queue.close()
        for queue in response_queues:
            queue.close()

    return [episodes[index] for index in range(games)]
