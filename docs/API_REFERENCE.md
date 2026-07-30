# API Reference

## alphazero.games.base

### `class GameState` (frozen dataclass, slots=True)

Represents a board position.

**Fields**:
- `board: np.ndarray` — shape `(board_size, board_size)`, dtype `int8`, read-only. 0=empty, 1=black, -1=white
- `current_player: int` — 1 (black) or -1 (white)
- `move_count: int` — number of stones placed so far
- `last_action: Optional[int]` — last action taken, None for initial state

**Notes**:
- Board is copied and made read-only in `__post_init__`
- `board.flags.writeable` is `False`

### `class Game(ABC)`

Abstract base class for AlphaZero games.

**Abstract Properties**:
- `board_size: int` — board side length
- `action_size: int` — total number of possible actions (board_size²)
- `input_channels: int` — number of input channels for neural network

**Abstract Methods**:
- `initial_state() -> GameState` — create initial board position
- `legal_actions(state: GameState) -> np.ndarray` — returns `[action_size]` float32 mask, 1.0=legal, 0.0=illegal
- `next_state(state: GameState, action: int) -> GameState` — returns new state after executing action (does not mutate original)
- `winner(state: GameState) -> int` — returns 1 (black wins), -1 (white wins), 0 (no winner)
- `terminal_value(state: GameState) -> Optional[float]` — from `state.current_player`'s perspective: 1.0=win, -1.0=loss, 0.0=draw, None=ongoing
- `encode_state(state: GameState) -> np.ndarray` — returns `[input_channels, board_size, board_size]` float32

---

## alphazero.games.gomoku

### `class GomokuGame(Game)`

No-forbidden-move Gomoku implementation.

**Constructor**:
```python
GomokuGame(board_size: int = 9, connect: int = 5) -> None
```

**Parameters**:
- `board_size: int` — board side length (default 9)
- `connect: int` — number of consecutive stones needed to win (default 5)

**Raises**: `ValueError` if `board_size <= 0`, `connect <= 1`, or `connect > board_size`

**Instance Attributes**:
- `_board_size: int` — stored board size
- `connect: int` — win length (public, non-private)

**Properties**:
- `board_size -> int` — returns `self._board_size`
- `action_size -> int` — returns `board_size * board_size`
- `input_channels -> int` — always returns 3

**Methods**:

```python
def initial_state(self) -> GameState
```
Creates empty board with `current_player=1` (black), `move_count=0`, `last_action=None`.

```python
def action_to_coord(self, action: int) -> Tuple[int, int]
```
Converts action index to `(row, col)`. `action = row * board_size + col`.
Raises `TypeError` if action is not integer, `ValueError` if out of range.

```python
def coord_to_action(self, row: int, col: int) -> int
```
Converts `(row, col)` to action index.
Raises `TypeError` if row/col are not integers, `ValueError` if out of range.

```python
def legal_actions(self, state: GameState) -> np.ndarray
```
Returns `[action_size]` float32 mask. If state is terminal, returns all zeros. Requires state parameter — NOT a no-argument method.

```python
def next_state(self, state: GameState, action: int) -> GameState
```
Returns new state with stone placed. Does NOT mutate original state. Flips `current_player` to `-current_player`. Raises `ValueError` if game is terminal or position is occupied.

```python
def winner(self, state: GameState) -> int
```
Scans entire board for 5+ consecutive stones. Returns 1 (black), -1 (white), or 0 (no winner).

```python
def terminal_value(self, state: GameState) -> Optional[float]
```
Returns value from `state.current_player`'s perspective. After `next_state()` switches player, the player who just won gets `-1.0` (loss for the now-current player).

```python
def encode_state(self, state: GameState) -> np.ndarray
```
Returns `[3, board_size, board_size]` float32:
- Channel 0: current player's stones
- Channel 1: opponent's stones
- Channel 2: all 1.0 if black to move, all 0.0 if white to move

```python
def render(self, state: GameState) -> str
```
Returns terminal-displayable board string. X=black, O=white, .=empty.

**Important**: There is NO `is_terminal()` method, NO `play()` method, NO `.board` attribute, NO `.current_player` attribute, NO `.win_length` attribute. All game state is accessed through `GameState` objects, not the `GomokuGame` instance.

---

## alphazero.games.symmetry

### `apply_symmetry()`

```python
def apply_symmetry(
    encoded_state: np.ndarray,    # [channels, board_size, board_size]
    policy: np.ndarray,            # [action_size]
    board_size: int,
    rotation: int,                 # 0, 1, 2, 3 (90° CCW rotations)
    flip: bool,                    # horizontal flip after rotation
) -> tuple[np.ndarray, np.ndarray]  # (transformed_state, transformed_policy)
```
Returns C-contiguous float32 copies (safe for `torch.from_numpy`).

### `generate_symmetries()`

```python
def generate_symmetries(
    encoded_state: np.ndarray,
    policy: np.ndarray,
    board_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]
```
Returns 8 samples: 4 rotations × (flip=False, flip=True).

---

## alphazero.mcts.node

### `class MCTSNode` (dataclass)

**Fields**:
- `prior: float` — prior probability P(s, a) from network
- `visit_count: int = 0`
- `value_sum: float = 0.0`
- `children: Dict[int, MCTSNode] = field(default_factory=dict)`

**Properties**:
- `is_expanded: bool` — True if children dict is non-empty
- `q_value: float` — `value_sum / visit_count` (0.0 if unvisited), from THIS node's current player perspective

**Methods**:
```python
def expand(self, action_priors: dict[int, float]) -> None
```
Creates child nodes from legal actions and their priors. Does NOT overwrite existing children.

```python
def update(self, value: float) -> None
```
Increments `visit_count` and adds value to `value_sum`. Value must be from this node's current player perspective.

```python
def child_visit_counts(self, action_size: int) -> np.ndarray
```
Returns `[action_size]` float32 array of child visit counts.

```python
def child_priors(self, action_size: int) -> np.ndarray
```
Returns `[action_size]` float32 array of child priors.

**Value convention**: `child.q_value` is from the child's player perspective. When parent uses child for PUCT, it uses `-child.q_value`.

---

## alphazero.mcts.search

### `class PositionEvaluator` (Protocol)

```python
def evaluate(self, state: GameState) -> tuple[np.ndarray, float]
```
Returns `(policy_logits: [action_size], value: float)`. Value is from state's current player perspective, range typically [-1, 1].

### `class TorchNetworkEvaluator`

```python
TorchNetworkEvaluator(game: Game, model: nn.Module, device: torch.device | str)
```
Wraps a PyTorch model as a PositionEvaluator. Clips value to [-1, 1]. Returns policy_logits as float64 numpy array, value as Python float.

**Call sites**: smoke_test.py, train.py (single-worker), play_cli.py, evaluator.py

### `class MCTSConfig` (frozen dataclass)

**Fields**:
- `num_simulations: int = 64`
- `c_puct: float = 1.5`
- `dirichlet_alpha: float = 0.3`
- `dirichlet_epsilon: float = 0.25`
- `initial_q: float = 0.0`

**Method**: `validate()` — raises ValueError on invalid values.

### `class SearchResult` (dataclass)

**Fields**:
- `action: Optional[int]` — None if terminal state
- `visit_policy: np.ndarray` — `[action_size]` float32, sums to 1.0
- `root_value: float`
- `root: MCTSNode`

### `class MCTS`

```python
MCTS(
    game: Game,
    evaluator: PositionEvaluator,
    config: MCTSConfig | None = None,
    seed: Optional[int] = None,
) -> None
```

**Method**:
```python
def search(
    self,
    state: GameState,
    *,
    add_root_noise: bool = False,
    temperature: float = 1.0,
) -> SearchResult
```

**Parameters**:
- `add_root_noise`: True for self-play (adds Dirichlet noise), False for Arena/play
- `temperature`: 0 = argmax, >0 = sample proportional to `visit_count^(1/temperature)`

**Internal flow**:
1. If terminal → return immediately (no network evaluation)
2. Expand root: `_expand_and_evaluate()` → masked-softmax over legal actions → create children
3. If `add_root_noise`: `_add_dirichlet_noise()`
4. Run `num_simulations` times: select → expand/evaluate leaf → backpropagate
5. Build visit policy from root child visit counts
6. Sample action

**Internal methods** (static/private):
- `_select_child(node) -> (action, child)` — PUCT: `score = -child.q_value + c_puct * child.prior * sqrt(parent_visits) / (1 + child.visit_count)`
- `_expand_and_evaluate(node, state) -> float` — network evaluation + masked softmax + node expand
- `_masked_softmax(logits, legal_mask) -> np.ndarray` — numerically stable softmax over legal actions only
- `_add_dirichlet_noise(root)` — adds Dirichlet noise to root's children priors
- `_backpropagate(path, leaf_value)` — static method; flips value sign at each level
- `_visit_policy(root, temperature) -> np.ndarray` — builds policy from visit counts
- `_sample_action(visit_policy, temperature) -> Optional[int]`

---

## alphazero.networks.residual_net

### `class NetworkConfig` (frozen dataclass)

**Fields**:
- `board_size: int = 9`
- `input_channels: int = 3`
- `channels: int = 64`
- `residual_blocks: int = 5`
- `value_hidden_channels: int = 32`
- `value_hidden_size: int = 64`

**Property**: `action_size -> int` (board_size²)
**Method**: `validate()`

### `class ConvBlock(nn.Module)`

```python
ConvBlock(in_channels: int, out_channels: int, kernel_size: int = 3)
```
Conv2d (bias=False) + BatchNorm2d + ReLU(inplace).

### `class ResidualBlock(nn.Module)`

```python
ResidualBlock(channels: int)
```
Two Conv2d+BatchNorm layers with skip connection and ReLU after addition.

### `class PolicyHead(nn.Module)`

```python
PolicyHead(in_channels: int, board_size: int)
```
Conv2d(in→1, 1×1) → flatten → logits. No softmax.

### `class ValueHead(nn.Module)`

```python
ValueHead(in_channels: int, hidden_channels: int, hidden_size: int)
```
Conv2d(in→hidden, 1×1) + BN + ReLU → AdaptiveAvgPool2d(1) → FC(hidden→hidden_size) + ReLU → FC(hidden_size→1) → tanh.

### `class PolicyValueNet(nn.Module)`

```python
PolicyValueNet(config: NetworkConfig | None = None) -> None
```

**Properties**: `action_size: int`

**Methods**:
```python
def forward(self, states: Tensor) -> tuple[Tensor, Tensor]
```
- Input: `[batch, 3, board_size, board_size]` float32
- Output: `(policy_logits: [batch, action_size], value: [batch, 1])`
- Policy logits are raw (no softmax), value is in [-1, 1]

```python
@torch.no_grad()
def predict(self, states: Tensor) -> tuple[Tensor, Tensor]
```
Inference helper. Returns `(policy_probs (softmaxed), value)`. Does NOT mask illegal actions (MCTS handles that). Restores training mode after.

```python
def parameter_count(self) -> int
```
Returns total trainable parameter count.

**Weight initialization**:
- Conv2d: Kaiming normal (fan_out)
- BatchNorm2d: weight=1, bias=0
- Linear: Xavier uniform
- Policy head final conv: N(0, 0.01) — smaller init to avoid over-confident random policy
- Value head final FC: N(0, 0.01)

---

## alphazero.selfplay.episode

### `class SelfPlayConfig` (frozen dataclass)

**Fields**:
- `temperature_moves: int = 10` — first N moves use sampling temperature
- `sampling_temperature: float = 1.0`
- `add_root_noise: bool = True`
- `augment_symmetries: bool = True`
- `max_moves: Optional[int] = None` — safety limit (default: action_size)

### `class TrainingExample` (frozen dataclass)

**Fields**:
- `state: np.ndarray` — `[channels, board_size, board_size]` float32, read-only, C-contiguous
- `policy: np.ndarray` — `[action_size]` float32, sums to 1.0, read-only, C-contiguous
- `value: float` — in [-1.0, 1.0]

### `class EpisodeResult` (frozen dataclass)

**Fields**:
- `examples: list[TrainingExample]`
- `winner: int` — 1 (black), -1 (white), 0 (draw)
- `move_count: int`
- `final_state: GameState`
- `actions: tuple[int, ...]` — sequence of actions played

**Properties**: `black_won`, `white_won`, `is_draw`

### `play_episode()`

```python
def play_episode(
    game: Game,
    mcts: MCTS,
    config: SelfPlayConfig | None = None,
) -> EpisodeResult
```

**Flow**:
1. Start from `game.initial_state()`
2. For each move:
   - Check terminal
   - Determine temperature (sampling for first N moves, 0 after)
   - Run `mcts.search(state, add_root_noise, temperature)`
   - Validate: action is not None, legal, policy sums to 1, no probability on illegal moves
   - Save `(encoded_state, visit_policy, current_player)` as pending example
   - Apply action via `game.next_state()`
3. After game ends: convert pending examples to TrainingExamples with value = +1 (player won), -1 (lost), 0 (draw)
4. Optionally augment each example with 8 symmetries

**Validates**: mcts.game is the same object as game parameter.

---

## alphazero.selfplay.worker

### `class QueueEvaluator`

Worker-side evaluator that sends encoded states to the main process for batched GPU inference.

```python
QueueEvaluator(game: GomokuGame, worker_id: int, request_queue, response_queue)
```

**Method**: `evaluate(state: GameState) -> tuple[np.ndarray, float]` — blocks until main process returns result.

### `play_episodes_parallel()`

```python
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
) -> list[EpisodeResult]
```

Spawns worker processes (using `mp.get_context("spawn")`), each running `_worker_main`. Main process collects encoded states, batches them, runs GPU inference, sends results back. Workers are distributed round-robin across games.

---

## alphazero.selfplay.records

### `write_episode_records()`

```python
def write_episode_records(
    path: str | Path,
    episodes: Sequence[EpisodeResult],
    *,
    iteration: int,
    board_size: int,
    connect: int,
) -> Path
```
Writes JSONL file with one JSON object per game. Atomic via temp file + os.replace().

### `read_episode_record()`

```python
def read_episode_record(
    path: str | Path,
    game_number: int,  # 1-based
) -> dict
```
Reads a single game record from a JSONL file.

---

## alphazero.training.replay_buffer

### `class ReplayBatch` (frozen dataclass)

**Fields**: `states: np.ndarray` ([batch, channels, H, W]), `policies: np.ndarray` ([batch, action_size]), `values: np.ndarray` ([batch, 1])

**Method**: `to_torch(device, non_blocking=False) -> TorchReplayBatch`

### `class TorchReplayBatch` (frozen dataclass)

**Fields**: `states: Tensor`, `policies: Tensor`, `values: Tensor`

### `class ReplayBuffer`

```python
ReplayBuffer(capacity: int, seed: Optional[int] = None)
```

**Properties**: `capacity`, `is_empty`, `is_full`, `state_shape`, `policy_shape`, `total_added`

**Methods**:
- `add(example: TrainingExample) -> None` — add one example; validates shape consistency
- `extend(examples: Iterable[TrainingExample]) -> int` — returns count added
- `sample(batch_size: int, *, replace: bool = False) -> ReplayBatch` — uniform random sampling
- `sample_torch(batch_size, device, *, replace=False, non_blocking=False) -> TorchReplayBatch`
- `examples() -> tuple[TrainingExample, ...]` — snapshot from oldest to newest
- `save(path: str | Path) -> None` — atomic save as compressed NPZ
- `load(path, *, seed=None, capacity=None) -> ReplayBuffer` — class method; restores from NPZ
- `clear() -> None` — clears examples and shapes, preserves total_added

**Shape validation**: First added example sets expected state/policy shapes; subsequent additions must match.

---

## alphazero.training.trainer

### `class TrainerConfig` (frozen dataclass)

**Fields**: `learning_rate=1e-3`, `weight_decay=1e-4`, `value_loss_weight=1.0`, `max_grad_norm=5.0`, `use_amp=True`

### `class TrainMetrics` (frozen dataclass)

Fields: `total_loss`, `policy_loss`, `value_loss`, `policy_entropy`, `predicted_value_mean`, `target_value_mean`, `grad_norm`, `learning_rate`, `batch_size`

### `class AveragedTrainMetrics` (frozen dataclass)

Same fields as TrainMetrics plus `steps: int`. All values are averages.

### `class Trainer`

```python
Trainer(
    model: nn.Module,
    device: torch.device | str,
    config: TrainerConfig | None = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
)
```

**Key attributes**: `model`, `device`, `config`, `optimizer`, `grad_scaler` (GradScaler), `use_amp`, `training_steps`

**Methods**:
```python
@staticmethod
def policy_loss(policy_logits: Tensor, target_policy: Tensor) -> Tensor
```
Cross-entropy with soft targets (not one-hot). Returns scalar loss.

```python
def train_step(self, batch: TorchReplayBatch) -> TrainMetrics
```
Single optimization step: forward → loss → backward (with AMP) → grad clip → optimizer step.
Batch must be on the trainer's device.

```python
def train_from_buffer(
    self, replay_buffer: ReplayBuffer, batch_size: int, steps: int,
    *, replace: bool = False,
) -> AveragedTrainMetrics
```
Samples `steps` batches and trains. Returns averaged metrics.

```python
def optimizer_state_dict(self) -> dict
def scaler_state_dict(self) -> dict
def load_optimizer_state_dict(self, state_dict: dict) -> None
def load_scaler_state_dict(self, state_dict: dict) -> None
```

**AMP details**: `init_scale=1024` (lower than default 65536), `growth_interval=2000`.

---

## alphazero.training.evaluator

### `class ArenaConfig` (frozen dataclass)

**Fields**: `games=50`, `num_simulations=64`, `c_puct=1.5`, `promotion_threshold=0.55`, `max_moves=None`, `opening_moves=4`

### `class ArenaGameResult` (frozen dataclass)

**Fields**: `winner`, `candidate_player`, `move_count`, `final_state`
**Property**: `candidate_result` — 1 (candidate win), 0 (draw), -1 (candidate loss)

### `class ArenaResult` (frozen dataclass)

**Fields**: `candidate_wins`, `best_wins`, `draws`, `games`, `promotion_threshold`, `game_results`
**Properties**: `candidate_score` (includes 0.5 for draws), `candidate_decisive_win_rate`, `should_promote`

### `play_arena_game()`

```python
def play_arena_game(
    *, game, candidate_evaluator, best_evaluator, candidate_player: int,
    config: ArenaConfig, seed=None, initial_state=None,
) -> ArenaGameResult
```
Deterministic (temperature=0, no noise). Uses `initial_state` if provided (for paired openings).

### `evaluate_agents()`

```python
def evaluate_agents(
    *, game, candidate_evaluator, best_evaluator, config: ArenaConfig, seed=None,
) -> ArenaResult
```
Runs `config.games` games. Generates a random opening every 2 games, with colors swapped.

### `evaluate_models()`

```python
def evaluate_models(
    *, game, candidate_model, best_model, device, config: ArenaConfig, seed=None,
) -> ArenaResult
```
Convenience wrapper: creates TorchNetworkEvaluators for both models.

---

## alphazero.utils.config

- `load_config(path) -> dict` — loads and validates YAML config
- `validate_config(config)` — checks required sections and value constraints
- `get_config_value(config, dotted_key)` — nested dict access
- `set_config_value(config, dotted_key, value)` — nested dict set (key must exist)
- `apply_overrides(config, overrides: list[str]) -> dict` — applies CLI key=value overrides

**Required config sections**: experiment, game, network, mcts, self_play, training, arena
**Fixed constraint**: `network.input_channels` must be 3 (Gomoku encoder requirement)

---

## alphazero.utils.checkpoint

### `save_checkpoint()`

```python
def save_checkpoint(
    path: str | Path,
    *, model, trainer, iteration: int, config, metrics=None,
) -> Path
```
Saves complete training state (model, optimizer, scaler, config, metrics, iteration, training_steps, RNG state). Atomic via temp file + os.replace.

### `load_checkpoint()`

```python
def load_checkpoint(
    path: str | Path,
    *, model, trainer=None, map_location="cpu", restore_rng=True, strict=True,
) -> CheckpointMetadata
```
Returns `CheckpointMetadata(iteration, training_steps, config, metrics, path)`. If trainer is provided, restores optimizer, scaler, and training_steps. Validates trainer.model is same object as model.

### `class CheckpointMetadata` (frozen dataclass)

**Fields**: `iteration: int`, `training_steps: int`, `config: dict`, `metrics: dict`, `path: Path`

---

## alphazero.utils.seed

- `seed_everything(seed, *, deterministic=False)` — seeds Python random, numpy, PyTorch (CPU and CUDA), sets cudnn flags
- `capture_rng_state() -> dict` — captures all RNG states for checkpointing
- `restore_rng_state(state: dict)` — restores RNG states (moves CUDA RNG tensors to CPU)

---

## alphazero.baselines.heuristic

### `class HeuristicConfig` (frozen dataclass)

```python
HeuristicConfig(
    candidate_radius: int = 2,
    candidate_limit: int = 12,
    search_depth: int = 2,
    center_bonus: int = 8,
)
```

**Constraints** (enforced by `validate()`):
- `candidate_radius >= 1`
- `candidate_limit >= 1`
- `search_depth >= 1`

### `class HeuristicPlayer`

```python
HeuristicPlayer(config: HeuristicConfig | None = None)
```

A deterministic, non-neural Gomoku baseline. Uses pattern-based position evaluation and negamax with alpha-beta pruning. Never calls a neural network, ONNX, or MCTS.

**Public method**:

```python
def select_action(self, game: GomokuGame, state: GameState) -> int:
```

**Parameters**:
- `game: GomokuGame` — the game rules object (provides `board_size`, `connect`, `legal_actions()`, `terminal_value()`)
- `state: GameState` — the current board position

**Returns**: Python `int` — the chosen action (`row * board_size + col`).

**Raises**: `ValueError` if the state is terminal (match string contains "terminal").

**Guarantees**:
- Does NOT modify the input `state` or `state.board`
- Returned action is always in the legal action mask

**Algorithm flow**:
1. Empty board → returns center point
2. Checks for immediate winning move (one move to connect)
3. Checks for opponent's immediate winning move and blocks it
4. Generates candidate actions within Chebyshev distance `candidate_radius` of existing stones
5. Orders candidates by combined offensive + defensive pattern scores
6. Runs `search_depth`-ply negamax with alpha-beta pruning over top `candidate_limit` candidates
7. Returns action with best negamax score

**Internal design**:
- Search works on a mutable 2D `board.copy()` (`np.int8`), never creates `GameState` per node
- Win detection uses `game.connect` (supports non-5 connect rules, e.g., 5×5/4-connect)
- All helper functions (`_find_immediate_win`, `_negamax`, `_evaluate_board`, etc.) are module-level keyword-only functions
- Pattern scores are defined in `PATTERN_SCORES` dict

**Usage example**:
```python
from alphazero.baselines.heuristic import HeuristicPlayer, HeuristicConfig
from alphazero.games.gomoku import GomokuGame

game = GomokuGame(board_size=9, connect=5)
state = game.initial_state()
ai = HeuristicPlayer(HeuristicConfig(search_depth=2))

while game.terminal_value(state) is None:
    action = ai.select_action(game, state)
    state = game.next_state(state, action)
```

**Tests**: `tests/test_heuristic.py` — 10 tests, all passing. Covers center opening, immediate win/block, legal action, terminal rejection, state immutability, 5×5/connect=4, self-play game, performance, config validation.

---

## Scripts

### `scripts/train.py`

```bash
python -m scripts.train --config configs/train.yaml [--device auto|cpu|cuda] [--resume] [--set key=value ...]
```
Main training loop: self-play → train → arena → promote/reject → checkpoint. Default config: `configs/smoke.yaml`.

### `scripts/smoke_test.py`

```bash
python -m scripts.smoke_test --config configs/smoke.yaml [--device auto|cpu|cuda] [--set key=value ...]
```
End-to-end test: self-play → training → checkpoint → load verification.

### `scripts/export_onnx.py`

```bash
python -m scripts.export_onnx --checkpoint <path> --output <path>
```
Exports a checkpoint to ONNX format (opset 17, dynamic batch).

### `scripts/play_cli.py`

```bash
python -m scripts.play_cli --checkpoint <path> [--device cpu|cuda] [--simulations 100] [--human black|white|random]
```
CLI human-vs-AI play using MCTS.

### `scripts/replay_game.py`

```bash
python -m scripts.replay_game --file <jsonl> [--game 1] [--delay 0.5]
```
Interactive replay of self-play game records.

### `scripts/benchmark_selfplay.py`

```bash
python -m scripts.benchmark_selfplay --workers N [--games 8] [--simulations 32] [--device cuda]
```
Benchmarks parallel self-play throughput.

---

## Tests

| File | Test Count | Status |
|------|-----------|--------|
| `test_game.py` | 17 | All pass |
| `test_mcts.py` | 11 | All pass (1 skipped: CUDA) |
| `test_network.py` | 12 | All pass (1 skipped: CUDA) |
| `test_selfplay.py` | 7 | All pass |
| `test_trainer.py` | 9 | All pass (2 skipped: CUDA) |
| `test_replay_buffer.py` | 18 | All pass (1 skipped: CUDA) |
| `test_evaluator.py` | 9 | All pass (1 skipped: CUDA) |
| `test_checkpoint.py` | 9 | All pass (1 skipped: CUDA) |
| `test_symmetry.py` | 6 | All pass |
| `test_heuristic.py` | 6 | **ALL FAIL** (API mismatch) |
| `test_seed.py` | — | All pass (2 skipped) |
| `test_worker.py` | — | All pass (3 skipped) |
| **Total** | **128** | **113 pass, 9 skip, 6 fail** |
