# Progress Log

## 2026-07-30 — Complete Project Audit

### Inspected
All source files in the project: game module, MCTS, networks, self-play, training, baselines (heuristic), utilities, all scripts, all tests, all config files, web deployment, checkpoints, ONNX exports, and git history.

### Commands
```bash
# Environment
pwd                                    # /root/autodl-tmp/Alphazero
git rev-parse --show-toplevel          # /root/autodl-tmp/Alphazero
git status --short
git branch --show-current              # main
git remote -v                          # git@github.com:arce-star/gomoku-alphazero.git
python --version                       # Python 3.10.20
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # 2.5.1+cu121, False
python -c "import onnxruntime; print(onnxruntime.__version__)"  # 1.23.2

# Directory
find . -maxdepth 4 -type f | sort
find alphazero scripts tests -type f -name '*.py' -print | sort

# Tests
python -m pytest tests -q              # 113 passed, 9 skipped, 6 failed
python -m pytest tests/test_heuristic.py -q  # 6 failed (all: TypeError in make_game)

# Checkpoints and models
find checkpoints -maxdepth 3 -type f -printf '%p %s bytes\n' | sort
find . -type f \( -name '*.onnx' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) -printf '%p %s bytes\n' | sort

# ONNX inspection
python -c "import onnx; model = onnx.load('exports/gomoku-9x9-iter21.onnx'); ..."
# Input: states [batch, 3, 9, 9]
# Output: policy_logits [batch, 81], value [batch, 1]
# Opset: 17

# Web project
find arce-star.github.io -maxdepth 4 -type f | sort
```

### Results

**Source files read**: 25 Python files, 3 YAML configs, 1 HTML, 1 JS, 1 CSS, 1 .gitignore, 1 requirements.txt

**Tests**:
- 113 passed: all game, MCTS, network, self-play, training, replay buffer, evaluator, checkpoint, symmetry, seed, and worker tests
- 9 skipped: CUDA-related tests (no GPU available)
- 6 failed: all in `test_heuristic.py` — the heuristic module uses a wrong API

**Key finding: Heuristic API mismatch**:
The `alphazero/baselines/heuristic.py` module was written against an imagined GomokuGame API that doesn't exist:
- Uses `GomokuGame(board_size, win_length=...)` — actual parameter is `connect`
- Uses `game.is_terminal()` — doesn't exist
- Uses `game.legal_actions()` — requires `state` parameter
- Uses `game.play(action)` — doesn't exist; use `game.next_state(state, action)`
- Uses `game.board` — doesn't exist on game; is `state.board`
- Uses `game.current_player` — doesn't exist on game; is `state.current_player`
- Uses `game.win_length` — doesn't exist; is `game.connect`

**Checkpoints**: Multiple training runs exist. Most complete is `baseline_9x9_v2` (30 iterations, best at iter 12, 4.62 MB each).

**ONNX**: 4 exports available (iter 10, 12, 20, 21), all 1.52 MB, opset 17.

**Web**: Deployed at `arce-star.github.io/gomoku.html`. Uses ONNX Runtime Web with WASM backend. AI is policy argmax only — no MCTS.

### Changes
Documentation only. Created `docs/` directory with:
- `PROJECT_CONTEXT.md` — complete project overview
- `API_REFERENCE.md` — detailed API documentation
- `PROGRESS_LOG.md` — this file
- `KNOWN_ISSUES.md` — known issues and bugs

No source code, config, test, or web files were modified.

### Next Actions
1. ~~Fix heuristic module API to match actual GomokuGame~~ ✅ Done 2026-07-30
2. Continue training beyond 30 iterations
3. Add MCTS to web deployment
4. Update requirements.txt with actual dependencies

---

## 2026-07-30 — Fix Heuristic Baseline

### Changes
Rewrote `alphazero/baselines/heuristic.py` and `tests/test_heuristic.py` to use the actual `GomokuGame` + `GameState` API.

### Key changes in heuristic.py
- `select_action(self, game: GomokuGame, state: GameState) -> int` — requires state parameter
- Uses `game.legal_actions(state)` with `np.flatnonzero(legal_mask > 0.0)` for legal actions
- Uses `state.board` (2D readonly int8), copies via `state.board.copy()` for mutable search
- Uses `game.terminal_value(state)` instead of non-existent `game.is_terminal()`
- Uses `game.connect` instead of `game.win_length`
- All internal helpers converted to module-level keyword-only functions using 2D `(row, col)` board access
- `HeuristicConfig.validate()` added with bounds checks

### Key changes in test_heuristic.py
- `make_game(board_size, connect)` uses correct constructor params
- All tests use `state = game.next_state(state, int(action))` (immutable state pattern)
- `game.legal_actions(state)` with proper mask access
- Added tests: state immutability, 5×5/connect=4 variant, performance timing
- Removed tests that depended on non-existent API

### Test results
```bash
python -m pytest tests/test_heuristic.py -q  # 10 passed
python -m pytest tests -q                      # 123 passed, 9 skipped, 0 failed
```

### HeuristicPlayer final API
```python
from alphazero.baselines.heuristic import HeuristicPlayer, HeuristicConfig
from alphazero.games.gomoku import GomokuGame

game = GomokuGame(board_size=9, connect=5)
state = game.initial_state()
ai = HeuristicPlayer(HeuristicConfig(search_depth=2))

action = ai.select_action(game, state)
state = game.next_state(state, action)
```

### Remaining
- Web JS implementation of heuristic for browser fallback mode
- Integrate heuristic into CLI play and Arena evaluation scripts
