# AI Handoff — Heuristic Gomoku Baseline

## Quick reference

```python
from alphazero.baselines.heuristic import HeuristicPlayer, HeuristicConfig
from alphazero.games.gomoku import GomokuGame

game = GomokuGame(board_size=9, connect=5)
state = game.initial_state()
ai = HeuristicPlayer(HeuristicConfig(search_depth=2))

while game.terminal_value(state) is None:
    action = ai.select_action(game, state)   # returns int
    state = game.next_state(state, action)    # returns new GameState
```

## Key facts

| Item | Reality |
|------|---------|
| Signature | `select_action(game: GomokuGame, state: GameState) -> int` |
| State mutation | Never. `state` and `state.board` are untouched |
| Board access | `state.board` is 2-D readonly int8. Copy for mutable search: `state.board.copy()` |
| Legal actions | `game.legal_actions(state)` returns `[action_size]` float32 mask; use `np.flatnonzero(mask > 0.0)` |
| Win length | `game.connect` (NOT `win_length`) |
| Terminal check | `game.terminal_value(state)` returns `None`/`1.0`/`-1.0`/`0.0` |
| Winner | `game.winner(state)` returns `1`/`-1`/`0` |
| Action↔coord | `row, col = divmod(action, board_size)`; `action = row * board_size + col` |
| Constructor | `GomokuGame(board_size=9, connect=5)` |
| No | `game.play()`, `game.is_terminal()`, `game.board`, `game.current_player`, `game.win_length` |

## Test status

```bash
python -m pytest tests/test_heuristic.py -q   # 10 passed
python -m pytest tests -q                       # 123 passed, 9 skipped, 0 failed
```

## Module layout

- `HeuristicConfig` — frozen dataclass with `validate()`
- `HeuristicPlayer` — public class, only `__init__` + `select_action`
- Module-level helpers (all keyword-only): `_find_immediate_win`, `_is_winning_move`, `_candidate_actions`, `_ordered_actions`, `_move_score`, `_point_pattern_score`, `_line_info`, `_line_score`, `_negamax`, `_evaluate_board`, `_score_player`

## NOT implemented (intentionally)

- No web/JS version yet
- No integration with AlphaZero training or MCTS rollout
- No opening book
- No iterative deepening
- The heuristic is standalone — it does not import from `alphazero.training`, `alphazero.mcts`, or `alphazero.networks`
