# Known Issues

---

## Medium

### CUDA not available in current environment

**现象**: `torch.cuda.is_available()` 返回 `False`，虽然 PyTorch 是 CUDA 版本 (2.5.1+cu121)。

**影响**: 无法使用 GPU 训练或推理，所有 CUDA 测试被跳过。

**复现命令**:
```bash
python -c "import torch; print(torch.cuda.is_available())"  # False
```

**当前判断**: 环境没有 GPU 驱动或 GPU 未被分配（可能是 AutoDL 实例的当前状态）。

**建议处理方式**: 检查 AutoDL 实例配置，确认 GPU 是否已分配。如果不需要 GPU 训练，当前 CPU-only 模式可以工作但会很慢。

**相关文件**: `alphazero/utils/seed.py:26`, `alphazero/training/trainer.py:138`

---

### requirements.txt incomplete

**现象**: `requirements.txt` 只包含 `numpy`, `PyYAML`, `tqdm`, `tensorboard`, `pytest`, `psutil`，但缺少 `torch`, `onnx`, `onnxruntime` 等核心依赖。

**影响**: 新环境无法通过 `pip install -r requirements.txt` 安装完整依赖。

**复现命令**: 查看 `requirements.txt` 内容。

**当前判断**: 项目实际依赖远多于列出的包。

**建议处理方式**: 补充完整的依赖列表，包括版本号。

**相关文件**: `requirements.txt:1-6`

---

### Web AI uses policy argmax without MCTS

**现象**: 网页 `gomoku.js` 中的 `chooseAiAction()` 只选择 `policy_logits` 最大的合法动作，不使用 MCTS 搜索和价值网络。

**影响**: 网页 AI 水平远低于训练端，且与训练时的决策过程不匹配（训练使用 200 次 MCTS 模拟，网页只用单次前向传播）。

**复现**: 查看 `arce-star.github.io/assets/js/gomoku.js:165-188`。

**当前判断**: 可能是因为 ONNX Runtime Web WASM 后端速度不够快，无法在浏览器中实时运行 MCTS。这是一个有意为之的简化。

**建议处理方式**:
1. 短期：在网页说明中注明 AI 使用的是简化版推理
2. 长期：实现浏览器端 MCTS（纯 JS，不需要 ONNX Runtime 即可运行），或用 WebGL/WebGPU 加速推理

**相关文件**: `arce-star.github.io/assets/js/gomoku.js:165-188`

---

### ONNX Runtime Web uses WASM backend only

**现象**: 网页使用 `executionProviders: ["wasm"]`，不使用 WebGL 或 WebGPU。

**影响**: 推理速度较慢。WASM 比 WebGL 慢数倍。

**复现**: 查看 `arce-star.github.io/assets/js/gomoku.js:277`。

**当前判断**: 可能因为 WebGL/WebGPU 支持需要额外的 ONNX Runtime Web 构建或后端文件。

**建议处理方式**: 添加 WebGL 后端支持（需包含对应的 ORT Web 文件），并在可用时优先使用。

**相关文件**: `arce-star.github.io/assets/js/gomoku.js:277`

---

## Low

### Empty script files

**现象**: `scripts/selfplay.py` 和 `scripts/evaluate.py` 是空文件。

**影响**: 无功能影响，但可能在 IDE 中引起困惑。

**建议处理方式**: 删除或实现。

**相关文件**: `scripts/selfplay.py`, `scripts/evaluate.py`

---

### PyTorch AMP init_scale lowered from default

**现象**: `Trainer.__init__` 将 `GradScaler` 的 `init_scale` 从默认 65536 降为 1024。

**影响**: 对小型网络的训练稳定性有帮助，但可能在大型网络训练时导致不必要的 scale 增长步骤。

**相关文件**: `alphazero/training/trainer.py:178`

---

### tic_tac_toe.py is empty

**现象**: `alphazero/games/tic_tac_toe.py` 是空文件（有 `__init__.py` 但无实现）。

**影响**: 无。可能是未来扩展的占位符。

**相关文件**: `alphazero/games/tic_tac_toe.py`

---

### Multiple stale checkpoints from different training runs

**现象**: `checkpoints/` 下有多个训练运行的残留，包括 `gomoku_9x9`, `parallel_9x9`, `integration_9x9`, `baseline_9x9`, `baseline_9x9_v2`, `smoke`。

**影响**: 占用 ~30 MB 磁盘空间。不紧急但建议清理。

**当前判断**: `baseline_9x9_v2` 是最完整的运行（30 次迭代）。

**建议处理方式**: 归档旧运行，只保留最相关的一个用于恢复/导出。

**相关文件**: `checkpoints/` 下所有 `.ckpt` 和 `.npz` 文件

---

## Pending Verification

1. **训练收敛性**: 现有 best model (iter 12) 的实力尚未在正式 Arena 评估中与基线做对比。iter 12 模型是否已经足够强大并不清楚。

2. **Web ONNX Runtime Web 兼容性**: iter 21 的 ONNX 模型在浏览器中加载是否正常、推理结果是否与 PyTorch 一致，尚未做端到端验证。

3. **并行自对弈稳定性**: `play_episodes_parallel` 在极端条件下（大量 worker、长时间运行）的稳定性待确认。

4. **FP16 训练稳定性**: AMP 在 9×9 小网络上的行为已在代码中处理（降低 init_scale），但长时间训练中是否会出现梯度下溢尚待观察。

## Resolved Issues

### Heuristic module API mismatch (resolved 2026-07-30)

**Was**: `alphazero/baselines/heuristic.py` used non-existent `GomokuGame` API (`win_length`, `is_terminal()`, `play()`, `game.board`, 1D board access, etc.). All 6 tests failed.

**Fix**: Rewrote `heuristic.py` to use the actual API:
- `select_action(self, game: GomokuGame, state: GameState) -> int`
- Uses `state.board` (2D readonly int8), copies for mutable search
- Uses `game.legal_actions(state)` returning float32 mask, converted via `np.flatnonzero`
- Uses `game.terminal_value(state)`, `game.connect`
- All module-level helpers use keyword-only 2D `(row, col)` board access
- `HeuristicConfig.validate()` added
- Internal search works on mutable `board.copy()` without creating GameState per node

**Tests**: 10 tests, all pass. Covers: center opening, immediate win, immediate block, legal action validation, terminal rejection, state immutability, 5×5/connect=4 variant, full self-play game, performance, config validation.

**Related files**: `alphazero/baselines/heuristic.py`, `tests/test_heuristic.py`
