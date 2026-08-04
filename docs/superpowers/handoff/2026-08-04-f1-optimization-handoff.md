# PA-DTSF 跌倒风险算法优化交接说明

> 给后续 AI 使用。请先阅读本文件，再继续代码；确认集结果不能表述为完全未观察过的测试集结果。

## 1. 目标与硬约束

提升跨受试者跌倒二分类 F1，同时保持召回率下限并改善概率校准。本计划只优化跌倒风险；心理健康输出只能是非诊断性的风险提示，不得制造或宣称心理健康准确率。

- 主选择指标：subject-macro binary F1。
- 内层 Recall ≥ 0.750；每个受试者 Recall ≥ 0.600。
- 确认集不能用于选择超参数、温度、阈值或训练轮数。
- validation/confirmation 禁止随机增强。
- 当前 release 固定使用 64 帧 COCO-17 long branch，`short_quality=0`。
- 记录 seed、split hash、config hash、checkpoint SHA256。
- 未通过 promotion gate 的候选不得覆盖旧 release。

## 2. 工作区与当前提交

- 工作区：`F:\邵吉锦\look model\.worktrees\elderly-monitoring`
- 分支：`codex/elderly-monitoring`
- 远程：`https://github.com/jj-s2/look-model.git`
- PR：[jj-s2/look-model#1](https://github.com/jj-s2/look-model/pull/1)
- 推送：`git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 push`

已提交：

| 提交 | 内容 | 状态 |
|---|---|---|
| `3498e54` | 优化设计与实施计划 | 已审查 |
| `9e61b7e` | Task 1：64 帧、连续 2–6 帧遮挡、short 禁用、身体尺度噪声 | Task 1 审查通过 |
| `5ff5cd0` | Task 2：可配置 Dropout 与 fall 标签平滑 | Task 2 审查通过 |
| `52b63e0` | Task 3：指标、阈值、温度校准、promotion gate | 已实现，任务级审查待完成 |

`5db1b2f` 是 Task 1 的旧实现，已被 `9e61b7e` 修复提交覆盖。工作树中有用户已有未跟踪文件，禁止使用 `git add .`，不要删除或覆盖这些文件。

## 3. 已知基线

此前完整测试为 `401 passed`。旧模型评估：

| 分区 | n | 正例 | Precision | Recall | F1 | ROC-AUC | AP | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 91 | 44 | 0.881 | 0.841 | 0.860 | 0.935 | 0.933 | 0.107 |
| validation | 37 | 15 | 0.579 | 0.733 | 0.647 | 0.794 | 0.763 | 0.300 |
| confirmation（原 test） | 32 | 16 | 0.800 | 0.750 | 0.774 | 0.836 | 0.890 | 0.215 |

原 test 已被查看，后续统一称“固定确认集”，不要称 pristine test。

## 4. 接手第一步：审查 Task 3

```powershell
git status -sb
git log --oneline -6
python -m pytest tests/risk/phase_model/test_selection.py tests/risk/phase_model/test_evaluation.py -q
```

审查 `52b63e0` 的 `risk/phase_model/selection.py` 和对应测试，确认：

- `metrics_at_threshold` 正确计算 TP/FP/FN/TN、Precision、Recall、F1、FPR；
- `choose_threshold` 先满足 Recall floor，再按 subject-macro F1、worst-subject F1、低 FPR 排序；
- `fit_temperature` 只接收内层验证 logits/labels，并用 LBFGS 优化 `log_temperature`；
- `apply_temperature` 拒绝非有限或非正温度；
- `candidate_passes` 缺证据时返回 False，并执行 F1、Recall、确认集 F1、ECE、Brier gate；
- 确认集不能进入阈值或温度拟合。

审查结果写入 `.superpowers/sdd/2026-08-04-f1-optimization/progress.md`，再做 Task 4。

## 5. Task 4：内层三折搜索

创建 `configs/skeleton/padtfs_f1_v2.py`、`scripts/optimize_phase_model.py`、`tests/integration/test_optimize_phase_model.py`。

固定配置：

```python
SEEDS = (42, 43, 44)
HIDDEN_DIMS = (64, 96, 128)
DROPOUTS = (0.25, 0.35, 0.45)
LEARNING_RATES = (1e-4, 3e-4)
BATCH_SIZE = 8
MAX_EPOCHS = 60
PATIENCE = 8
WEIGHT_DECAY = 1e-3
GRADIENT_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.05
RECALL_FLOOR = 0.75
MAX_CANDIDATES = 18
```

只用 subject-2/3/4 做内层三折；subject-1 固定确认集只能在候选确定后使用。必须有真实 DataLoader、shuffle、训练增强、验证无增强、早停，并输出 fold metrics、validation predictions、checkpoint 和 seed/split/config/hash 元数据。测试要证明确认集 ID 不会进入候选选择。

## 6. Task 5：校准与 release

修改 `scripts/optimize_phase_model.py`、`risk/phase_model/release.py`、`risk/phase_model/torch_predictor.py`。checkpoint 增加：

```json
{"calibration": {"temperature": 1.0, "threshold": 0.5, "source": "inner_validation"}}
```

旧 checkpoint 缺少 calibration 时必须使用 `temperature=1.0`、`fall_threshold=0.5`。最终 epoch 取三折 best epoch 中位数；确认集只评估一次。release 必须包含 inner-fold summary、校准参数、confirmation metrics、checksums 和 `promoted` 判定。

## 7. Task 6：运行实验与 gate

先检查输入真实存在：

```text
outputs/releases/padtfs-gmdcsa24-gpu-norm/dataset_lock.json
outputs/releases/padtfs-gmdcsa24-gpu-norm/split_manifest.json
datasets/raw/training/GMDCSA24
```

运行 18 个以内候选的 GPU 搜索：

```powershell
python scripts/optimize_phase_model.py `
  --dataset-lock outputs/releases/padtfs-gmdcsa24-gpu-norm/dataset_lock.json `
  --split-manifest outputs/releases/padtfs-gmdcsa24-gpu-norm/split_manifest.json `
  --data-root datasets/raw/training/GMDCSA24 `
  --output outputs/optimization/padtfs-f1-v2 `
  --device cuda
```

GPU/数据不足时只做 `--max-candidates 1 --max-epochs 2` smoke，不能把 smoke 指标当最终结果。选定配置再跑 seed 42/43/44；ensemble 仅在 inner mean F1 比 single 提升至少 0.02 时保留。

promotion gate：inner mean F1 相对旧基线提升 ≥0.03；inner Recall ≥0.750；确认集 F1 ≥0.800、Recall ≥0.750；ECE ≤0.150；Brier ≤0.215。失败必须写 `promoted=false` 和具体原因，并保留旧 release。

## 8. Task 7：图表、报告、全量验证

修改 `scripts/plot_phase_results.py`、`tests/integration/test_phase_plots.py`、`README.md`；生成：

- `docs/figures/padtfs-f1-v2/ablation_f1.png`
- `docs/figures/padtfs-f1-v2/subject_folds.png`
- `docs/figures/padtfs-f1-v2/report.md`

报告展示 E0–E4 mean F1、标准差、worst-subject F1、Recall、ECE、baseline delta，并写明只有 4 个受试者、确认集已观察、样本量有限、结果非临床结论。

```powershell
python -m pytest tests/integration/test_phase_plots.py tests/integration/test_optimize_phase_model.py tests/risk/phase_model -q
python -m pytest -q
git diff --check
```

## 9. 每个任务的执行规则

1. 先写失败测试并记录 RED 输出，再写最小生产代码。
2. 运行 GREEN 测试，把完整输出写入 `.superpowers/sdd/2026-08-04-f1-optimization/task-N-report.md`。
3. 每个任务只提交计划范围内文件，生成 review-package，完成任务级规格/质量审查；Critical/Important finding 必须修复并复审。
4. 在 `progress.md` 记录完成和 deferred minor；最后做 whole-branch review，再按 finishing-a-development-branch 流程交付。

禁止：伪造 GPU 指标、在确认集上调参、把心理健康提示写成诊断、发送设备远程控制指令、删除用户已有输出。

只有 Task 3–7 全部完成、测试与审查通过、promotion gate 有证据、远端 HEAD 与本地一致时，才能声称优化完成。若资源不足，应交付可复现 smoke 和明确阻塞原因。
