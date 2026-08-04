# PA-DTSF F1 Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 提升 PA-DTSF 跨受试者跌倒二分类 F1，同时维持 Recall 下限并改善概率校准。

**Architecture:** 将训练数据、增强、受试者内层验证、训练循环、校准和 release 选择拆成独立模块。所有候选只由 subject-2/3/4 的内层验证选择，subject-1 只用于固定确认；最终产物继续兼容 `TorchPhasePredictor` 和实时监测入口。

**Tech Stack:** Python 3.12、PyTorch 2.4.1+cu121、NumPy、scikit-learn、Matplotlib、pytest、RTX 4060 Laptop GPU。

## Global Constraints

- 主选择指标是 subject-macro binary F1，Recall 必须 ≥ 0.750。
- test split 不得用于选择超参数、温度、阈值或训练轮数。
- validation/test 禁止随机增强。
- 当前 release 使用 64 帧 COCO-17 长时骨架，`short_quality=0`。
- 所有随机运行必须记录 seed、split hash、config hash 和 checkpoint SHA256。
- 未通过 promotion gate 的候选不得覆盖当前 release。
- 心理健康输出不作诊断，本计划不制造心理健康准确率。
- 每个生产代码改动必须先有失败测试，再实现，再运行绿色测试。

---

### Task 1: 受试者感知的数据集与骨架增强

**Files:**
- Create: `risk/phase_model/training_data.py`
- Test: `tests/risk/phase_model/test_training_data.py`

**Interfaces:**
- Produces: `PhaseSample`, `PhasePoseDataset`, `augment_pose(pose, rng, config)`, `subject_folds(clips, subjects)`。
- Consumes: `normalize_pose_array()`、dataset lock clip records 和 `.npz` pose cache。

- [ ] **Step 1: 写失败测试，锁定分组与增强行为**

```python
import numpy as np
from risk.phase_model.training_data import AugmentationConfig, augment_pose, subject_folds


def test_subject_folds_never_overlap_subjects():
    clips = [
        {"clip_id": "a", "subject_id": "subject-2"},
        {"clip_id": "b", "subject_id": "subject-3"},
        {"clip_id": "c", "subject_id": "subject-4"},
    ]
    folds = subject_folds(clips, ("subject-2", "subject-3", "subject-4"))
    assert len(folds) == 3
    for fold in folds:
        assert set(fold.train_subjects).isdisjoint(fold.validation_subjects)


def test_augmentation_is_seeded_and_preserves_shape():
    pose = np.ones((64, 17, 3), dtype=np.float32)
    config = AugmentationConfig(flip_probability=1.0, noise_std=0.0, keypoint_dropout=0.0, frame_mask_probability=0.0)
    first = augment_pose(pose, np.random.default_rng(42), config)
    second = augment_pose(pose, np.random.default_rng(42), config)
    assert first.shape == (64, 17, 3)
    np.testing.assert_array_equal(first, second)
```

- [ ] **Step 2: 运行测试并确认因缺少模块失败**

Run: `python -m pytest tests/risk/phase_model/test_training_data.py -q`

Expected: collection fails with `ModuleNotFoundError: risk.phase_model.training_data`。

- [ ] **Step 3: 实现数据契约、fold 与确定性增强**

```python
@dataclass(frozen=True)
class AugmentationConfig:
    flip_probability: float = 0.5
    noise_std: float = 0.01
    keypoint_dropout: float = 0.05
    frame_mask_probability: float = 0.20
    time_scale_min: float = 0.90
    time_scale_max: float = 1.10


@dataclass(frozen=True)
class SubjectFold:
    train_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]


def subject_folds(clips, subjects):
    present = {str(item["subject_id"]) for item in clips}
    selected = tuple(subject for subject in subjects if subject in present)
    return tuple(SubjectFold(tuple(item for item in selected if item != held_out), (held_out,)) for held_out in selected)
```

实现 COCO 左右索引映射 `(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)`；遮挡时同步把 xy 与 confidence 置零。`PhasePoseDataset.__getitem__` 必须先加载 raw pose，再在 train 模式增强，最后调用 `normalize_pose_array()`。

- [ ] **Step 4: 运行数据模块测试**

Run: `python -m pytest tests/risk/phase_model/test_training_data.py tests/risk/phase_model/test_normalization.py -q`

Expected: all tests pass。

- [ ] **Step 5: 提交 Task 1**

```powershell
git add risk/phase_model/training_data.py tests/risk/phase_model/test_training_data.py
git commit -m "feat: add subject-aware pose training data"
```

---

### Task 2: 可配置正则化模型与平滑 fall loss

**Files:**
- Modify: `risk/phase_model/model.py`
- Modify: `risk/phase_model/losses.py`
- Test: `tests/risk/phase_model/test_model.py`
- Test: `tests/risk/phase_model/test_losses.py`

**Interfaces:**
- Produces: `PhaseAwareFusionModel(*, short_dim: int = 512, joints: int = 17, hidden_dim: int = 128, dropout: float = 0.2)`。
- Produces: `compute_multitask_loss(outputs, targets, supervision_mask, *, fall_label_smoothing: float = 0.0)`。
- Preserves: current checkpoint loading when dropout is omitted。

- [ ] **Step 1: 写失败测试**

```python
def test_fall_label_smoothing_reduces_extreme_target():
    outputs = _outputs(fall_logit=torch.tensor([0.0]))
    targets = LossTargets(_phase(), torch.tensor([1.0]), _masked(), _masked())
    raw = compute_multitask_loss(outputs, targets, {"fall_event"}, fall_label_smoothing=0.0)
    smooth = compute_multitask_loss(outputs, targets, {"fall_event"}, fall_label_smoothing=0.05)
    assert torch.isfinite(raw.total)
    assert torch.isfinite(smooth.total)


def test_model_accepts_configurable_dropout():
    model = PhaseAwareFusionModel(short_dim=512, joints=17, hidden_dim=64, dropout=0.35)
    assert model.long_branch.network[3].p == 0.35
```

- [ ] **Step 2: 运行测试并确认新参数缺失**

Run: `python -m pytest tests/risk/phase_model/test_model.py tests/risk/phase_model/test_losses.py -q`

Expected: failures mention unexpected `dropout` and `fall_label_smoothing` arguments。

- [ ] **Step 3: 实现最小正则化接口**

```python
class PhaseAwareFusionModel:
    def __init__(self, *, short_dim=512, joints=17, hidden_dim=128, dropout=0.2):
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.long_branch = LongBranchTCN(joints=joints, hidden_dim=hidden_dim, dropout=dropout)


def compute_multitask_loss(outputs, targets, supervision_mask, *, fall_label_smoothing=0.0):
    if not 0.0 <= fall_label_smoothing < 0.5:
        raise ValueError("fall_label_smoothing must be in [0, 0.5)")
    labels = targets.fall_event.float()
    labels = labels * (1.0 - 2.0 * fall_label_smoothing) + fall_label_smoothing
```

- [ ] **Step 4: 运行模型与损失测试**

Run: `python -m pytest tests/risk/phase_model/test_model.py tests/risk/phase_model/test_losses.py tests/risk/phase_model/test_torch_predictor.py -q`

Expected: all tests pass，旧 checkpoint 预测器仍可加载。

- [ ] **Step 5: 提交 Task 2**

```powershell
git add risk/phase_model/model.py risk/phase_model/losses.py tests/risk/phase_model/test_model.py tests/risk/phase_model/test_losses.py
git commit -m "feat: regularize phase model training"
```

---

### Task 3: 指标、温度校准与验证阈值选择

**Files:**
- Create: `risk/phase_model/selection.py`
- Test: `tests/risk/phase_model/test_selection.py`

**Interfaces:**
- Produces: `BinaryMetrics`、`metrics_at_threshold(labels, scores, threshold)`。
- Produces: `choose_threshold(records, recall_floor=0.75)`。
- Produces: `fit_temperature(logits, labels)` 和 `apply_temperature(logits, temperature)`。
- Produces: `candidate_passes(candidate, baseline)`。

- [ ] **Step 1: 写失败测试，禁止低 Recall 阈值获选**

```python
from risk.phase_model.selection import choose_threshold, fit_temperature


def test_threshold_selection_enforces_recall_floor():
    records = [
        {"subject_id": "s2", "label": 1, "score": 0.70},
        {"subject_id": "s2", "label": 1, "score": 0.40},
        {"subject_id": "s3", "label": 0, "score": 0.60},
        {"subject_id": "s3", "label": 0, "score": 0.10},
    ]
    selected = choose_threshold(records, recall_floor=1.0, lower=0.20, upper=0.80, step=0.01)
    assert selected.recall == 1.0
    assert selected.threshold <= 0.40


def test_temperature_is_positive_and_finite():
    result = fit_temperature([5.0, -5.0, 2.0, -2.0], [1, 0, 0, 1])
    assert result > 0.0
    assert math.isfinite(result)
```

- [ ] **Step 2: 运行测试并确认模块不存在**

Run: `python -m pytest tests/risk/phase_model/test_selection.py -q`

Expected: collection fails with `ModuleNotFoundError`。

- [ ] **Step 3: 实现 subject-macro F1、temperature 和 promotion gate**

```python
@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    macro_f1: float
    precision: float
    recall: float


def apply_temperature(logits, temperature):
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    return [1.0 / (1.0 + math.exp(-float(value) / temperature)) for value in logits]
```

`choose_threshold` 必须先过滤 Recall 低于 floor 的候选，再按 `(subject_macro_f1, worst_subject_f1, -false_positive_rate)` 降序选择。`fit_temperature` 使用 LBFGS 优化单一 `log_temperature`，最终返回 `exp(log_temperature)`。

- [ ] **Step 4: 运行选择模块与现有 evaluation 测试**

Run: `python -m pytest tests/risk/phase_model/test_selection.py tests/risk/phase_model/test_evaluation.py -q`

Expected: all tests pass。

- [ ] **Step 5: 提交 Task 3**

```powershell
git add risk/phase_model/selection.py tests/risk/phase_model/test_selection.py
git commit -m "feat: add subject-safe model selection"
```

---

### Task 4: 内层三折训练、早停与有限候选搜索

**Files:**
- Create: `configs/skeleton/padtfs_f1_v2.py`
- Create: `scripts/optimize_phase_model.py`
- Test: `tests/integration/test_optimize_phase_model.py`

**Interfaces:**
- Consumes: `PhasePoseDataset`、`subject_folds()`、`choose_threshold()`、`fit_temperature()`。
- Produces: `run_candidate(config, fold, seed, paths) -> FoldResult`。
- Produces: `run_search(search_space, paths) -> SelectionSummary`。

- [ ] **Step 1: 写失败测试，确保 test IDs 不进入候选选择**

```python
def test_search_never_passes_test_partition_to_fold_trainer(tmp_path):
    seen = []
    def fake_train(candidate, train_ids, validation_ids, output_dir):
        seen.extend(train_ids)
        seen.extend(validation_ids)
        return _fold_result(f1=0.8, recall=0.8)
    summary = run_search(_fixture_paths(tmp_path), candidates=[_candidate()], fold_trainer=fake_train)
    assert set(seen).isdisjoint(summary.test_ids)


def test_search_rejects_candidate_below_recall_floor(tmp_path):
    summary = run_search(_fixture_paths(tmp_path), candidates=[_candidate()], fold_trainer=lambda *args: _fold_result(f1=0.9, recall=0.6))
    assert summary.selected is None
    assert "recall_floor" in summary.rejections[0]
```

- [ ] **Step 2: 运行集成测试并确认入口不存在**

Run: `python -m pytest tests/integration/test_optimize_phase_model.py -q`

Expected: import failure for `scripts.optimize_phase_model`。

- [ ] **Step 3: 实现可注入的候选搜索与真实 GPU trainer**

配置文件必须固定：

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

trainer 每个 epoch 必须计算 validation logits 和固定 0.5 F1/AP；按 `(F1, AP)` 保存最佳 state dict。DataLoader 的 generator 使用候选 seed，CUDA 与 NumPy 同步设种子。写出 `fold_metrics.json`、`validation_predictions.jsonl` 和 `checkpoint.pt`。

- [ ] **Step 4: 运行离线集成测试和单候选 GPU smoke**

Run: `python -m pytest tests/integration/test_optimize_phase_model.py -q`

Run:

```powershell
python scripts/optimize_phase_model.py `
  --dataset-lock outputs/releases/padtfs-gmdcsa24-gpu-norm/dataset_lock.json `
  --split-manifest outputs/releases/padtfs-gmdcsa24-gpu-norm/split_manifest.json `
  --data-root datasets/raw/training/GMDCSA24 `
  --output outputs/optimization/padtfs-f1-v2-smoke `
  --max-candidates 1 --max-epochs 2
```

Expected: exit 0，输出三个内层 fold，且 summary 中没有 test 指标。

- [ ] **Step 5: 提交 Task 4**

```powershell
git add configs/skeleton/padtfs_f1_v2.py scripts/optimize_phase_model.py tests/integration/test_optimize_phase_model.py
git commit -m "feat: add leakage-safe F1 optimization"
```

---

### Task 5: 最终训练、校准与 release 打包

**Files:**
- Modify: `scripts/optimize_phase_model.py`
- Modify: `risk/phase_model/release.py`
- Modify: `risk/phase_model/torch_predictor.py`
- Test: `tests/integration/test_phase_release.py`
- Test: `tests/risk/phase_model/test_torch_predictor.py`

**Interfaces:**
- Checkpoint payload adds `calibration: {temperature, threshold, source}`。
- `TorchPhasePredictor.temperature` defaults to 1.0 for old checkpoints。
- `TorchPhasePredictor.fall_threshold` defaults to 0.5 for old checkpoints。

- [ ] **Step 1: 写失败测试，锁定兼容与校准**

```python
def test_predictor_applies_checkpoint_temperature(checkpoint_factory, dual_window):
    checkpoint = checkpoint_factory(calibration={"temperature": 2.0, "threshold": 0.42, "source": "inner_validation"})
    predictor = TorchPhasePredictor(checkpoint, device="cpu")
    output = predictor.predict(dual_window)
    assert predictor.temperature == 2.0
    assert predictor.fall_threshold == 0.42
    assert 0.0 <= output.fall_event_prob <= 1.0


def test_old_checkpoint_uses_safe_calibration_defaults(old_checkpoint):
    predictor = TorchPhasePredictor(old_checkpoint, device="cpu")
    assert predictor.temperature == 1.0
    assert predictor.fall_threshold == 0.5
```

- [ ] **Step 2: 运行测试并确认 calibration 属性缺失**

Run: `python -m pytest tests/risk/phase_model/test_torch_predictor.py tests/integration/test_phase_release.py -q`

Expected: failures mention missing `temperature` and `fall_threshold`。

- [ ] **Step 3: 实现 release metadata 和 predictor 温度缩放**

```python
calibration = payload.get("calibration", {})
self.temperature = float(calibration.get("temperature", 1.0))
self.fall_threshold = float(calibration.get("threshold", 0.5))
if self.temperature <= 0 or not 0.0 < self.fall_threshold < 1.0:
    raise ValueError("invalid checkpoint calibration")

fall_prob = float(torch.sigmoid(output.fall_event_logit / self.temperature)[0].detach().cpu())
```

最终训练使用 subject-2/3/4，epoch 取三折 best epoch 中位数。只在候选确定并打包完成后运行 subject-1 确认评估。release 必须包含 inner-fold summary、temperature、threshold、test confirmation、checksums 和 `promoted` 判定。

- [ ] **Step 4: 运行兼容性与 release 测试**

Run: `python -m pytest tests/risk/phase_model/test_torch_predictor.py tests/integration/test_phase_release.py tests/pipeline/test_phase_risk_service.py -q`

Expected: all tests pass，旧 release 仍能推理。

- [ ] **Step 5: 提交 Task 5**

```powershell
git add scripts/optimize_phase_model.py risk/phase_model/release.py risk/phase_model/torch_predictor.py tests/integration/test_phase_release.py tests/risk/phase_model/test_torch_predictor.py
git commit -m "feat: package calibrated phase releases"
```

---

### Task 6: 运行完整优化实验并执行 promotion gate

**Files:**
- Generate: `outputs/optimization/padtfs-f1-v2/selection_summary.json`
- Generate: `outputs/releases/padtfs-f1-v2/checkpoint.pt`
- Generate: `outputs/releases/padtfs-f1-v2/metrics.json`
- Generate: `outputs/releases/padtfs-f1-v2/model_card.md`

**Interfaces:**
- Consumes: Task 4 搜索脚本与 Task 5 release 打包。
- Produces: 冻结、可校验、不会覆盖旧 release 的 candidate release。

- [ ] **Step 1: 运行 18 候选内层三折搜索**

```powershell
python scripts/optimize_phase_model.py `
  --dataset-lock outputs/releases/padtfs-gmdcsa24-gpu-norm/dataset_lock.json `
  --split-manifest outputs/releases/padtfs-gmdcsa24-gpu-norm/split_manifest.json `
  --data-root datasets/raw/training/GMDCSA24 `
  --output outputs/optimization/padtfs-f1-v2 `
  --device cuda
```

Expected: `selection_summary.json` 包含 18 个或更少候选、三折指标、拒绝理由和一个满足 Recall floor 的 selected candidate。

- [ ] **Step 2: 仅对选定配置运行 seed 42/43/44**

```powershell
python scripts/optimize_phase_model.py `
  --resume outputs/optimization/padtfs-f1-v2/selection_summary.json `
  --final-seeds 42 43 44 `
  --output outputs/optimization/padtfs-f1-v2 `
  --device cuda
```

Expected: `ensemble_summary.json` 明确 single 与 ensemble 的 mean F1 delta；delta < 0.02 时选择 single。

- [ ] **Step 3: 打包候选并运行固定确认集**

```powershell
python scripts/optimize_phase_model.py `
  --finalize outputs/optimization/padtfs-f1-v2/selection_summary.json `
  --release-id padtfs-f1-v2 `
  --release-output outputs/releases/padtfs-f1-v2 `
  --device cuda
```

Expected: test confirmation 只出现一次，metrics 同时包含 baseline delta 和 promotion gate。

- [ ] **Step 4: 检查 promotion gate**

仅当 inner mean F1 提升 ≥0.03、inner Recall ≥0.750、confirmation F1 ≥0.800、ECE ≤0.150 时，允许 `promoted=true`。否则保留 `promoted=false` 和具体失败原因。

- [ ] **Step 5: 提交可发布小文件与经过批准的 checkpoint**

```powershell
git add outputs/releases/padtfs-f1-v2/metrics.json outputs/releases/padtfs-f1-v2/model_card.md outputs/releases/padtfs-f1-v2/checksums.sha256
git add -f outputs/releases/padtfs-f1-v2/checkpoint.pt
git commit -m "release: publish F1-optimized phase checkpoint"
```

---

### Task 7: 消融图、报告、全量验证与推送

**Files:**
- Modify: `scripts/plot_phase_results.py`
- Create: `docs/figures/padtfs-f1-v2/ablation_f1.png`
- Create: `docs/figures/padtfs-f1-v2/subject_folds.png`
- Create: `docs/figures/padtfs-f1-v2/report.md`
- Modify: `README.md`
- Test: `tests/integration/test_phase_plots.py`

**Interfaces:**
- Consumes: E0–E4 selection summaries and final predictions。
- Produces: paired baseline/candidate figures and evidence report。

- [ ] **Step 1: 写失败测试，要求图表包含 baseline delta**

```python
def test_comparison_artifacts_include_ablation_and_subject_folds(tmp_path):
    paths = generate_comparison_artifacts(_baseline(), _candidate(), _folds(), tmp_path)
    assert paths["ablation"].name == "ablation_f1.png"
    assert paths["subject_folds"].name == "subject_folds.png"
    assert "baseline_delta" in json.loads(paths["metrics"].read_text(encoding="utf-8"))
```

- [ ] **Step 2: 运行测试并确认函数不存在**

Run: `python -m pytest tests/integration/test_phase_plots.py -q`

Expected: import or attribute failure for `generate_comparison_artifacts`。

- [ ] **Step 3: 实现 paired 图表与中文证据报告**

图表必须同时展示 E0–E4 mean F1、标准差、worst-subject F1、Recall 和 ECE；报告必须包含 4 个受试者、确认集已知、样本量小和非临床限制。

- [ ] **Step 4: 运行目标测试、全量测试和差异检查**

```powershell
python -m pytest tests/integration/test_phase_plots.py tests/integration/test_optimize_phase_model.py tests/risk/phase_model -q
python -m pytest -q
git diff --check
```

Expected: all tests pass，`git diff --check` 无错误。

- [ ] **Step 5: 提交文档图表并推送当前分支**

```powershell
git add scripts/plot_phase_results.py tests/integration/test_phase_plots.py docs/figures/padtfs-f1-v2 README.md
git commit -m "docs: publish F1 optimization evidence"
git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 push
```

Expected: `origin/codex/elderly-monitoring` 指向本地 HEAD，草稿 PR 自动包含新提交。
