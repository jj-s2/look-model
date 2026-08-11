# UP-Fall Temporal Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 训练按主体留出的轻量 TCN，以评估 UP-Fall 的 33 关节预冲击窗口。

**Architecture:** 一个新模块加载适配器的 manifest/NPZ 窗口、执行样本内运动归一化并定义 TCN；训练 CLI 只协调 LOSO、文件写入和模型卡。现有 GMDCSA24/17 关节模型不修改。

**Tech Stack:** Python 3.12、PyTorch 2.4、NumPy、标准库 JSON、pytest。

## Global Constraints

- 训练集与验证集的 subject_id 绝不重叠。
- 判定阈值固定为 0.5，留出主体不得参与阈值选择。
- 输出明确标记为帧窗口的实验基线，promoted=false。
- 原始数据、窗口与权重不提交到 GitHub。

---

### Task 1: 时序数据加载与模型

**Files:**
- Create: risk/upfall_temporal_training.py
- Create: tests/risk/test_upfall_temporal_training.py

**Interfaces:**
- Produces: load_upfall_samples(manifest_path: Path, data_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]
- Produces: UpFallTemporalTCN(joints: int = 33, hidden_dim: int = 32, dropout: float = 0.2)
- Produces: normalize_upfall_pose(pose: object) -> object

- [ ] **Step 1: Write the failing test**

    def test_loader_preserves_33_joint_windows_and_subjects(tmp_path):
        poses, labels, subjects = load_upfall_samples(manifest, root)
        assert poses.shape == (2, 4, 33, 3)
        assert labels.tolist() == [0, 1]

- [ ] **Step 2: Run test to verify it fails**

Run: python -m pytest tests/risk/test_upfall_temporal_training.py -q

Expected: import failure because the temporal training module does not exist.

- [ ] **Step 3: Write minimal implementation**

    class UpFallTemporalTCN(nn.Module):
        def forward(self, pose):
            # normalize input, apply two temporal conv blocks, return one logit per sample

- [ ] **Step 4: Run test to verify it passes**

Run: python -m pytest tests/risk/test_upfall_temporal_training.py -q

Expected: PASS.

- [ ] **Step 5: Commit**

    git add risk/upfall_temporal_training.py tests/risk/test_upfall_temporal_training.py
    git commit -m "feat: add UP-Fall temporal TCN"

### Task 2: 主体留出训练入口

**Files:**
- Create: scripts/train_upfall_prefall_tcn.py
- Modify: tests/risk/test_upfall_temporal_training.py

**Interfaces:**
- Consumes: Task 1 的样本加载器和 TCN。
- Produces: train_upfall_loso(manifest_path, data_root, output_dir, epochs, device) -> dict[str, object]。

- [ ] **Step 1: Write the failing test**

    def test_loso_training_never_promotes_the_experimental_upfall_model(tmp_path):
        report = train_upfall_loso(manifest, root, output, epochs=1, device="cpu")
        assert report["promoted"] is False
        assert report["not_for_clinical_performance"] is True

- [ ] **Step 2: Run test to verify it fails**

Run: python -m pytest tests/risk/test_upfall_temporal_training.py -q

Expected: missing training function.

- [ ] **Step 3: Write minimal implementation**

    for held_out_subject in sorted(set(subjects)):
        # train only indices where subject differs; save predictions for held-out samples

- [ ] **Step 4: Run test to verify it passes**

Run: python -m pytest tests/risk/test_upfall_temporal_training.py -q

Expected: PASS.

- [ ] **Step 5: Commit**

    git add risk/upfall_temporal_training.py scripts/train_upfall_prefall_tcn.py tests/risk/test_upfall_temporal_training.py
    git commit -m "feat: train UP-Fall temporal prefall baseline"

### Task 3: GPU 实验与结果审计

**Files:**
- No repository data changes; artifacts under F:/datasets/fall_prediction/upfall_3d_skeletons/temporal_tcn/.

- [ ] **Step 1: Train the LOSO experiment**

    python scripts/train_upfall_prefall_tcn.py --manifest F:/datasets/fall_prediction/upfall_3d_skeletons/prefall_windows/manifest.jsonl --data-root F:/datasets/fall_prediction/upfall_3d_skeletons/prefall_windows --output-dir F:/datasets/fall_prediction/upfall_3d_skeletons/temporal_tcn --epochs 80 --device cuda

- [ ] **Step 2: Inspect metrics**

    python -c "import json; print(json.load(open(r'F:/datasets/fall_prediction/upfall_3d_skeletons/temporal_tcn/metrics.json', encoding='utf-8')))"

- [ ] **Step 3: Verify experimental provenance**

    python -c "import json; p=json.load(open(r'F:/datasets/fall_prediction/upfall_3d_skeletons/temporal_tcn/model_card.json', encoding='utf-8')); assert p['promoted'] is False and p['window_unit'] == 'frames'"
