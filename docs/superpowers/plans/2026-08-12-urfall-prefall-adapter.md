# UR Fall 冲击前预测接入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** 从 UR Fall 官方 RGB 与同步元数据构建可审计的冲击前姿态窗口，并输出独立外部实验结果。

**Architecture:** 新建 UR Fall 适配器，负责读取无表头同步 CSV、以 \`SV_total\` 峰值选择冲击锚点、验证 RGB 归档的连续帧，并生成泄漏安全的窗口清单。姿态提取器在归档内逐帧读取 PNG，以每帧的单人体关键点与质量状态生成窗口；训练入口复用轻量时序分类器的特征契约，但结果固定标记为外部实验，禁止晋升为部署模型。

**Tech Stack:** Python 3、NumPy、标准库 \`zipfile\`、PyTorch、MediaPipe Tasks/Pose Landmarker（若本机不可用则明确失败且不生成伪特征）。

## Global Constraints

- 仅用 \`fall-XX-cam0-rgb.zip\` 与其官方同步 CSV；不得把加速度作为线上推理特征。
- 正例窗口必须完全满足 \`end_frame <= impact_frame - guard_frames\`。
- 每段序列的正负窗口必须始终在同一个划分中。
- 输出必须保留源文件 SHA-256、锚点依据、窗口边界及排除原因。
- UR Fall 结果标为 \`external_experiment=true\`、\`promoted=false\`，不能替代 GMDCSA24 的正式结论。
- 保留用户已有脏文件；提交前运行 GitNexus \`detect_changes\`，每项任务独立测试并提交。

---

### Task 1: 审计 UR Fall 同步元数据并构建窗口清单

**Files:**
- Create: \`risk/urfall_prefall_dataset.py\`
- Create: \`scripts/build_urfall_prefall_dataset.py\`
- Test: \`tests/risk/test_urfall_prefall_dataset.py\`

**Interfaces:**
- Consumes: \`source_dir: Path\`，其中包含 \`fall-XX-data.csv\`、\`fall-XX-acc.csv\` 与 \`rgb_dir/fall-XX-cam0-rgb.zip\`。
- Produces: \`build_urfall_prefall_manifest(source_dir, rgb_dir, output_dir, pre_frames=10, guard_frames=5) -> dict[str, int | str]\`，写出 \`manifest.jsonl\` 与 \`summary.json\`。

- [ ] **Step 1: Write the failing test**

~~~python
def test_builds_guarded_pair_from_peak_and_records_hashes(tmp_path: Path):
    summary = build_urfall_prefall_manifest(metadata, rgb, out, pre_frames=3, guard_frames=2)
    rows = _read_jsonl(out / "manifest.jsonl")
    assert summary["positive_windows"] == summary["negative_windows"] == 1
    assert rows[0]["window_end_frame"] <= rows[0]["impact_frame"] - 2
    assert len(rows[0]["source_sha256"]) == 64
~~~

- [ ] **Step 2: Run test to verify it fails**

Run: \`python -m pytest tests/risk/test_urfall_prefall_dataset.py::test_builds_guarded_pair_from_peak_and_records_hashes -q\`

Expected: FAIL because \`risk.urfall_prefall_dataset\` does not exist.

- [ ] **Step 3: Write minimal implementation**

~~~python
def build_urfall_prefall_manifest(source_dir: Path, rgb_dir: Path, output_dir: Path, *, pre_frames: int = 10, guard_frames: int = 5) -> dict[str, int | str]:
    # Parse strict three-column frame,timestamp,SV_total rows.
    # Select a unique first maximal SV_total impact frame and verify corresponding
    # 1-based PNG names exist in the zip. Write one preimpact and one early-safe row.
    ...
~~~

- [ ] **Step 4: Extend edge tests**

~~~python
def test_rejects_missing_frames_and_excludes_short_sequences(tmp_path: Path):
    summary = build_urfall_prefall_manifest(metadata, rgb, out, pre_frames=3, guard_frames=2)
    assert summary["excluded_missing_rgb_frame"] == 1
    assert summary["excluded_no_safe_window"] == 1
~~~

- [ ] **Step 5: Verify and commit**

Run: \`python -m pytest tests/risk/test_urfall_prefall_dataset.py -q\`

Run: \`node .gitnexus/run.cjs detect_changes --repo look-model --scope staged\`

~~~bash
git add risk/urfall_prefall_dataset.py scripts/build_urfall_prefall_dataset.py tests/risk/test_urfall_prefall_dataset.py
git commit -m "feat: build audited UR Fall prefall windows"
~~~

### Task 2: 从 RGB 归档提取具质量信息的人体关键点

**Files:**
- Create: \`risk/urfall_pose_extraction.py\`
- Create: \`scripts/extract_urfall_pose_windows.py\`
- Test: \`tests/risk/test_urfall_pose_extraction.py\`

**Interfaces:**
- Consumes: Task 1 的 \`manifest.jsonl\`，每行包含 \`source_zip\`、\`window_start_frame\`、\`window_end_frame\`。
- Produces: \`extract_urfall_pose_windows(manifest_path, rgb_dir, output_dir, detector) -> dict[str, int]\`，每个成功窗口保存 \`(T, 33, 3)\` float32 关键点和可见性；输出 \`pose_manifest.jsonl\`。

- [ ] **Step 1: Write the failing test**

~~~python
def test_extracts_one_pose_window_and_preserves_manifest_lineage(tmp_path: Path):
    summary = extract_urfall_pose_windows(manifest, rgb_dir, out, detector=FakeDetector())
    sample = np.load(out / "windows" / "fall-01-preimpact.npz")
    assert sample["pose"].shape == (3, 33, 3)
    assert summary["extracted_windows"] == 1
~~~

- [ ] **Step 2: Run test to verify it fails**

Run: \`python -m pytest tests/risk/test_urfall_pose_extraction.py::test_extracts_one_pose_window_and_preserves_manifest_lineage -q\`

Expected: FAIL because \`risk.urfall_pose_extraction\` does not exist.

- [ ] **Step 3: Write minimal implementation**

~~~python
def extract_urfall_pose_windows(manifest_path: Path, rgb_dir: Path, output_dir: Path, detector: PoseDetector) -> dict[str, int]:
    # Open each PNG directly from the validated zip; reject missing/corrupt frames,
    # multi-person ambiguity, invalid keypoint shape, or low frame coverage.
    # Never synthesize undetected joints as valid data.
    ...
~~~

- [ ] **Step 4: Cover failure-quality paths**

~~~python
def test_marks_low_coverage_window_excluded_without_writing_npz(tmp_path: Path):
    summary = extract_urfall_pose_windows(manifest, rgb_dir, out, detector=MostlyMissingDetector())
    assert summary["excluded_low_pose_coverage"] == 1
~~~

- [ ] **Step 5: Verify and commit**

Run: \`python -m pytest tests/risk/test_urfall_pose_extraction.py -q\`

Run: \`node .gitnexus/run.cjs detect_changes --repo look-model --scope staged\`

~~~bash
git add risk/urfall_pose_extraction.py scripts/extract_urfall_pose_windows.py tests/risk/test_urfall_pose_extraction.py
git commit -m "feat: extract audited UR Fall pose windows"
~~~

### Task 3: 外部实验训练与报告

**Files:**
- Create: \`risk/urfall_temporal_training.py\`
- Create: \`scripts/train_urfall_prefall_tcn.py\`
- Test: \`tests/risk/test_urfall_temporal_training.py\`
- Modify: \`docs/superpowers/specs/2026-08-12-urfall-prefall-adapter-design.md\`

**Interfaces:**
- Consumes: Task 2 的 \`pose_manifest.jsonl\` 与 \`.npz\` 关键点窗口。
- Produces: \`train_urfall_external_experiment(manifest_path, data_root, output_dir, ...) -> dict[str, object]\`，输出每折预测、指标、模型卡和显式不可晋升声明。

- [ ] **Step 1: Write the failing test**

~~~python
def test_external_training_keeps_sequence_pairs_in_one_fold_and_never_promotes(tmp_path: Path):
    report = train_urfall_external_experiment(manifest, data_root, out, epochs=1, device="cpu")
    assert report["external_experiment"] is True
    assert report["promoted"] is False
    assert report["split_unit"] == "sequence_id"
~~~

- [ ] **Step 2: Run test to verify it fails**

Run: \`python -m pytest tests/risk/test_urfall_temporal_training.py::test_external_training_keeps_sequence_pairs_in_one_fold_and_never_promotes -q\`

Expected: FAIL because \`risk.urfall_temporal_training\` does not exist.

- [ ] **Step 3: Write minimal implementation**

~~~python
def train_urfall_external_experiment(manifest_path: Path, data_root: Path, output_dir: Path, *, epochs: int = 80, device: str = "auto") -> dict[str, object]:
    # Group folds by sequence_id so paired samples cannot leak; reuse only finite
    # pose windows; write metrics and an explicit external-only model card.
    ...
~~~

- [ ] **Step 4: Cover evaluation guard**

~~~python
def test_rejects_sequence_split_that_leaks_positive_negative_pair(tmp_path: Path):
    with pytest.raises(ValueError, match="sequence"):
        train_urfall_external_experiment(bad_manifest, data_root, out, epochs=1, device="cpu")
~~~

- [ ] **Step 5: Run real UR Fall data**

Run: \`python scripts/build_urfall_prefall_dataset.py --source-dir F:/datasets/fall_prediction/urfall/metadata --rgb-dir F:/datasets/fall_prediction/urfall/rgb_cam0_fall --output-dir F:/datasets/fall_prediction/urfall/prefall_windows --pre-frames 10 --guard-frames 5\`

Run: \`python scripts/extract_urfall_pose_windows.py --manifest F:/datasets/fall_prediction/urfall/prefall_windows/manifest.jsonl --rgb-dir F:/datasets/fall_prediction/urfall/rgb_cam0_fall --output-dir F:/datasets/fall_prediction/urfall/pose_windows\`

Run: \`python scripts/train_urfall_prefall_tcn.py --manifest F:/datasets/fall_prediction/urfall/pose_windows/pose_manifest.jsonl --data-root F:/datasets/fall_prediction/urfall/pose_windows --output-dir F:/datasets/fall_prediction/urfall/temporal_tcn --epochs 80 --device cuda\`

- [ ] **Step 6: Verify and commit**

Run: \`python -m pytest tests/risk/test_urfall_temporal_training.py tests/risk/test_urfall_pose_extraction.py tests/risk/test_urfall_prefall_dataset.py -q\`

Run: \`node .gitnexus/run.cjs detect_changes --repo look-model --scope staged\`

~~~bash
git add risk/urfall_temporal_training.py scripts/train_urfall_prefall_tcn.py tests/risk/test_urfall_temporal_training.py docs/superpowers/specs/2026-08-12-urfall-prefall-adapter-design.md
git commit -m "feat: train UR Fall external prefall baseline"
~~~

