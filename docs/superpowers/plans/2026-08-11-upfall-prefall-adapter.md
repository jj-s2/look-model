# UP-Fall Prefall Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 UP-Fall 的 33 关节 CSV 生成按主体可审计的预跌倒时序窗口数据集。

**Architecture:** `risk/upfall_prefall_dataset.py` 负责文件解析、窗口选择和外部数据产物；CLI 仅负责读取参数和调用构建器。输出窗口保留 33 关节，不修改既有 17 关节 GMDCSA24 训练契约。

**Tech Stack:** Python 3.12、NumPy、标准库 CSV/JSON、pytest。

## Global Constraints

- 首次 `LABEL=1` 及之后的帧绝不能出现在预跌倒样本中。
- 所有窗口参数以帧数表示，不推断秒数。
- `subject_id` 必须可用于主体隔离；摄像机不是分组单位。
- 原始数据、生成窗口和权重不提交到 GitHub。

---

### Task 1: 无泄漏窗口构建模块

**Files:**
- Create: `risk/upfall_prefall_dataset.py`
- Create: `tests/risk/test_upfall_prefall_dataset.py`

**Interfaces:**
- Produces: `parse_upfall_identity(path: Path) -> dict[str, str | int]`
- Produces: `build_upfall_prefall_dataset(source_dir: Path, output_dir: Path, pre_frames: int = 10, guard_frames: int = 5) -> dict[str, int]`

- [ ] **Step 1: Write the failing test**

```python
def test_builder_writes_a_guarded_positive_window_before_first_impact(tmp_path):
    summary = build_upfall_prefall_dataset(source, output, pre_frames=4, guard_frames=2)
    assert sample["window_end_frame"] == 6
    assert sample["onset_frame"] == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/risk/test_upfall_prefall_dataset.py -q`

Expected: import failure because the adapter module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def build_upfall_prefall_dataset(source_dir, output_dir, *, pre_frames=10, guard_frames=5):
    # load only finite (T, 33, 3) coordinates; find first LABEL == 1;
    # write positive and earlier negative windows plus a canonical manifest.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/risk/test_upfall_prefall_dataset.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add risk/upfall_prefall_dataset.py tests/risk/test_upfall_prefall_dataset.py
git commit -m "feat: build UP-Fall prefall windows"
```

### Task 2: 可复现 CLI 与真实数据审计

**Files:**
- Create: `scripts/build_upfall_prefall_dataset.py`
- Modify: `tests/risk/test_upfall_prefall_dataset.py`

**Interfaces:**
- Consumes: Task 1 的 `build_upfall_prefall_dataset`。
- Produces: `manifest.jsonl`、`summary.json` 与 `windows/*.npz`。

- [ ] **Step 1: Write the failing test**

```python
def test_cli_records_frame_units_and_exclusion_counts(tmp_path):
    completed = subprocess.run([...], check=True)
    assert json.loads((output / "summary.json").read_text())["window_unit"] == "frames"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/risk/test_upfall_prefall_dataset.py -q`

Expected: CLI script path does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
parser.add_argument("--source-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--pre-frames", type=int, default=10)
parser.add_argument("--guard-frames", type=int, default=5)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/risk/test_upfall_prefall_dataset.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_upfall_prefall_dataset.py tests/risk/test_upfall_prefall_dataset.py
git commit -m "feat: add UP-Fall prefall dataset CLI"
```

### Task 3: 真实数据构建验证

**Files:**
- No repository data changes; artifacts under `F:/datasets/fall_prediction/upfall_3d_skeletons/`.

- [ ] **Step 1: Run the builder**

```bash
python scripts/build_upfall_prefall_dataset.py --source-dir F:/datasets/fall_prediction/upfall_3d_skeletons/raw --output-dir F:/datasets/fall_prediction/upfall_3d_skeletons/prefall_windows --pre-frames 10 --guard-frames 5
```

- [ ] **Step 2: Inspect the audit**

```bash
python -c "import json; print(json.load(open(r'F:/datasets/fall_prediction/upfall_3d_skeletons/prefall_windows/summary.json', encoding='utf-8')))"
```

- [ ] **Step 3: Verify artifact shapes**

```bash
python -c "import glob,numpy as np; p=glob.glob(r'F:/datasets/fall_prediction/upfall_3d_skeletons/prefall_windows/windows/*.npz')[0]; print(np.load(p)['pose'].shape)"
```
