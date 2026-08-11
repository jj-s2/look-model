# 预跌倒真实窗口重训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于真实视频总时长和跌倒开始标注构建无泄漏的 GMDCSA24 预跌倒训练数据，并完成按受试者隔离训练。

**Architecture:** 新的数据构建模块从统一标注、视频元数据与姿态缓存生成数值 CSV；已有 `train_prefall_model.py` 继续负责模型训练和发布门槛。窗口构建与训练解耦，便于后续替换为 RG-PCNet。

**Tech Stack:** Python 3.12、NumPy、pandas、scikit-learn、现有 GaitStabilityAnalyzer。

## Global Constraints

- 正样本绝不包含跌倒开始时刻及之后的帧。
- 训练和评估按 `subject_id` 隔离。
- 真实实验未达 F1 0.90 / Recall 0.88 时不可发布。
- 原始数据和训练权重不提交到 GitHub。

---

### Task 1: 无泄漏窗口构建器

**Files:**
- Create: `risk/prefall_dataset.py`
- Create: `tests/risk/test_prefall_dataset.py`

**Interfaces:**
- Produces: `build_prefall_rows(lock, metadata, cache_dir, horizon_sec=3.0, guard_sec=0.5)`，返回数值特征行与审计统计。

- [ ] **Step 1: 写失败测试**：构造 10 秒、64 帧、跌倒开始 6 秒的片段，断言正窗口只使用 19–35 帧（3 秒预测窗且 0.5 秒保护带）。
- [ ] **Step 2: 运行测试**：`python -m pytest tests/risk/test_prefall_dataset.py -q`，预期导入失败。
- [ ] **Step 3: 最小实现**：根据 `duration_seconds` 映射秒到帧，生成窗口和 `GaitStabilityAnalyzer` 特征；拒绝缺失元数据和无窗口样本。
- [ ] **Step 4: 运行测试**：同一命令通过。
- [ ] **Step 5: 提交**：`git add risk/prefall_dataset.py tests/risk/test_prefall_dataset.py && git commit -m "feat: build leakage-safe prefall windows"`。

### Task 2: 可复现训练入口

**Files:**
- Create: `scripts/build_prefall_gmdcsa24_dataset.py`
- Modify: `scripts/train_prefall_model.py`
- Test: `tests/risk/test_prefall_dataset.py`

**Interfaces:**
- Consumes: Task 1 的行与审计统计。
- Produces: CSV、`dataset_summary.json` 和包含窗口参数的模型卡。

- [ ] **Step 1: 写失败测试**：断言训练模型卡记录 `horizon_sec`、`guard_sec` 和排除计数。
- [ ] **Step 2: 运行测试**：目标测试失败。
- [ ] **Step 3: 最小实现**：CLI 生成 CSV；训练入口读取并写入数据审计字段。
- [ ] **Step 4: 运行回归**：`python -m pytest tests/risk/test_prefall_dataset.py tests/risk/test_prefall_model.py tests/risk/test_prefall_evaluation.py -q`。
- [ ] **Step 5: 提交**：只提交脚本、模块与测试。

### Task 3: 真实 LOSO 训练与验收

**Files:**
- No repository data changes; artifacts under `F:/datasets/mental_health/`.

- [ ] **Step 1: 生成真实 CSV**：使用 GMDCSA24 metadata、锁文件和姿态缓存。
- [ ] **Step 2: 训练模型**：运行 `train_prefall_model.py`。
- [ ] **Step 3: 验证**：运行分组评估，检查 `promoted` 只能在双门槛均满足时为 true。
- [ ] **Step 4: 记录结果**：保存模型、指标、模型卡；不提交原始数据/权重。
