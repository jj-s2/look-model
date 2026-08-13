# PA-DTSF Core Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 实现相位感知双时间尺度质量门控骨架模型，并用统一公开数据完成无泄漏训练、消融、跨数据集评估和模型晋级。

**Architecture:** 短分支通过 48 帧、10 fps 窗口复用 PoseC3D 事件能力；长分支通过 64 帧、2 fps 窗口使用轻量 TCN 和步态特征。两个分支经过质量门控后输出六相位、跌倒事件、跌倒前异常和恢复概率，再由现有状态机平滑。

**Tech Stack:** Python 3.10+、NumPy、scikit-learn、PyTorch 2.4+、OpenMMLab 可选运行边界、pytest。

## Global Constraints

- 必须先完成 2026-08-03-padtfs-data-foundation.md。
- 只有 fall_event 可以产生 critical。
- fall_coarse 样本不能伪造 impact 或精细相位标签。
- 缺失关键点保留 mask，不得当成真实零坐标。
- 模型、阈值、数据划分和指标必须共享同一个 release_id。
- 无 PyTorch 或无 OpenMMLab 时，schema、质量、状态机和规则回退测试仍须通过。
- 现有 PoseC3D 权重和公开接口保持兼容。

---

### Task 1: 定义模型输入输出契约

**Files:**
- Create: risk/phase_model/__init__.py
- Create: risk/phase_model/schema.py
- Create: tests/risk/phase_model/test_schema.py

**Interfaces:**
- Produces: PoseObservation、DualWindow、PhaseModelOutput、Phase

- [ ] **Step 1: 写严格 schema 失败测试**

    def test_phase_output_is_normalized_and_versioned():
        output = PhaseModelOutput(
            phase_probs=(0.7, 0.1, 0.1, 0.05, 0.03, 0.02),
            fall_event_prob=0.18,
            prefall_prob=0.1,
            recovery_prob=0.02,
            quality_score=0.9,
            embedding_version="padtfs-v1",
            model_version="fixture-v1",
        )
        assert sum(output.phase_probs) == pytest.approx(1.0)

    @pytest.mark.parametrize("value", [-0.1, 1.1, True])
    def test_probabilities_reject_invalid_values(value):
        with pytest.raises(ValueError):
            PhaseModelOutput(
                phase_probs=(0.7, 0.1, 0.1, 0.05, 0.03, 0.02),
                fall_event_prob=value, prefall_prob=0.1,
                recovery_prob=0.02, quality_score=0.9,
                embedding_version="padtfs-v1", model_version="fixture-v1",
            )

    def test_pose_observation_requires_timezone_and_mask():
        with pytest.raises(ValueError):
            PoseObservation(
                timestamp=datetime(2026, 8, 3),
                tracking_id="resident-1",
                keypoints=((0.0, 0.0),) * 17,
                scores=(0.9,) * 17,
                visible_mask=(True,) * 17,
                bbox=(0.0, 0.0, 10.0, 20.0),
                frame_size=(640, 480),
                stream_fresh=True,
            )

- [ ] **Step 2: 确认失败**

    python -m pytest tests/risk/phase_model/test_schema.py -q

- [ ] **Step 3: 实现契约**

Phase 枚举固定为 normal_adl、prefall_abnormal、descending、impact、fallen、recovering。

PoseObservation 字段：

    timestamp: datetime
    tracking_id: str
    keypoints: tuple[tuple[float, float], ...]
    scores: tuple[float, ...]
    visible_mask: tuple[bool, ...]
    bbox: tuple[float, float, float, float] | None
    frame_size: tuple[int, int]
    stream_fresh: bool

PhaseModelOutput 必须验证六项概率、各标量范围、版本非空和有限数值。

- [ ] **Step 4: JSON 往返和无 NumPy 导入测试**

使用子进程屏蔽 numpy、torch 后导入 risk.phase_model.schema，必须成功。

- [ ] **Step 5: 运行测试并提交**

    python -m pytest tests/risk/phase_model/test_schema.py -q
    git add risk/phase_model/__init__.py risk/phase_model/schema.py tests/risk/phase_model/test_schema.py
    git commit -m "feat: define phase model contracts"

---

### Task 2: 按真实时间戳构建双窗口

**Files:**
- Create: risk/phase_model/windows.py
- Create: tests/risk/phase_model/test_windows.py
- Modify: vision/pose_buffer.py
- Modify: tests/vision/test_pose_pipeline.py

**Interfaces:**
- Produces: DualTimescaleBuffer.append(observation) -> DualWindow | None
- Consumes: PoseObservation

- [ ] **Step 1: 写非恒定帧率失败测试**

    def test_sampler_uses_timestamps_not_frame_indices():
        buffer = DualTimescaleBuffer(
            short_frames=4, short_fps=2.0,
            long_frames=4, long_fps=1.0,
        )
        observations = jittered_observations(
            seconds=[0.0, 0.4, 1.1, 1.5, 2.2, 3.0, 4.1]
        )
        window = last_non_none(buffer.append(item) for item in observations)
        assert [round_gap(item) for item in window.short] == pytest.approx([0.5] * 3, abs=0.2)
        assert [round_gap(item) for item in window.long] == pytest.approx([1.0] * 3, abs=0.25)

    def test_out_of_order_timestamp_is_rejected():
        buffer.append(observation_at(2.0))
        with pytest.raises(ValueError, match="monotonic"):
            buffer.append(observation_at(1.0))

- [ ] **Step 2: 确认失败**

    python -m pytest tests/risk/phase_model/test_windows.py -q

- [ ] **Step 3: 实现最近时间采样**

保存最大 40 秒观察。短窗口目标间隔 0.1 秒，长窗口目标间隔 0.5 秒；对每个目标时间选择容差内最近观察。有效帧不足 80% 时不发出完整窗口。

默认参数：

    DualTimescaleBuffer(
        short_frames=48,
        short_fps=10.0,
        long_frames=64,
        long_fps=2.0,
        min_coverage=0.8,
    )

- [ ] **Step 4: 保持 PoseSequenceBuffer 回归兼容**

不得删除旧类。vision/pose_buffer.py 只增加适配函数：

    def to_pose_observation(result: PoseFrameResult, tracking_id: str) -> PoseObservation

- [ ] **Step 5: 运行测试并提交**

    python -m pytest tests/risk/phase_model/test_windows.py tests/vision/test_pose_pipeline.py -q
    git add risk/phase_model/windows.py tests/risk/phase_model/test_windows.py vision/pose_buffer.py tests/vision/test_pose_pipeline.py
    git commit -m "feat: sample dual-timescale pose windows"

---

### Task 3: 实现骨架归一化和质量门控

**Files:**
- Create: risk/phase_model/normalization.py
- Create: risk/phase_model/quality.py
- Create: tests/risk/phase_model/test_normalization.py
- Create: tests/risk/phase_model/test_quality.py

**Interfaces:**
- Produces: normalize_pose_window(window) -> NormalizedPoseWindow
- Produces: assess_window_quality(window) -> QualityAssessment

- [ ] **Step 1: 写尺度和平移不变测试**

    first = normalize_pose_window(window(scale=1.0, offset=(0.0, 0.0)))
    second = normalize_pose_window(window(scale=2.0, offset=(100.0, 50.0)))
    assert first.coordinates == pytest.approx(second.coordinates)
    assert first.visible_mask == second.visible_mask

- [ ] **Step 2: 写质量阈值失败测试**

    @pytest.mark.parametrize(
        "score,mode,max_level",
        [(0.71, "normal", "critical"),
         (0.60, "degraded", "warning"),
         (0.49, "abstained", "warning")],
    )
    def test_quality_modes(score, mode, max_level):
        result = classify_quality(score)
        assert (result.mode, result.max_level) == (mode, max_level)

    def test_two_seconds_without_reliable_pose_does_not_become_fall():
        assessment = assess_window_quality(no_pose_window(seconds=2.1))
        assert assessment.mode == "abstained"
        assert "no_reliable_pose" in assessment.reasons

- [ ] **Step 3: 确认失败**

    python -m pytest tests/risk/phase_model/test_normalization.py tests/risk/phase_model/test_quality.py -q

- [ ] **Step 4: 实现归一化**

每帧以髋中心为原点，以左右肩中点到髋中心的中位距离为尺度；尺度不可用时使用人体框对角线；仍不可用则设置 frame_valid=false。不可见关键点坐标可存零，但只有 visible_mask=false，任何下游统计都必须同时使用 mask。

- [ ] **Step 5: 实现质量评分**

    score = (
        0.25 * visible_ratio
        + 0.20 * mean_keypoint_score
        + 0.20 * torso_completeness
        + 0.15 * tracking_continuity
        + 0.10 * valid_frame_ratio
        + 0.10 * stream_freshness
    )

所有分量限制在 0 到 1，并在 QualityAssessment.components 中公开。

- [ ] **Step 6: 运行测试并提交**

    python -m pytest tests/risk/phase_model/test_normalization.py tests/risk/phase_model/test_quality.py -q
    git add risk/phase_model/normalization.py risk/phase_model/quality.py tests/risk/phase_model/test_normalization.py tests/risk/phase_model/test_quality.py
    git commit -m "feat: normalize poses and gate low-quality evidence"

---

### Task 4: 实现相位迁移约束

**Files:**
- Create: risk/phase_model/transitions.py
- Create: tests/risk/phase_model/test_transitions.py

**Interfaces:**
- Produces: allowed_transition(previous: Phase, current: Phase) -> bool
- Produces: transition_penalty(probabilities) -> TensorLike
- Produces: PhaseSmoother.update(output, timestamp) -> SmoothedPhase

- [ ] **Step 1: 写允许和禁止迁移测试**

    assert allowed_transition(Phase.NORMAL_ADL, Phase.PREFALL_ABNORMAL)
    assert allowed_transition(Phase.DESCENDING, Phase.IMPACT)
    assert allowed_transition(Phase.FALLEN, Phase.RECOVERING)
    assert not allowed_transition(Phase.NORMAL_ADL, Phase.FALLEN)
    assert not allowed_transition(Phase.IMPACT, Phase.PREFALL_ABNORMAL)

    def test_single_noisy_fallen_frame_is_not_confirmed():
        smoother = PhaseSmoother(min_persistence=2)
        assert not smoother.update(output_for("fallen", 0.91), t0).confirmed_fall

- [ ] **Step 2: 确认失败**

    python -m pytest tests/risk/phase_model/test_transitions.py -q

- [ ] **Step 3: 实现迁移矩阵**

矩阵必须逐项复制设计文档第 8.4 节。self-transition 始终允许。非法迁移概率质量之和构成 transition_penalty。

- [ ] **Step 4: 实现平滑器**

默认连续两个更新周期确认 prefall_warning；fall_event 必须出现 descending 后接 impact 或 fallen，或由现有 FallStateMachine 确认。恢复要求 recovering 后连续 normal_adl。

- [ ] **Step 5: 运行测试并提交**

    python -m pytest tests/risk/phase_model/test_transitions.py tests/risk/test_fall_state_machine.py -q
    git add risk/phase_model/transitions.py tests/risk/phase_model/test_transitions.py
    git commit -m "feat: constrain fall phase transitions"

---

### Task 5: 实现长分支、融合头和可选依赖边界

**Files:**
- Create: risk/phase_model/model.py
- Create: risk/phase_model/losses.py
- Create: tests/risk/phase_model/test_model.py
- Create: tests/risk/phase_model/test_losses.py

**Interfaces:**
- Produces: LongBranchTCN
- Produces: PhaseAwareFusionModel.forward(short_embedding, long_pose, quality) -> PhaseModelOutputTensor
- Produces: compute_multitask_loss(outputs, targets, supervision_mask) -> Tensor
- Produces: LossTargets(phase, fall_event, prefall, recovery)
- Consumes: 短分支 512 维 PoseC3D embedding 或实现协议的测试替身

- [ ] **Step 1: 写模型形状和门控失败测试**

    torch = pytest.importorskip("torch")
    model = PhaseAwareFusionModel(short_dim=16, joints=17, hidden_dim=32)
    result = model(
        short_embedding=torch.zeros(2, 16),
        long_pose=torch.zeros(2, 64, 17, 3),
        short_quality=torch.tensor([1.0, 0.0]),
        long_quality=torch.tensor([0.0, 1.0]),
    )
    assert result.phase_logits.shape == (2, 6)
    assert result.fall_event_logit.shape == (2,)
    assert result.prefall_logit.shape == (2,)
    assert result.recovery_logit.shape == (2,)

- [ ] **Step 2: 写 supervision mask 失败测试**

    outputs = PhaseModelOutputTensor(
        phase_logits=torch.zeros(1, 6),
        fall_event_logit=torch.zeros(1),
        prefall_logit=torch.zeros(1),
        recovery_logit=torch.zeros(1),
    )
    targets = LossTargets(
        phase=torch.tensor([-1]),
        fall_event=torch.tensor([1.0]),
        prefall=torch.tensor([-1.0]),
        recovery=torch.tensor([-1.0]),
    )
    loss = compute_multitask_loss(outputs, targets, {"fall_event"})
    assert loss.components["phase"].item() == 0.0
    assert loss.components["fall_event"].item() > 0.0

- [ ] **Step 3: 确认失败**

    python -m pytest tests/risk/phase_model/test_model.py tests/risk/phase_model/test_losses.py -q

- [ ] **Step 4: 实现长分支**

LongBranchTCN 使用三层一维深度可分离卷积，扩张率 1、2、4，隐藏维度默认 128，dropout 0.2。输入最后一维为 x、y、visible_mask，步态特征在池化后拼接。

- [ ] **Step 5: 实现质量融合**

    weights = torch.stack([short_quality, long_quality], dim=1)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    fused = (
        weights[:, :1] * short_projection(short_embedding)
        + weights[:, 1:] * long_projection(long_embedding)
    )

当两个质量都为零时输出 abstain_logit，并由 service 拒绝，不使用随机等权。

- [ ] **Step 6: 实现精确损失权重**

    total = (
        1.0 * phase_loss
        + 0.8 * prefall_focal_loss
        + 0.7 * fall_event_loss
        + 0.3 * recovery_loss
        + 0.2 * order_penalty
    )

每项损失只有在 supervision_mask 有对应字段时参与。全空 mask 必须抛 ValueError。

- [ ] **Step 7: 延迟导入 PyTorch**

risk.phase_model.schema、quality、transitions 在无 torch 环境可导入。model.py 内部导入失败时抛出带安装说明的 RuntimeError，不能在包 __init__.py 自动导入模型实现。

- [ ] **Step 8: 运行测试并提交**

    python -m pytest tests/risk/phase_model/test_model.py tests/risk/phase_model/test_losses.py -q
    git add risk/phase_model/model.py risk/phase_model/losses.py tests/risk/phase_model/test_model.py tests/risk/phase_model/test_losses.py
    git commit -m "feat: add quality-gated phase fusion model"

---

### Task 6: 建立推理服务和事件契约

**Files:**
- Create: risk/phase_model/service.py
- Create: tests/risk/phase_model/test_service.py
- Modify: core/events.py
- Modify: fusion/decision_engine.py
- Modify: tests/fusion/test_decision_engine.py

**Interfaces:**
- Produces: PhaseRiskService.observe(observation) -> tuple[SensorEvent, ...]
- Adds: EventType.PREFALL_WARNING
- Adds: RiskDecision.kind prefall_warning

- [ ] **Step 1: 写 critical 边界失败测试**

    @pytest.mark.parametrize("kind", [
        EventType.PREFALL_WARNING,
        EventType.FALL_FORECAST,
        EventType.WELLBEING_CHANGE,
    ])
    def test_non_event_risk_never_becomes_critical(kind):
        decision = engine.evaluate([event(kind, score=0.99)], NOW)[0]
        assert decision.level == "warning"

- [ ] **Step 2: 写服务降级失败测试**

    output = service.observe(low_quality_observation(score=0.49))
    assert all(item.quality.confidence <= 0.49 for item in output)
    assert not any(
        item.event_type is EventType.FALL_EVENT and item.payload.get("confirmed")
        for item in output
    )

- [ ] **Step 3: 确认失败**

    python -m pytest tests/risk/phase_model/test_service.py tests/fusion/test_decision_engine.py -q

- [ ] **Step 4: 扩展事件枚举和决策类型**

PREFALL_WARNING 单独分支；fall_forecast 分数大于等于 0.6 时最高 warning，不再返回 critical。wellbeing_change 保持 screening_only。

- [ ] **Step 5: 实现服务编排**

service 依赖注入：

    class PhasePredictor(Protocol):
        def predict(self, window: DualWindow) -> PhaseModelOutput: ...

    class PhaseRiskService:
        def __init__(self, predictor, buffer, smoother, clock): ...
        def observe(self, observation: PoseObservation) -> tuple[SensorEvent, ...]: ...

服务不读取文件、不初始化 OpenMMLab、不发送网络请求。

- [ ] **Step 6: 运行测试并提交**

    python -m pytest tests/risk/phase_model/test_service.py tests/fusion/test_decision_engine.py tests/test_events.py -q
    git add risk/phase_model/service.py tests/risk/phase_model/test_service.py core/events.py fusion/decision_engine.py tests/fusion/test_decision_engine.py
    git commit -m "feat: separate prefall warnings from fall events"

---

### Task 7: 实现受试者级与跨数据集评估

**Files:**
- Create: risk/phase_model/evaluation.py
- Create: tests/risk/phase_model/test_evaluation.py
- Create: scripts/evaluate_phase_model.py
- Create: tests/integration/test_phase_evaluation.py

**Interfaces:**
- Produces: evaluate_predictions(records, threshold) -> PhaseEvaluation
- Produces: should_promote_phase_model(candidate, baseline) -> PromotionDecision

- [ ] **Step 1: 写指标失败测试**

测试必须覆盖：

- phase macro F1；
- fall precision、recall、F1；
- prefall AUPRC 和 ROC-AUC；
- 90% 召回附近 FPR；
- lead_time_seconds 中位数；
- false_alarms_per_hour；
- per_subject 和 per_dataset；
- abstention coverage。

    decision = should_promote_phase_model(
        candidate={"auprc": 0.63, "recall": 0.90, "fpr_at_recall": 0.24},
        baseline={"auprc": 0.60, "recall": 0.90, "fpr_at_recall": 0.33},
    )
    assert decision.promoted is True

- [ ] **Step 2: 确认失败**

    python -m pytest tests/risk/phase_model/test_evaluation.py -q

- [ ] **Step 3: 实现评估**

当标签只有一个类别时 AUC 返回 unavailable 和 reason，不返回 0。false_alarms_per_hour 需要 duration_seconds 大于零。lead time 只计算首次有效预警早于 descending 的事件。

- [ ] **Step 4: 实现晋级门槛**

满足以下任一性能条件且召回不下降：

- AUPRC 相对提高至少 5%；或
- FPR 在同等召回下相对下降至少 20%。

同时要求至少三轮 leave-one-dataset-out 不劣于基线。合成或 demo 结果永远 promoted=false。

- [ ] **Step 5: 实现 CLI**

    python scripts/evaluate_phase_model.py \
      --predictions experiments/releases/padtfs-v1-seed42/predictions.jsonl \
      --split-manifest datasets/annotations/unified/v1/split_manifest.json \
      --output experiments/releases/padtfs-v1-seed42/metrics.json

CLI 验证所有输入 release_id 一致。

- [ ] **Step 6: 运行测试并提交**

    python -m pytest tests/risk/phase_model/test_evaluation.py tests/integration/test_phase_evaluation.py -q
    git add risk/phase_model/evaluation.py tests/risk/phase_model/test_evaluation.py scripts/evaluate_phase_model.py tests/integration/test_phase_evaluation.py
    git commit -m "feat: evaluate phase model across subjects and datasets"

---

### Task 8: 训练、校准和发布产物

**Files:**
- Create: risk/phase_model/calibration.py
- Create: tests/risk/phase_model/test_calibration.py
- Create: scripts/train_phase_model.py
- Create: scripts/generate_phase_release.py
- Create: tests/integration/test_phase_release.py
- Create: configs/skeleton/padtfs_v1.py

**Interfaces:**
- Produces: TemperatureCalibrator.fit(validation_logits, labels)
- Produces: create_release_artifacts(release_id, inputs, output_dir)

- [ ] **Step 1: 写测试折隔离失败测试**

    def test_calibrator_refuses_test_partition():
        with pytest.raises(ValueError, match="validation"):
            calibrator.fit(logits, labels, partition="test")

    def test_release_requires_matching_ids(tmp_path):
        with pytest.raises(ValueError, match="release_id"):
            create_release_artifacts(
                release_id="r1",
                metrics={"release_id": "r2"},
                benchmark={"release_id": "r1"},
                output_dir=tmp_path,
            )

- [ ] **Step 2: 确认失败**

    python -m pytest tests/risk/phase_model/test_calibration.py tests/integration/test_phase_release.py -q

- [ ] **Step 3: 实现校准器**

温度参数必须为正，使用验证折最小化负对数似然。保存 before/after Brier、ECE、样本数和 split hash。

- [ ] **Step 4: 实现训练入口**

训练入口只接受冻结 dataset_lock 和 split_manifest：

    python scripts/train_phase_model.py \
      --config configs/skeleton/padtfs_v1.py \
      --dataset-lock datasets/annotations/unified/v1/dataset_lock.json \
      --split-manifest datasets/annotations/unified/v1/split_manifest.json \
      --release-id padtfs-v1-seed42

固定 seed=42，保存依赖版本、Git commit、配置哈希和权重 SHA-256。无法加载短分支预训练权重时终止，不得使用随机短分支冒充微调。

- [ ] **Step 5: 实现发布目录**

必须生成：

    dataset_lock.json
    split_manifest.json
    config.py
    metrics.json
    per_subject_metrics.csv
    per_dataset_metrics.csv
    confusion_matrices.json
    calibration.json
    ablation.json
    benchmark.json
    model_card.md
    checksums.sha256

缺失指标写 unavailable 和 reason。模型卡自动写入 simulated_young_subjects=true、clinical_validation=false。

- [ ] **Step 6: 使用 fixture 做端到端测试**

fixture 训练允许使用微型 fake predictor，不消耗 GPU，并强制 promoted=false。

    python -m pytest tests/integration/test_phase_release.py -q

- [ ] **Step 7: 运行本计划验证并提交**

    python -m pytest tests/risk/phase_model tests/integration/test_phase_evaluation.py tests/integration/test_phase_release.py tests/fusion/test_decision_engine.py -q
    git add risk/phase_model/calibration.py tests/risk/phase_model/test_calibration.py scripts/train_phase_model.py scripts/generate_phase_release.py tests/integration/test_phase_release.py configs/skeleton/padtfs_v1.py
    git commit -m "feat: train calibrate and package PA-DTSF releases"

---

## Plan Completion Gate

运行：

    python -m pytest tests/risk/phase_model tests/datasets tests/integration/test_phase_evaluation.py tests/integration/test_phase_release.py tests/fusion/test_decision_engine.py -q
    python -m pytest -q

具备数据和 GPU 后再运行真实训练；没有训练完成前不得声称 PA-DTSF 超过基线。

验收：

- 短窗口 48 帧 10 fps、长窗口 64 帧 2 fps；
- 六相位输出和监督 mask 正确；
- 质量门控与非法迁移测试通过；
- fall_forecast 和 prefall_warning 最高 warning；
- 评估包含受试者级、数据集级、误报和提前量；
- 晋级只使用冻结验证规则；
- 所有发布文件共享 release_id；
- 无真实训练证据时 promoted=false。
