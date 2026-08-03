# C6c Live Inference and Alert Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 将 C6c 实时视频接入 PA-DTSF 完整推理链，建立可审计告警交付，并生成 P50/P95 延迟和每小时误报的真实设备证据。

**Architecture:** 现有 EzvizStreamAdapter 负责有界拉流和重连，新的 VisionPhaseSource 负责人体检测、姿态、双窗口和事件生成。AlertDispatcher 保持本地审计核心，DeliveryAdapter 在其后提供 dashboard、webhook 和经验证的萤石能力。

**Tech Stack:** Python 3.10+、OpenCV、现有萤石客户端、pytest、标准库 HTTP 测试替身；生产模型依赖按核心模型计划安装。

## Global Constraints

- 必须先完成数据底座和 PA-DTSF 核心模型计划。
- 单元测试不得联系真实萤石平台。
- URL、Token、验证码和序列号不得进入异常文本或 Git。
- 网络推送失败不能阻塞本地告警或实时采集。
- 摄像头扬声器只有官方能力和实测同时通过才启用。
- SDNL1 未配置真实接口时保持 unavailable。

---

### Task 1: 将 PA-DTSF 封装为实时视觉事件源

**Files:**
- Create: pipeline/vision_phase_source.py
- Create: tests/pipeline/test_vision_phase_source.py
- Modify: pipeline/live_service.py
- Modify: tests/pipeline/test_live_service.py

**Interfaces:**
- Produces: VisionPhaseSource.poll(now: datetime) -> SourceBatch
- Consumes: EzvizStreamAdapter、PosePipeline、PhaseRiskService

- [ ] **Step 1: 写无帧和正常帧失败测试**

    def test_failed_read_returns_availability_without_inference():
        source = VisionPhaseSource(
            stream=FakeStream([(False, None)]),
            pose_pipeline=pose_pipeline,
            tracker=tracker,
            phase_service=phase_service,
        )
        batch = source.poll(NOW)
        assert batch.frame is None
        assert batch.events[0].event_type is EventType.AVAILABILITY
        assert batch.events[0].quality.available is False
        assert predictor.calls == 0

    def test_frame_flows_to_pose_and_phase_service():
        source = VisionPhaseSource(
            stream=FakeStream([(True, FRAME)]),
            pose_pipeline=pose_pipeline,
            tracker=tracker,
            phase_service=phase_service,
        )
        batch = source.poll(NOW)
        assert batch.frame is FRAME
        assert phase_service.observations

- [ ] **Step 2: 确认失败**

    python -m pytest tests/pipeline/test_vision_phase_source.py -q

- [ ] **Step 3: 实现有界编排**

VisionPhaseSource 构造时注入 stream、pose_pipeline、tracker、phase_service 和 clock。poll 只处理一个采样周期，不包含无限循环。无人体时生成 POSE 可用事件，payload 标记 no_person=true，不输出 normal phase。

- [ ] **Step 4: 接入 LiveMonitoringService**

保持 EventSource 协议不变。source.name 固定为 vision，使现有健康状态映射继续工作。

- [ ] **Step 5: 运行测试并提交**

    python -m pytest tests/pipeline/test_vision_phase_source.py tests/pipeline/test_live_service.py tests/vision -q
    git add pipeline/vision_phase_source.py tests/pipeline/test_vision_phase_source.py pipeline/live_service.py tests/pipeline/test_live_service.py
    git commit -m "feat: stream C6c poses into phase inference"

---

### Task 2: 建立告警交付协议和安全 webhook

**Files:**
- Create: alerts/delivery/__init__.py
- Create: alerts/delivery/base.py
- Create: alerts/delivery/webhook.py
- Create: alerts/delivery/dashboard.py
- Create: tests/alerts/test_delivery.py
- Modify: alerts/dispatcher.py
- Modify: tests/alerts/test_dispatcher.py

**Interfaces:**
- Produces: DeliveryAdapter.send(decision) -> DeliveryResult
- Produces: CompositeDelivery.send(decision) -> tuple[DeliveryResult, ...]

- [ ] **Step 1: 写网络失败不阻塞失败测试**

    def test_webhook_failure_is_reported_without_raising(tmp_path):
        adapter = WebhookDelivery(
            "https://alerts.example.test",
            client=RaisingClient(),
            timeout=2.0,
        )
        result = adapter.send(WARNING)
        assert result.sent is False
        assert result.reason == "webhook_request_failed"

    def test_dispatcher_persists_before_delivery(tmp_path):
        dispatcher = AlertDispatcher(tmp_path / "alerts.jsonl", delivery=RaisingDelivery())
        result = dispatcher.dispatch(WARNING)
        assert result.sent is True
        assert dispatcher.recent_alerts()[0]["sent"] is True
        assert dispatcher.recent_alerts()[0]["delivery"][0]["sent"] is False

- [ ] **Step 2: 确认失败**

    python -m pytest tests/alerts/test_delivery.py tests/alerts/test_dispatcher.py -q

- [ ] **Step 3: 实现协议**

    @dataclass(frozen=True)
    class DeliveryResult:
        channel: str
        sent: bool
        reason: str
        attempted_at: datetime

    class DeliveryAdapter(Protocol):
        def send(self, decision: RiskDecision) -> DeliveryResult: ...

Webhook 只发送 kind、level、score、reasons、subject_id 的脱敏别名、timestamp 和 recommended_action。禁止发送视频 URL、Token、验证码或完整序列号。

- [ ] **Step 4: 修改 dispatcher**

先写本地 JSONL，再调用 delivery。现有 DispatchResult.sent 仍代表本地审计写入成功，网络结果写入 delivery 字段。重复告警不调用网络。

- [ ] **Step 5: 运行测试并提交**

    python -m pytest tests/alerts -q
    git add alerts/delivery alerts/dispatcher.py tests/alerts
    git commit -m "feat: add nonblocking alert delivery adapters"

---

### Task 3: 萤石交付能力探测与保守适配

**Files:**
- Create: alerts/delivery/ezviz.py
- Create: tests/alerts/test_ezviz_delivery.py
- Create: scripts/probe_ezviz_delivery.py
- Modify: docs/device-capability-report.md

**Interfaces:**
- Produces: EzvizDeliveryCapability
- Produces: EzvizDelivery.send(decision) -> DeliveryResult

- [ ] **Step 1: 写未验证能力失败测试**

    def test_unverified_speaker_capability_never_sends():
        delivery = EzvizDelivery(client=FakeClient(), capability=capability(verified=False))
        result = delivery.send(CRITICAL_FALL)
        assert result.sent is False
        assert result.reason == "ezviz_delivery_not_verified"
        assert FakeClient.calls == []

- [ ] **Step 2: 确认失败**

    python -m pytest tests/alerts/test_ezviz_delivery.py -q

- [ ] **Step 3: 实现能力对象**

能力字段固定：

    account_region
    device_model
    app_notification_supported
    two_way_audio_supported
    proactive_audio_supported
    verified_at
    evidence_source

只有 verified_at、evidence_source 和对应 supported=true 同时存在才允许调用。

- [ ] **Step 4: 实现只读探测脚本**

脚本默认只查询设备和账户能力，不发送消息。真实发送必须显式传 --send-test，并在终端再次显示目标设备脱敏标识。脚本输出到 outputs/device-delivery-capability.json，不提交 Git。

- [ ] **Step 5: 更新能力报告并提交**

文档只记录可公开的型号、能力结果和时间，不包含密钥或完整序列号。

    python -m pytest tests/alerts/test_ezviz_delivery.py -q
    git add alerts/delivery/ezviz.py tests/alerts/test_ezviz_delivery.py scripts/probe_ezviz_delivery.py docs/device-capability-report.md
    git commit -m "feat: gate EZVIZ delivery on verified capability"

---

### Task 4: 完整链路基准和误报统计

**Files:**
- Modify: scripts/benchmark_live_pipeline.py
- Create: tests/integration/test_live_benchmark.py
- Modify: scripts/generate_evaluation_report.py
- Modify: tests/integration/test_monitoring_flow.py

**Interfaces:**
- Produces: benchmark.json，pipeline_scope=full_inference
- Produces: false_alarms_per_hour、p50_seconds、p95_seconds

- [ ] **Step 1: 写拒绝 capture-only 失败测试**

    def test_release_report_rejects_capture_only_benchmark():
        report = generate_report(
            classification=valid_metrics(),
            benchmark={"pipeline_scope": "capture_decode_only"},
        )
        assert report["release_gate"] == "FAIL"
        assert "full inference" in report["reasons"]

    def test_false_alarm_rate_uses_observed_duration():
        assert false_alarms_per_hour(false_alerts=2, duration_seconds=1800) == 4.0

- [ ] **Step 2: 确认失败**

    python -m pytest tests/integration/test_live_benchmark.py tests/integration/test_monitoring_flow.py -q

- [ ] **Step 3: 测量完整阶段**

每个样本记录：

    capture_ms
    detection_ms
    pose_ms
    window_ms
    phase_model_ms
    decision_ms
    total_ms

benchmark 只有在 detection、pose、phase_model 和 decision 都实际执行时才标记 full_inference。预热样本数和测量样本数单独记录。

- [ ] **Step 4: 统计误报**

误报需要人工标注或确认的 normal_activity 观察时长。unknown 时间段不进入分母。任何 demo 回放不得写成 live_device。

- [ ] **Step 5: 运行测试并提交**

    python -m pytest tests/integration/test_live_benchmark.py tests/integration/test_monitoring_flow.py tests/test_scripts.py -q
    git add scripts/benchmark_live_pipeline.py tests/integration/test_live_benchmark.py scripts/generate_evaluation_report.py tests/integration/test_monitoring_flow.py
    git commit -m "feat: benchmark full live inference and false alarms"

---

### Task 5: C6c 30 分钟真实验证和演示入口

**Files:**
- Modify: scripts/run_pipeline.py
- Modify: ui/dashboard.py
- Modify: docs/demo-script.md
- Modify: docs/deployment.md
- Create: docs/c6c-validation-protocol.md
- Modify: tests/ui/test_dashboard_view_model.py

**Interfaces:**
- Produces: --live-ezviz、--record-benchmark、--subject-alias 参数
- Produces: 分离展示 fall_event、prefall_warning、fall_forecast、wellbeing_change

- [ ] **Step 1: 写 UI 分流失败测试**

    view = build_dashboard_view([
        decision("fall_event", "critical"),
        decision("prefall_warning", "warning"),
        decision("wellbeing_change", "watch"),
    ])
    assert len(view.emergency_events) == 1
    assert len(view.prefall_warnings) == 1
    assert len(view.wellbeing_prompts) == 1

- [ ] **Step 2: 确认失败**

    python -m pytest tests/ui/test_dashboard_view_model.py -q

- [ ] **Step 3: 修改演示入口**

实时启动必须先验证：

- 环境变量存在但不打印值；
- 模型 SHA-256 与 model card 一致；
- 数据 schema 和阈值版本兼容；
- C6c 流可读；
- 输出目录可写。

失败时显示非技术原因并停止，不能加载随机模型。

- [ ] **Step 4: 编写 30 分钟协议**

协议包含走动、坐下、起身、捡物、下蹲、遮挡、离开再进入。记录开始结束时间、release_id、阈值、观察者确认、误报和重连。明确不要求老人模拟跌倒。

- [ ] **Step 5: 运行离线测试并提交**

    python -m pytest tests/ui/test_dashboard_view_model.py tests/pipeline tests/alerts tests/integration/test_live_benchmark.py -q
    git add scripts/run_pipeline.py ui/dashboard.py docs/demo-script.md docs/deployment.md docs/c6c-validation-protocol.md tests/ui/test_dashboard_view_model.py
    git commit -m "feat: present separated risks in C6c live demo"

- [ ] **Step 6: 在设备在线时运行真实验证**

    python scripts/run_pipeline.py --live-ezviz --record-benchmark --subject-alias resident-1

真实产物写 outputs，不提交。之后运行：

    python scripts/generate_evaluation_report.py

只有报告显示 full_inference、P95 小于等于 2 秒、误报小于等于 1 次/小时，才能把实时门槛标记通过。

---

## Plan Completion Gate

    python -m pytest tests/pipeline tests/vision tests/alerts tests/ui tests/integration/test_live_benchmark.py -q
    python -m pytest -q

验收：

- C6c 帧进入检测、姿态、PA-DTSF、决策和本地审计；
- 断流、超时和低质量不产生虚假 critical；
- 网络通知失败不阻塞；
- 萤石语音能力未经验证时保持关闭；
- 四类结果在 UI 分开；
- 真实 30 分钟验证产物包含完整延迟和误报证据；
- outputs 未提交。
