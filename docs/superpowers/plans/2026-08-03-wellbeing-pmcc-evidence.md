# Wellbeing and U-PMCC Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 将低打扰心理筛查、日级活动摘要和 U-PMCC 长期风险安全接入演示，同时保持非诊断、非紧急和证据可追溯。

**Architecture:** 心理模块继续使用 GDS-15、趋势和交互冷却；日级活动摘要从骨架派生，不存原视频。U-PMCC 读取日级摘要产生 24 小时、72 小时和 7 天研究预测，但与实时 fall_event 完全分流。

**Tech Stack:** Python 标准库、现有 mental 和 risk/pmcc 包、pytest；CHARLS 分析可选使用 pandas/scikit-learn，但不得成为运行时依赖。

## Global Constraints

- 不持续录音，不做人脸、语调或摄像头抑郁诊断。
- GDS-15 是筛查，不是诊断。
- 完整邀请冷却 28 天，短问答冷却 7 天，21:00 至 08:00 为安静时段。
- wellbeing_change 和 fall_forecast 最高 warning。
- synthetic_research 和 offline_fixture 始终 promoted=false。
- SDNL1 未验证时使用 vision_only 或 unavailable，不补造数据。

---

### Task 1: 固化低打扰和非诊断回归测试

**Files:**
- Modify: tests/mental/test_interaction_policy.py
- Modify: tests/mental/test_gds15.py
- Modify: tests/mental/test_trend.py
- Create: mental/copy_policy.py
- Create: tests/mental/test_copy_policy.py

**Interfaces:**
- Produces: validate_wellbeing_copy(text: str) -> None

- [ ] **Step 1: 写禁止诊断语言失败测试**

    @pytest.mark.parametrize("text", [
        "系统判断您患有抑郁症",
        "摄像头已诊断心理疾病",
        "您将在72小时内跌倒",
    ])
    def test_diagnostic_copy_is_rejected(text):
        with pytest.raises(ValueError, match="prohibited"):
            validate_wellbeing_copy(text)

    def test_screening_copy_is_allowed():
        validate_wellbeing_copy(
            "最近一周活动规律有变化，是否愿意做一个简短状态问答？"
        )

- [ ] **Step 2: 增加时间策略边界测试**

覆盖 07:59、08:00、20:59、21:00；覆盖完整和短问答各自冷却前一秒与到期时刻；用户主动发起必须允许。

- [ ] **Step 3: 确认失败**

    python -m pytest tests/mental -q

- [ ] **Step 4: 实现 copy policy**

使用审核过的允许模板 ID，不允许运行时自由生成医疗结论。validate_wellbeing_copy 是最后防线，检测“诊断、患有、确定抑郁、将在…跌倒”等禁止表达。

- [ ] **Step 5: 运行测试并提交**

    python -m pytest tests/mental -q
    git add mental/copy_policy.py tests/mental tests/mental/test_copy_policy.py
    git commit -m "test: enforce low-burden non-diagnostic wellbeing flow"

---

### Task 2: 从骨架事件生成日级活动摘要

**Files:**
- Create: risk/daily_summary.py
- Create: tests/risk/test_daily_summary.py
- Modify: storage/retention.py
- Modify: tests/storage/test_retention.py

**Interfaces:**
- Produces: DailyActivitySummary
- Produces: DailySummaryBuilder.observe(event) -> None
- Produces: DailySummaryBuilder.finalize(subject_id, local_date) -> DailyActivitySummary

- [ ] **Step 1: 写质量和隐私失败测试**

    def test_low_quality_minutes_do_not_count_as_inactivity():
        summary = build_summary([
            minute(activity=0.0, available=False),
            minute(activity=0.0, available=True),
        ])
        assert summary.observed_minutes == 1
        assert summary.missing_minutes == 1

    def test_summary_contains_no_raw_video_or_device_secret():
        payload = summary.to_dict()
        assert "frame" not in payload
        assert "stream_url" not in payload
        assert "device_serial" not in payload

- [ ] **Step 2: 确认失败**

    python -m pytest tests/risk/test_daily_summary.py -q

- [ ] **Step 3: 实现摘要**

字段固定：

    subject_id
    local_date
    observed_minutes
    missing_minutes
    active_minutes
    sedentary_minutes
    transition_count
    gait_quality_median
    prefall_warning_count
    confirmed_fall_count
    source_quality
    provenance_ids

少于 30 个有效观察分钟时 available=false。缺失不等于静止。

- [ ] **Step 4: 设置保留策略**

原始帧遵守现有短期保留；日级摘要可长期保留但使用 subject alias。删除授权后同时删除个人摘要和身份映射。

- [ ] **Step 5: 运行测试并提交**

    python -m pytest tests/risk/test_daily_summary.py tests/storage/test_retention.py -q
    git add risk/daily_summary.py tests/risk/test_daily_summary.py storage/retention.py tests/storage/test_retention.py
    git commit -m "feat: derive privacy-preserving daily activity summaries"

---

### Task 3: 将日级摘要安全接入 U-PMCC

**Files:**
- Create: risk/pmcc/adapters.py
- Create: tests/risk/pmcc/test_adapters.py
- Modify: risk/pmcc/service.py
- Modify: tests/risk/pmcc/test_service.py
- Modify: fusion/decision_engine.py
- Modify: tests/fusion/test_decision_engine.py

**Interfaces:**
- Produces: daily_summary_to_observation(summary) -> DailyObservation
- Consumes: PMCCService.observe、PMCCService.forecast

- [ ] **Step 1: 写缺模态和等级失败测试**

    def test_visual_summary_maps_missing_sdnl1_to_unavailable():
        observation = daily_summary_to_observation(summary(), physiology=None)
        assert observation.provenance["mode"] == "vision_only"
        assert observation.physiology_available is False

    def test_pmcc_forecast_never_becomes_critical():
        decision = engine.evaluate([forecast_event(score=0.99)], NOW)[0]
        assert decision.kind == "fall_forecast"
        assert decision.level == "warning"

- [ ] **Step 2: 确认失败**

    python -m pytest tests/risk/pmcc/test_adapters.py tests/risk/pmcc/test_service.py tests/fusion/test_decision_engine.py -q

- [ ] **Step 3: 实现适配器**

只有 observed_minutes、active_minutes、transition_count、gait_quality_median 和 warning/fall count 进入 U-PMCC。GDS-15 原始答案不进入跌倒模型；只允许 screening_completed 布尔上下文。

- [ ] **Step 4: 保持拒绝规则**

少于 14 日可按冷启动规则预测；缺少视觉步态证据时 abstained 或最多 watch；只存在睡眠或心理变化时不能提高到 warning。

- [ ] **Step 5: 运行测试并提交**

    python -m pytest tests/risk/pmcc tests/fusion/test_decision_engine.py -q
    git add risk/pmcc/adapters.py tests/risk/pmcc/test_adapters.py risk/pmcc/service.py tests/risk/pmcc/test_service.py fusion/decision_engine.py tests/fusion/test_decision_engine.py
    git commit -m "feat: feed daily vision summaries into PMCC"

---

### Task 4: 分离 UI、证据报告和 CHARLS 可选研究

**Files:**
- Modify: ui/dashboard.py
- Modify: tests/ui/test_dashboard_view_model.py
- Modify: scripts/generate_pmcc_report.py
- Modify: docs/pmcc-evaluation-report.md
- Create: scripts/analyze_charls_associations.py
- Create: tests/integration/test_charls_analysis.py
- Modify: docs/demo-script.md

**Interfaces:**
- Produces: analyze_charls(input_path, output_path, seed) -> AssociationReport

- [ ] **Step 1: 写 UI 分离失败测试**

    view = build_dashboard_view([
        decision("fall_event", "critical"),
        decision("fall_forecast", "warning"),
        decision("wellbeing_change", "watch"),
    ])
    assert view.emergency_events[0].kind == "fall_event"
    assert view.long_horizon_forecasts[0].level == "warning"
    assert view.wellbeing_prompts[0].quality == "screening_only"

- [ ] **Step 2: 写 CHARLS 边界失败测试**

    report = analyze_charls(FIXTURE_CSV, tmp_path / "report.json", seed=42)
    assert report.kind == "population_association"
    assert report.trains_camera_model is False
    assert report.clinical_claim is False

- [ ] **Step 3: 确认失败**

    python -m pytest tests/ui/test_dashboard_view_model.py tests/integration/test_charls_analysis.py -q

- [ ] **Step 4: 实现报告**

PMCC 报告分开显示 24h、72h、7d 风险、区间、覆盖率和拒绝原因。synthetic_research 报告固定：

    promoted: false
    release_eligible: false
    clinical_validation: false

CHARLS 脚本只做预注册变量的描述统计和可解释回归，输入由用户自行从官方源下载；脚本不下载数据、不上传数据、不输出个人行。

- [ ] **Step 5: 更新演示文案**

演示顺序：

1. 实时跌倒事件；
2. 秒级跌倒前 warning；
3. 24h/72h/7d 研究风险；
4. 自愿心理筛查和趋势提示。

每屏显示边界说明，不把四类分数合并。

- [ ] **Step 6: 运行测试并提交**

    python -m pytest tests/ui/test_dashboard_view_model.py tests/integration/test_charls_analysis.py tests/risk/pmcc tests/mental -q
    git add ui/dashboard.py tests/ui/test_dashboard_view_model.py scripts/generate_pmcc_report.py docs/pmcc-evaluation-report.md scripts/analyze_charls_associations.py tests/integration/test_charls_analysis.py docs/demo-script.md
    git commit -m "feat: separate wellbeing and long-horizon evidence"

---

## Plan Completion Gate

    python -m pytest tests/mental tests/risk/pmcc tests/risk/test_daily_summary.py tests/ui/test_dashboard_view_model.py tests/integration/test_charls_analysis.py -q
    python scripts/generate_pmcc_report.py
    python -m pytest -q

验收：

- 心理邀请严格遵守安静时段和冷却；
- 用户可主动发起；
- 输出不含诊断性语言；
- 日级摘要不含原始视频和设备凭据；
- 缺失 SDNL1 明确标为 unavailable；
- U-PMCC 与实时事件分流且最高 warning；
- 合成 PMCC 保持 promoted=false；
- CHARLS 只产生群体关联报告；
- UI 不合并四类风险。
