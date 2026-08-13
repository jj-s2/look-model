# U-PMCC 个体化多模态时序风险链 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有跌倒事件、步态、SDNL1 和心理筛查模块之上，增加可审计的 U-PMCC 跌倒前风险预测路径，默认 CPU 可运行，并在缺失设备或模型时安全降级。

**Architecture:** 新建 `risk/pmcc/` 包，采用严格 schema → 个体基线 → 变化事件 → 72 小时时间关联链 → 离散时间风险率 → 不确定性拒绝 → 反馈存储的单向数据流。规则/CPU 校准器是默认路径，PyTorch TCN 只是可选模型；不重写现有 `fall_event` 状态机和告警接口。

**Tech Stack:** Python 3.12+/3.14 compatible, dataclasses, NumPy, scikit-learn (training optional at import time), optional PyTorch, pytest, JSONL/JSON artifacts.

## Global Constraints

- 默认路径只能依赖 NumPy 和 scikit-learn；PyTorch TCN 必须是可选依赖。
- `fall_event`、心理筛查、告警、设备客户端和 `gait_stability` 的既有公开行为保持兼容。
- 所有时间戳必须带时区；所有日级聚合按记录时区执行。
- 缺失字段保留为 `null`/掩码，不得用零代表正常。
- `synthetic_research` 和 `offline_fixture` 产物必须 `promoted=false`，不得进入真实发布指标。
- 时间关联链表示时间关联，不表示医学因果。
- U-PMCC 只产生 `fall_forecast`；`critical` 仍只由现有 `fall_event` 紧急流程产生。
- 心理模块只输出筛查/变化提示，不输出疾病诊断或治疗建议。
- 不提交 `outputs/`、密钥、令牌、设备验证码、完整序列号或本机绝对路径。
- 每个任务先写失败测试，再写最小实现；每个任务完成后独立提交。
- 使用受试者级划分；同一受试者的窗口不得同时进入训练和测试。

---

## 文件结构总览

创建以下新文件：

```text
risk/pmcc/__init__.py
risk/pmcc/schema.py
risk/pmcc/baseline.py
risk/pmcc/changes.py
risk/pmcc/chains.py
risk/pmcc/features.py
risk/pmcc/survival.py
risk/pmcc/uncertainty.py
risk/pmcc/decisions.py
risk/pmcc/feedback.py
risk/pmcc/service.py
scripts/generate_pmcc_fixture.py
scripts/build_pmcc_dataset.py
scripts/train_pmcc.py
scripts/evaluate_pmcc.py
scripts/generate_pmcc_report.py
tests/risk/pmcc/test_schema.py
tests/risk/pmcc/test_baseline.py
tests/risk/pmcc/test_changes.py
tests/risk/pmcc/test_chains.py
tests/risk/pmcc/test_features.py
tests/risk/pmcc/test_survival.py
tests/risk/pmcc/test_uncertainty.py
tests/risk/pmcc/test_decisions.py
tests/risk/pmcc/test_feedback.py
tests/risk/pmcc/test_service.py
tests/integration/test_pmcc_offline_flow.py
```

已有文件只允许做兼容适配：

- `risk/personal_baseline.py`：保留 `RobustPersonalBaseline` 行为；新包通过适配器复用或转换其统计量。
- `risk/gait_stability.py`：只读取已有结果，不改变旧方法签名。
- `core/events.py`：不改变既有 `SensorEvent` 契约；PMCC 使用自己的日级 schema。
- `fusion/decision_engine.py`：只在后续集成任务中增加可选 `fall_forecast` 输入，不改变 `fall_event` 分支。

## Task 1: 严格 schema 与序列化

**Files:**

- Create: `risk/pmcc/schema.py`
- Create: `risk/pmcc/__init__.py`
- Create: `tests/risk/pmcc/test_schema.py`

**Interfaces:**

- Produce `DailyObservation`, `ChangeEvent`, `TemporalChain`, `PMCCForecast`, `OutcomeFeedback` dataclasses.
- Produce `EvidenceTier`, `BaselineState`, `DecisionBand`, `OutcomeType` enums.
- Every model exposes `to_dict()` and the top-level records expose `from_dict()`.

- [ ] **Step 1: Write failing schema tests**

测试必须覆盖：无时区日期、空 `subject_id`、质量超出 `[0,1]`、布尔值冒充质量、`available=false` 但质量非零、缺失特征保持为 `None`、JSON 往返保留 provenance。

```python
def test_daily_observation_rejects_boolean_quality():
    with pytest.raises(ValueError, match="quality"):
        DailyObservation(..., quality={"vision": True})

def test_round_trip_preserves_synthetic_provenance():
    original = make_observation(provenance={"evidence_tier": "synthetic_research"})
    assert DailyObservation.from_dict(original.to_dict()) == original
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `python -m pytest tests/risk/pmcc/test_schema.py -q`

Expected: FAIL because `risk.pmcc.schema` does not exist.

- [ ] **Step 3: Implement minimal strict dataclasses**

Use timezone-aware `datetime`/`date`, mapping copies, finite numeric checks, explicit enum parsing and immutable tuples. Never coerce boolean values to floats. Keep unknown provenance fields out of the model feature path but preserve the declared provenance mapping.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/risk/pmcc/test_schema.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add risk/pmcc/__init__.py risk/pmcc/schema.py tests/risk/pmcc/test_schema.py
git commit -m "feat: add PMCC data contracts"
```

## Task 2: 人群先验、三级冷启动与个人基线

**Files:**

- Create: `risk/pmcc/baseline.py`
- Create: `tests/risk/pmcc/test_baseline.py`
- Reference: `risk/personal_baseline.py`

**Interfaces:**

```python
class BaselineManager:
    def add(self, observation: DailyObservation) -> None: ...
    def state(self, feature: str) -> BaselineState: ...
    def directional_z(self, feature: str, value: float) -> float | None: ...
    def quality(self, feature: str) -> float: ...
    def explain_exclusion(self, day: date) -> str | None: ...
```

- [ ] **Step 1: Write failing baseline tests**

Cover 0/6/7/13/14 effective days, per-feature readiness, MAD-zero floor, abnormal/low-quality/fall days excluded, and no baseline contamination from a later day marked invalid.

```python
def test_each_feature_has_its_own_cold_start_state():
    manager = manager_with_sleep_days(14)
    manager.add(observation(features={"trunk_sway": 0.2}, available={"vision": True}))
    assert manager.state("sleep_duration_minutes") is BaselineState.PERSONAL
    assert manager.state("trunk_sway") is not BaselineState.PERSONAL
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/risk/pmcc/test_baseline.py -q`

Expected: FAIL because `BaselineManager` is not implemented.

- [ ] **Step 3: Implement the baseline manager**

Use a 30-valid-day update window, a 7-day transition into `blended`, 14 valid days for `personal`, median and `1.4826*MAD` with configured feature floor, and a separate valid-day count per feature. Exclude confirmed-fall/near-fall days plus three following days, quality `<0.5`, offline days, explicit environment-change days and composite anomaly days. Store exclusion reasons.

- [ ] **Step 4: Run focused and legacy baseline tests**

Run: `python -m pytest tests/risk/pmcc/test_baseline.py tests/risk/test_personal_baseline.py -q`

Expected: PASS; legacy tests must remain unchanged and green.

- [ ] **Step 5: Commit**

```powershell
git add risk/pmcc/baseline.py tests/risk/pmcc/test_baseline.py
git commit -m "feat: add staged personal PMCC baselines"
```

## Task 3: 变化事件与 72 小时时间关联链

**Files:**

- Create: `risk/pmcc/changes.py`
- Create: `risk/pmcc/chains.py`
- Create: `tests/risk/pmcc/test_changes.py`
- Create: `tests/risk/pmcc/test_chains.py`

**Interfaces:**

```python
def detect_changes(observations: Sequence[DailyObservation], baseline: BaselineManager) -> tuple[ChangeEvent, ...]: ...

class TemporalChainBuilder:
    def build(self, events: Sequence[ChangeEvent]) -> tuple[TemporalChain, ...]: ...
```

- [ ] **Step 1: Write failing event and chain tests**

Test 2-of-3 persistence, missing days not treated as normal, reverse order rejected, gaps over 72 hours rejected, low quality lowering the chain score, and output retaining node IDs/gaps/`association_only=true`.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `python -m pytest tests/risk/pmcc/test_changes.py tests/risk/pmcc/test_chains.py -q`

Expected: FAIL because the new modules do not exist.

- [ ] **Step 3: Implement changes and registered edges**

Use directional z thresholds `1.5` for ordinary and `2.5` for severe changes. Register only these edges: sleep→activity, sleep→gait, activity→sit-to-stand, sit-to-stand→near-fall, gait→near-fall, physiology→sleep. Calculate each edge as `w*persistence_a*persistence_b*quality_a*quality_b*exp(-delta_hours/48)`, combine into `[0,1]`, and never call the result causal.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/risk/pmcc/test_changes.py tests/risk/pmcc/test_chains.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add risk/pmcc/changes.py risk/pmcc/chains.py tests/risk/pmcc/test_changes.py tests/risk/pmcc/test_chains.py
git commit -m "feat: add PMCC temporal change chains"
```

## Task 4: 14 日特征窗口与离散生存数学

**Files:**

- Create: `risk/pmcc/features.py`
- Create: `risk/pmcc/survival.py`
- Create: `tests/risk/pmcc/test_features.py`
- Create: `tests/risk/pmcc/test_survival.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class FeatureWindow:
    values: tuple[tuple[float, ...], ...]
    missing_mask: tuple[tuple[bool, ...], ...]
    quality: tuple[tuple[float, ...], ...]
    feature_names: tuple[str, ...]

@dataclass(frozen=True)
class SurvivalLabel:
    event_day: int | None
    censor_day: int

    def __post_init__(self) -> None:
        if not 1 <= self.censor_day <= 7:
            raise ValueError("censor_day must be in [1, 7]")
        if self.event_day is not None and not 1 <= self.event_day <= self.censor_day:
            raise ValueError("event_day must be within the observed censoring window")

def build_feature_window(observations: Sequence[DailyObservation], baseline: BaselineManager, chains: Sequence[TemporalChain], as_of: date) -> FeatureWindow: ...
def hazards_to_cumulative(hazards: Sequence[float]) -> tuple[float, ...]: ...
def cumulative_risk_for_horizons(hazards: Sequence[float]) -> dict[str, float]: ...
```

- [ ] **Step 1: Write failing feature/survival tests**

Cover fixed 14-day shape, explicit missing mask, no zero-as-normal behavior, 7 hazard outputs in `[0,1]`, cumulative monotonicity, and horizon mapping `24h→1`, `72h→3`, `7d→7`.

```python
def test_cumulative_risk_is_monotonic():
    values = hazards_to_cumulative((0.1, 0.2, 0.05, 0.0, 0.1, 0.0, 0.0))
    assert list(values) == sorted(values)
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `python -m pytest tests/risk/pmcc/test_features.py tests/risk/pmcc/test_survival.py -q`

Expected: FAIL because the feature and survival modules do not exist.

- [ ] **Step 3: Implement fixed-shape feature window and pure survival functions**

Use date-aware ordering, pad only outside the requested 14-day window with mask=true, preserve quality separately, and reject windows with ambiguous timezone/date ownership. `hazards_to_cumulative` must validate exactly seven finite probabilities and calculate `1-product(1-h_k)` without any model dependency.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/risk/pmcc/test_features.py tests/risk/pmcc/test_survival.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add risk/pmcc/features.py risk/pmcc/survival.py tests/risk/pmcc/test_features.py tests/risk/pmcc/test_survival.py
git commit -m "feat: add PMCC feature windows and survival math"
```

## Task 5: CPU 校准器、可选 TCN 与模型卡

**Files:**

- Modify: `risk/pmcc/survival.py`
- Create: `tests/risk/pmcc/test_survival_models.py`
- Reference: `risk/prefall_model.py`

**Interfaces:**

```python
class SurvivalRiskModel(Protocol):
    def predict_hazards(self, features: FeatureWindow) -> Sequence[float]: ...

class RuleSurvivalCalibrator:
    def fit(self, windows: Sequence[FeatureWindow], labels: Sequence[SurvivalLabel]) -> "RuleSurvivalCalibrator": ...
    def predict_hazards(self, features: FeatureWindow) -> tuple[float, ...]: ...
    def metadata(self) -> dict[str, object]: ...

class OptionalTCNSurvivalModel:
    @classmethod
    def available(cls) -> bool: ...
```

- [ ] **Step 1: Write failing model tests**

Test that the CPU model imports without scikit-learn, refuses one-class labels, preserves feature schema, emits seven hazards, marks synthetic artifacts `promoted=false`, and that `OptionalTCNSurvivalModel.available()` returns a boolean without importing torch at module import time.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/risk/pmcc/test_survival_models.py -q`

Expected: FAIL because the model classes are not present.

- [ ] **Step 3: Implement CPU model and optional dependency gate**

The CPU path uses engineered rule/chain features plus one-vs-horizon logistic calibration with `class_weight="balanced"`, fixed seed and strict feature names. Training imports scikit-learn lazily. The optional TCN class must raise a clear runtime error only when training/inference is explicitly requested without torch; ordinary imports and the CPU path continue to work. Do not claim TCN performance without a real longitudinal evaluation artifact.

- [ ] **Step 4: Run model and existing pre-fall tests**

Run: `python -m pytest tests/risk/pmcc/test_survival_models.py tests/risk/test_prefall_model.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add risk/pmcc/survival.py tests/risk/pmcc/test_survival_models.py
git commit -m "feat: add CPU PMCC survival calibrator"
```

## Task 6: 不确定性、拒绝门控、风险决策与反馈

**Files:**

- Create: `risk/pmcc/uncertainty.py`
- Create: `risk/pmcc/decisions.py`
- Create: `risk/pmcc/feedback.py`
- Create: `tests/risk/pmcc/test_uncertainty.py`
- Create: `tests/risk/pmcc/test_decisions.py`
- Create: `tests/risk/pmcc/test_feedback.py`

**Interfaces:**

```python
class UncertaintyGate:
    def evaluate(self, forecasts: Sequence[Sequence[float]], coverage: float, quality: float, evidence_tier: EvidenceTier, release_mode: bool) -> UncertaintySummary: ...

@dataclass(frozen=True)
class UncertaintySummary:
    lower: tuple[float, ...]
    median: tuple[float, ...]
    upper: tuple[float, ...]
    width_72h: float
    abstained: bool
    reasons: tuple[str, ...]

@dataclass(frozen=True)
class DecisionSummary:
    decision: str
    forecast_band: str
    abstained: bool
    reasons: tuple[str, ...]

def make_forecast_decision(risk: Mapping[str, float], uncertainty: UncertaintySummary, baseline_state: BaselineState, evidence_groups: int, promoted: bool) -> DecisionSummary: ...

class FeedbackStore:
    def append(self, feedback: OutcomeFeedback) -> None: ...
    def read(self, forecast_id: str | None = None) -> tuple[OutcomeFeedback, ...]: ...
```

- [ ] **Step 1: Write failing safety tests**

Cover coverage/quality `<0.5`, interval width `>0.35`, synthetic release mode, missing visual evidence group, population-only capping at `watch`, `forecast_band=very_high` not mapping to `critical`, and append-only feedback where `unknown` is not a training label.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/risk/pmcc/test_uncertainty.py tests/risk/pmcc/test_decisions.py tests/risk/pmcc/test_feedback.py -q`

Expected: FAIL because the safety and feedback modules do not exist.

- [ ] **Step 3: Implement bootstrap summary, refusal rules and feedback JSONL**

Bootstrap default uses at least five forecast members and 10%–90% quantiles. Refusal returns a structured forecast with `abstained=true`, reason and no high level. Use existing risk levels `info/watch/warning`; add `forecast_band` for low/elevated/high/very_high. Never emit `critical` from U-PMCC. Feedback records must append new JSONL lines and preserve original forecast records.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/risk/pmcc/test_uncertainty.py tests/risk/pmcc/test_decisions.py tests/risk/pmcc/test_feedback.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add risk/pmcc/uncertainty.py risk/pmcc/decisions.py risk/pmcc/feedback.py tests/risk/pmcc/test_uncertainty.py tests/risk/pmcc/test_decisions.py tests/risk/pmcc/test_feedback.py
git commit -m "feat: add PMCC uncertainty and feedback safeguards"
```

## Task 7: PMCCService 编排与既有模块适配

**Files:**

- Create: `risk/pmcc/service.py`
- Create: `tests/risk/pmcc/test_service.py`
- Create: `tests/integration/test_pmcc_offline_flow.py`
- Read only: `fusion/decision_engine.py`; Task 7 must prove that existing `fall_event` behavior is unchanged. A forecast adapter is not required for the first implementation and must not be added unless a focused integration test demonstrates a concrete contract need.

**Interfaces:**

```python
class PMCCService:
    def observe(self, observation: DailyObservation) -> None: ...
    def forecast(self, subject_id: str, as_of: date, release_mode: bool = False) -> PMCCForecast: ...
    def record_feedback(self, feedback: OutcomeFeedback) -> None: ...
```

- [ ] **Step 1: Write failing service tests**

Test deterministic repeated forecasts, 14-day chain progression, removal of physiology while retaining vision/activity, model-load failure fallback, synthetic release-mode refusal, and no writes when feedback storage is disabled.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/risk/pmcc/test_service.py tests/integration/test_pmcc_offline_flow.py -q`

Expected: FAIL because `PMCCService` does not exist.

- [ ] **Step 3: Implement orchestration**

`observe` validates and stores the observation, updates only eligible baseline state, computes changes/chains lazily, and never calls the network. `forecast` builds the fixed window, invokes the configured model, applies uncertainty/decision gates, and returns provenance. If a model raises during inference, use the CPU rule path and record a degradation reason. The service must not instantiate C6c or SDNL1 clients.

- [ ] **Step 4: Run integration and legacy tests**

Run: `python -m pytest tests/risk/pmcc tests/integration/test_pmcc_offline_flow.py tests/fusion/test_decision_engine.py -q`

Expected: PASS; existing `fall_event` behavior remains unchanged.

- [ ] **Step 5: Commit**

```powershell
git add risk/pmcc/service.py tests/risk/pmcc/test_service.py tests/integration/test_pmcc_offline_flow.py
git commit -m "feat: add PMCC offline service"
```

## Task 8: 合成夹具、数据构建与训练脚本

**Files:**

- Create: `scripts/generate_pmcc_fixture.py`
- Create: `scripts/build_pmcc_dataset.py`
- Create: `scripts/train_pmcc.py`
- Create: `tests/integration/test_pmcc_scripts.py`

**Interfaces:**

```text
python scripts/generate_pmcc_fixture.py --output outputs/pmcc-fixture.jsonl --seed 42
python scripts/build_pmcc_dataset.py --input path/to/source --output outputs/pmcc-dataset.jsonl --evidence-tier real_public
python scripts/train_pmcc.py --input outputs/pmcc-fixture.jsonl --output outputs/pmcc-model --seed 42
```

- [ ] **Step 1: Write failing script tests**

Test deterministic fixture generation, required scenario names (`normal`, `sleep_activity_gait_cascade`, `missing_physiology`, `timestamp_shift`), provenance flags, schema validation, and training output metadata with `promoted=false` for synthetic input.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/integration/test_pmcc_scripts.py -q`

Expected: FAIL because the scripts do not exist.

- [ ] **Step 3: Implement scripts**

The fixture generator writes only JSONL and never contacts devices. The dataset builder refuses mixed evidence tiers unless explicitly given one tier and writes `release_id`, `dataset_id`, `subject_split_id` and `schema_version`. The trainer refuses missing labels, writes model card/metrics/provenance, and never sets `promoted=true` for synthetic or offline input.

- [ ] **Step 4: Run script smoke tests**

Run:

```powershell
python scripts/generate_pmcc_fixture.py --output outputs/pmcc-fixture.jsonl --seed 42
python scripts/train_pmcc.py --input outputs/pmcc-fixture.jsonl --output outputs/pmcc-model --seed 42
python -m pytest tests/integration/test_pmcc_scripts.py -q
```

Expected: commands exit 0; model card states synthetic/non-clinical/not promoted.

- [ ] **Step 5: Commit**

```powershell
git add scripts/generate_pmcc_fixture.py scripts/build_pmcc_dataset.py scripts/train_pmcc.py tests/integration/test_pmcc_scripts.py
git commit -m "feat: add PMCC offline data and training scripts"
```

## Task 9: 受试者级评估、消融报告与最终回归

**Files:**

- Create: `scripts/evaluate_pmcc.py`
- Create: `scripts/generate_pmcc_report.py`
- Create: `tests/integration/test_pmcc_evaluation.py`
- Modify: `docs/evaluation-report.md` only to link the PMCC report and its evidence boundary.

**Interfaces:**

```text
python scripts/evaluate_pmcc.py --input outputs/pmcc-dataset.jsonl --model outputs/pmcc-model --output outputs/pmcc-evaluation.json
python scripts/generate_pmcc_report.py --metrics outputs/pmcc-evaluation.json --output docs/pmcc-evaluation-report.md
```

- [ ] **Step 1: Write failing evaluation tests**

Test subject-group splits, same-release provenance enforcement, baseline-vs-full model rows, required ablations, monotonic horizon checks, abstention coverage, and refusal to pass synthetic-only metrics as release evidence.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/integration/test_pmcc_evaluation.py -q`

Expected: FAIL because the evaluation scripts do not exist.

- [ ] **Step 3: Implement evaluation and report generation**

Use grouped outer splits and inner calibration only. Report time-dependent AUC/AUPRC, Brier, calibration, fixed alert-budget recall/precision, false alarms per subject-day, lead time, abstention coverage and stratified quality/evidence results. Include ablations for baseline, chains, sleep/physiology, gait, mask, quality gate, uncertainty and TCN. If evidence tier is synthetic/offline, report `research_only` and exit non-zero for release mode.

- [ ] **Step 4: Run all PMCC and repository tests**

Run:

```powershell
python -m pytest tests/risk/pmcc tests/integration/test_pmcc_offline_flow.py tests/integration/test_pmcc_scripts.py tests/integration/test_pmcc_evaluation.py -q
python -m pytest -q
```

Expected: all new PMCC tests pass. The known pre-existing `GMDCSA24.expected_size_bytes=null` manifest failure must be reported separately and not hidden.

- [ ] **Step 5: Commit**

```powershell
git add scripts/evaluate_pmcc.py scripts/generate_pmcc_report.py tests/integration/test_pmcc_evaluation.py docs/evaluation-report.md docs/pmcc-evaluation-report.md
git commit -m "feat: add PMCC evaluation and audit report"
```

## Final Verification Checklist

- [ ] `python -m pytest tests/risk/pmcc tests/integration/test_pmcc_offline_flow.py tests/integration/test_pmcc_scripts.py tests/integration/test_pmcc_evaluation.py -q` passes.
- [ ] `python scripts/generate_pmcc_fixture.py --output outputs/pmcc-fixture.jsonl --seed 42` is deterministic.
- [ ] Missing physiology path retains vision/activity and never contacts a device.
- [ ] Any refusal/low quality result contains a machine-readable reason.
- [ ] All forecast horizons are monotonic and all probabilities are finite `[0,1]`.
- [ ] No synthetic or offline artifact is promoted.
- [ ] No `critical` decision is emitted by U-PMCC.
- [ ] Existing fall-event, alert, mental-health and device tests remain compatible.
- [ ] `git diff --check` passes and no secrets or absolute paths are added.
- [ ] Full-suite pre-existing manifest failure is documented rather than silently fixed with invented dataset size.
