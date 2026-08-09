# RG-PCNet Continuous Event Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate RG-PCNet as a timestamped continuous monitoring system and connect its calibrated, reliability-gated decisions to alert events without clip-boundary leakage.

**Architecture:** Add a time-based RG-PCNet event decoder that consumes calibrated frame predictions and produces suspected, confirmed, postfall, recovered, or abstained states. Add event matching and false-alarms-per-hour metrics, then a replay runner that processes chronological streams without exposing clip boundaries. Feed the same decoder configuration into live service and release promotion.

**Tech Stack:** Python 3.10 standard library, NumPy, pytest, existing event and service contracts, Matplotlib for final plots.

## Global Constraints

- All timers use timestamps in seconds, never frame counts.
- Replay input is chronological and model-visible records contain no clip-boundary marker.
- Ground-truth boundaries remain available only to the evaluator after predictions are complete.
- Reliability below the release gate produces `abstain`; it never produces a confirmed fall.
- An alert requires temporal evidence and is emitted once per event until recovery/cooldown re-arms the decoder.
- Report event recall, event precision, false alerts per hour, median/P90 alert delay, duplicate alert rate, recovery accuracy, coverage, and AURC.
- Offline replay and live inference use the same `RGPCReleaseConfig` timer fields.
- All chart values come from machine-readable JSON; no metric is manually entered into a plotting script.
- Existing `FallStateMachine` remains available for the baseline and is not silently redefined.
- Before changing shared service/event symbols, run GitNexus impact and inspect HIGH/CRITICAL results.
- Every production change follows RED → GREEN → REFACTOR.

---

## File Structure

- Create `risk/phase_model/event_state.py`: timestamp-based decoder and immutable state transitions.
- Create `risk/phase_model/event_evaluation.py`: interval matching and continuous monitoring metrics.
- Create `risk/phase_model/continuous_replay.py`: ordered replay engine independent of model implementation.
- Create `scripts/evaluate_continuous_rg_pcnet.py`: JSONL CLI and artifact writer.
- Modify `pipeline/vision_phase_source.py`: allow an RG-PCNet service configured from one release directory.
- Modify `pipeline/live_service.py`: preserve abstention quality and event IDs.
- Create `scripts/plot_rg_pcnet_results.py`: plots derived only from evaluation JSON.
- Add unit and integration tests for each boundary.

### Task 1: Implement a timestamp-based event decoder

**Files:**
- Create: `risk/phase_model/event_state.py`
- Test: `tests/risk/phase_model/test_event_state.py`

**Interfaces:**
- Consumes: `FrameDecision(timestamp, fall_probability, phase, reliable)` and `EventDecoderConfig`.
- Produces: `EventTransition(state, event_id, emitted, reason, timestamp)`.

- [ ] **Step 1: Write failing state and time tests**

Name the breaks: replacing seconds with frame counts, confirming during abstention, or emitting duplicates must fail.

```python
from risk.phase_model.event_state import EventDecoder, EventDecoderConfig, FrameDecision


def _frame(t, p, phase="normal", reliable=True):
    return FrameDecision(float(t), float(p), phase, reliable)


def test_confirmation_depends_on_elapsed_seconds_not_frame_count():
    decoder = EventDecoder(EventDecoderConfig(0.6, 0.8, 2.0, 10.0))
    assert decoder.update(_frame(0.0, 0.8, "descent_or_impact")).state == "suspected"
    assert decoder.update(_frame(0.2, 0.8, "descent_or_impact")).state == "suspected"
    result = decoder.update(_frame(0.81, 0.8, "descent_or_impact"))
    assert result.state == "confirmed"
    assert result.emitted == "fall_confirmed"


def test_abstention_clears_unconfirmed_evidence_and_never_confirms():
    decoder = EventDecoder(EventDecoderConfig(0.6, 0.8, 2.0, 10.0))
    decoder.update(_frame(0.0, 0.9, "descent_or_impact"))
    result = decoder.update(_frame(1.0, 0.9, "descent_or_impact", reliable=False))
    assert result.state == "abstain"
    assert result.emitted is None
    assert decoder.update(_frame(1.1, 0.9, "descent_or_impact")).state == "suspected"


def test_confirmed_event_emits_once_until_recovery():
    decoder = EventDecoder(EventDecoderConfig(0.6, 0.5, 1.0, 2.0))
    decoder.update(_frame(0.0, 0.9, "descent_or_impact"))
    first = decoder.update(_frame(0.6, 0.9, "postfall_or_recovery"))
    second = decoder.update(_frame(0.8, 0.9, "postfall_or_recovery"))
    assert first.emitted == "fall_confirmed"
    assert second.emitted is None
    assert first.event_id == second.event_id
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/risk/phase_model/test_event_state.py -q`.

- [ ] **Step 3: Implement explicit transitions**

Use these immutable contracts:

```python
@dataclass(frozen=True)
class EventDecoderConfig:
    fall_threshold: float
    confirm_seconds: float
    recovery_seconds: float
    cooldown_seconds: float


@dataclass(frozen=True)
class FrameDecision:
    timestamp: float
    fall_probability: float
    phase: str
    reliable: bool


@dataclass(frozen=True)
class EventTransition:
    state: str
    event_id: str | None
    emitted: str | None
    reason: str
    timestamp: float
```

Allowed states are `monitoring`, `suspected`, `confirmed`, `postfall`, `recovered`, and `abstain`. Validate strictly increasing finite timestamps and probability `[0,1]`. Start suspicion only when reliable and either probability reaches threshold or phase is `descent_or_impact`. Confirm when suspicion duration reaches `confirm_seconds`. Enter postfall when confirmed and phase is `postfall_or_recovery`. Recover only after continuous normal evidence for `recovery_seconds`; emit `fall_recovered` once. Cooldown prevents a new event until `cooldown_seconds` after recovery. Generate event IDs as `fall-{sequence:06d}`.

- [ ] **Step 4: Verify GREEN**

```powershell
python -m pytest tests/risk/phase_model/test_event_state.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add risk/phase_model/event_state.py tests/risk/phase_model/test_event_state.py
git commit -m "feat: decode rgpc fall events by timestamp"
```

### Task 2: Match predicted and true event intervals

**Files:**
- Create: `risk/phase_model/event_evaluation.py`
- Test: `tests/risk/phase_model/test_event_evaluation.py`

**Interfaces:**
- Consumes: truth intervals, predicted alerts, monitoring duration, frame decisions.
- Produces: `ContinuousMetrics` and risk-coverage points.

- [ ] **Step 1: Write failing hand-calculated metric tests**

```python
import pytest

from risk.phase_model.event_evaluation import Alert, TruthEvent, evaluate_continuous_events


def test_event_metrics_are_hand_auditable():
    truth = [TruthEvent("t1", 10.0, 15.0), TruthEvent("t2", 40.0, 45.0)]
    alerts = [Alert("a1", 11.0), Alert("a2", 12.0), Alert("a3", 70.0)]
    result = evaluate_continuous_events(truth, alerts, duration_seconds=3600.0, tolerance_seconds=5.0)
    assert result.true_events == 2
    assert result.matched_events == 1
    assert result.event_recall == 0.5
    assert result.event_precision == pytest.approx(1 / 3)
    assert result.false_alerts_per_hour == 2.0
    assert result.duplicate_alert_rate == pytest.approx(0.5)
    assert result.median_delay_seconds == 1.0


def test_alert_matching_is_one_to_one():
    truth = [TruthEvent("t1", 10.0, 15.0)]
    alerts = [Alert("a1", 11.0), Alert("a2", 12.0)]
    result = evaluate_continuous_events(truth, alerts, duration_seconds=100.0, tolerance_seconds=0.0)
    assert result.matched_events == 1
    assert result.duplicate_alerts == 1
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/risk/phase_model/test_event_evaluation.py -q`.

- [ ] **Step 3: Implement chronological one-to-one matching**

Sort truth and alerts by time. An alert matches the earliest unmatched truth where `truth.start - tolerance <= alert.timestamp <= truth.end + tolerance`. Extra alerts inside an already matched interval count as duplicates; all other unmatched alerts count as false alerts. Delay is `alert.timestamp - truth.start`, preserving negative early alerts. Event precision uses matched unique events divided by total alerts. Return `"unavailable"` only for delay percentiles when there are no matches; all rates remain numeric.

Use a frozen `ContinuousMetrics` dataclass whose `to_dict()` contains exactly:

```python
{
    "true_events", "matched_events", "alerts", "false_alerts", "duplicate_alerts",
    "event_recall", "event_precision", "false_alerts_per_hour", "duplicate_alert_rate",
    "median_delay_seconds", "p90_delay_seconds", "monitoring_hours",
}
```

- [ ] **Step 4: Verify GREEN**

```powershell
python -m pytest tests/risk/phase_model/test_event_evaluation.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add risk/phase_model/event_evaluation.py tests/risk/phase_model/test_event_evaluation.py
git commit -m "feat: measure continuous fall event performance"
```

### Task 3: Replay chronological streams without boundary leakage

**Files:**
- Create: `risk/phase_model/continuous_replay.py`
- Test: `tests/risk/phase_model/test_continuous_replay.py`

**Interfaces:**
- Consumes: ordered `ReplayFrame` items and an object implementing `predict(frame) -> FrameDecision`.
- Produces: `ReplayResult(transitions, alerts, coverage, abstained_frames, total_frames)`.

- [ ] **Step 1: Write failing ordering and boundary tests**

```python
import pytest

from risk.phase_model.continuous_replay import ReplayFrame, replay_stream


def test_replay_rejects_nonmonotonic_timestamps():
    frames = [ReplayFrame(1.0, "x"), ReplayFrame(0.5, "y")]
    with pytest.raises(ValueError, match="strictly increasing"):
        replay_stream(frames, predictor=lambda frame: frame.payload, decoder=_decoder())


def test_replay_payload_excludes_clip_boundary_metadata():
    frames = [ReplayFrame(0.0, {"pose": [1], "clip_id": "hidden"})]
    seen = []
    def predictor(frame):
        seen.append(frame.payload)
        return _normal_decision(frame.timestamp)
    replay_stream(frames, predictor=predictor, decoder=_decoder())
    assert seen == [{"pose": [1]}]
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/risk/phase_model/test_continuous_replay.py -q`.

- [ ] **Step 3: Implement replay with evaluator-only metadata**

`ReplayFrame` stores `timestamp`, `payload`, and optional `truth_event_id`; before predictor invocation, create a view whose payload excludes `clip_id`, `clip_boundary`, `truth_event_id`, `truth_start`, and `truth_end`. Feed every prediction to `EventDecoder`. Coverage is reliable predictions divided by total predictions. Persist all transitions, including abstentions, so coverage and audit plots remain reproducible.

- [ ] **Step 4: Verify GREEN**

```powershell
python -m pytest tests/risk/phase_model/test_continuous_replay.py tests/risk/phase_model/test_event_state.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add risk/phase_model/continuous_replay.py tests/risk/phase_model/test_continuous_replay.py
git commit -m "feat: replay continuous pose streams without boundaries"
```

### Task 4: Add continuous RG-PCNet evaluation artifacts

**Files:**
- Create: `scripts/evaluate_continuous_rg_pcnet.py`
- Test: `tests/integration/test_continuous_rg_pcnet.py`

**Interfaces:**
- Consumes: prediction JSONL, truth-event JSON, `release_config.json`.
- Produces: `continuous_metrics.json`, `transitions.jsonl`, `alerts.jsonl`, and `promotion_gate.json`.

- [ ] **Step 1: Write failing end-to-end artifact test**

The fixture contains one fall event, one matching alert, 60 seconds of monitoring, and 10% abstained frames. Assert:

```python
metrics = json.loads((output / "continuous_metrics.json").read_text(encoding="utf-8"))
gate = json.loads((output / "promotion_gate.json").read_text(encoding="utf-8"))
assert metrics["event_recall"] == 1.0
assert metrics["false_alerts_per_hour"] == 0.0
assert metrics["coverage"] == pytest.approx(0.9)
assert gate["passed"] is True
assert gate["release_id"] == "r1"
assert len(gate["input_sha256"]) == 64
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/integration/test_continuous_rg_pcnet.py -q`.

- [ ] **Step 3: Implement canonical CLI and promotion gate**

Arguments are `--predictions`, `--truth-events`, `--release-config`, `--output`, `--minimum-event-recall`, `--maximum-false-alerts-per-hour`, and `--minimum-coverage`. Validate that prediction timestamps are strictly increasing and release ID is consistent. Gate passes only when all three numeric constraints pass. Write sorted UTF-8 JSON with one terminal newline and JSONL in timestamp order. Store SHA-256 for every input and output artifact.

- [ ] **Step 4: Verify GREEN**

```powershell
python -m pytest tests/integration/test_continuous_rg_pcnet.py tests/risk/phase_model/test_event_evaluation.py tests/risk/phase_model/test_continuous_replay.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add scripts/evaluate_continuous_rg_pcnet.py tests/integration/test_continuous_rg_pcnet.py
git commit -m "feat: emit verifiable continuous evaluation artifacts"
```

### Task 5: Use the release decoder in the live monitoring path

**Files:**
- Modify: `pipeline/vision_phase_source.py`
- Modify: `pipeline/live_service.py:82-155`
- Test: `tests/pipeline/test_vision_phase_source.py`
- Test: `tests/pipeline/test_live_service.py`
- Test: `tests/integration/test_monitoring_flow.py`

**Interfaces:**
- Consumes: RG-PCNet release directory and live `SensorEvent` batches.
- Produces: stable fall-confirmed/recovered events with abstention health state.

- [ ] **Step 1: Write failing integration tests**

```python
def test_live_rgpc_source_loads_event_timers_from_release(release_dir):
    source = VisionPhaseSource.from_rgpc_release(release_dir, device="cpu")
    assert source.decoder.config.confirm_seconds == 0.8
    assert source.decoder.config.recovery_seconds == 2.0


def test_live_service_preserves_abstention_without_dispatching_alert(live_service, abstained_batch):
    result = live_service.step(abstained_batch)
    assert result.system_health == "degraded"
    assert not any(event.event_type is EventType.FALL_EVENT for event in result.events)
```

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/pipeline/test_vision_phase_source.py tests/pipeline/test_live_service.py tests/integration/test_monitoring_flow.py -q
```

- [ ] **Step 3: Add an explicit RG-PCNet constructor and event mapping**

`VisionPhaseSource.from_rgpc_release` loads `RGPCReleaseConfig`, `RGPredictor`, and `EventDecoder` from the same directory. It converts predictor outputs to `FrameDecision`; `fall_decision is None` sets `reliable=False`. Confirmed and recovered transitions become existing `FALL_EVENT` events with stable `event_id` and boolean payload fields. Abstention becomes a POSE event whose quality reason is `model_reliability_gate`.

In `LiveMonitoringService`, preserve the incoming event ID instead of generating a second one, and treat reliability abstention as degraded health rather than failure. Do not dispatch notifications for suspected or abstained transitions.

- [ ] **Step 4: Verify GREEN and alert regressions**

```powershell
python -m pytest tests/pipeline/test_vision_phase_source.py tests/pipeline/test_live_service.py tests/integration/test_monitoring_flow.py tests/alerts/test_dispatcher.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add pipeline/vision_phase_source.py pipeline/live_service.py tests/pipeline/test_vision_phase_source.py tests/pipeline/test_live_service.py tests/integration/test_monitoring_flow.py
git commit -m "feat: run release event decoder in live monitoring"
```

### Task 6: Generate traceable comparison and reliability plots

**Files:**
- Create: `scripts/plot_rg_pcnet_results.py`
- Test: `tests/integration/test_rg_pcnet_plots.py`

**Interfaces:**
- Consumes: aggregate LOSO JSON, continuous metrics JSON, calibration JSON, ablation JSON.
- Produces: PNG and SVG versions of four fixed plots plus `plot_manifest.json`.

- [ ] **Step 1: Write failing source-traceability test**

```python
def test_plot_manifest_hashes_every_metric_source(tmp_path, evaluation_artifacts):
    result = plot_rg_pcnet_results(**evaluation_artifacts, output_dir=tmp_path)
    manifest = json.loads((tmp_path / "plot_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["figures"]) == {
        "model_comparison", "subject_f1", "risk_coverage", "continuous_events"
    }
    assert all(len(item["source_sha256"]) == 64 for item in manifest["figures"].values())
    assert all(Path(path).exists() for path in result.values())
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/integration/test_rg_pcnet_plots.py -q`.

- [ ] **Step 3: Implement four fixed, data-driven figures**

Generate:

1. model comparison: baseline TCN, phase-only, reliability-only, full RG-PCNet, optional distillation;
2. per-subject F1 with a subject-macro line;
3. risk-coverage curves for clean and corrupted inputs;
4. continuous metrics: event recall, false alerts/hour, median delay, and coverage.

Reject missing metric keys instead of substituting zeros. Titles must state `LOSO` or `continuous replay`; captions stored in the manifest must state subject count and whether intervals are descriptive. Save `dpi=200` PNG and vector SVG from the same figure object.

- [ ] **Step 4: Verify GREEN and render files**

```powershell
python -m pytest tests/integration/test_rg_pcnet_plots.py -q
```

Open the generated PNG files and verify axes, legends, Chinese font fallback, and non-overlapping labels before publication.

- [ ] **Step 5: Commit**

```powershell
git add scripts/plot_rg_pcnet_results.py tests/integration/test_rg_pcnet_plots.py
git commit -m "feat: plot traceable rgpcnet evaluation results"
```

### Task 7: Close the promotion loop

**Files:**
- Modify: `scripts/evaluate_rg_pcnet_loso.py`
- Test: `tests/integration/test_rg_pcnet_promotion.py`

**Interfaces:**
- Consumes: nested LOSO summary and continuous `promotion_gate.json`.
- Produces: final release directory only if all gates and hashes agree.

- [ ] **Step 1: Write failing cross-artifact gate tests**

```python
def test_release_is_not_promoted_when_continuous_gate_fails(candidate_artifacts):
    candidate_artifacts.continuous_gate["passed"] = False
    result = finalize_rgpc_release(candidate_artifacts)
    assert result["promoted"] is False
    assert "continuous" in result["reasons"]


def test_release_requires_matching_release_and_input_hashes(candidate_artifacts):
    candidate_artifacts.continuous_gate["release_id"] = "another-release"
    with pytest.raises(ValueError, match="release_id mismatch"):
        finalize_rgpc_release(candidate_artifacts)
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/integration/test_rg_pcnet_promotion.py -q`.

- [ ] **Step 3: Implement final conjunctive gate**

Require all of: nested selection feasible, macro event F1 improved, recall floor met, false alerts/hour not degraded, bounded calibration valid, minimum coverage met, AURC improved, three seeds present, continuous gate passed, checkpoint/config hashes valid, and release IDs identical. Return all failed reasons; never stop at the first failed metric. Only a passing result writes `release/`.

- [ ] **Step 4: Verify GREEN and full project regression**

```powershell
python -m pytest tests/risk/phase_model tests/pipeline tests/alerts tests/integration/test_train_rg_pcnet.py tests/integration/test_rg_pcnet_loso_release.py tests/integration/test_continuous_rg_pcnet.py tests/integration/test_rg_pcnet_promotion.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add scripts/evaluate_rg_pcnet_loso.py tests/integration/test_rg_pcnet_promotion.py
git commit -m "feat: require continuous evidence for rgpc promotion"
```

### Task 8: Run the fixed-seed ablation and cross-domain matrix

**Files:**
- Create: `risk/phase_model/experiment_matrix.py`
- Create: `scripts/run_rg_pcnet_experiment_matrix.py`
- Test: `tests/risk/phase_model/test_experiment_matrix.py`
- Test: `tests/integration/test_rg_pcnet_experiment_matrix.py`

**Interfaces:**
- Consumes: frozen LOSO manifest, dataset lock, baseline paths, output root.
- Produces: immutable run specifications and `experiment_matrix.json` aggregating real run artifacts.

- [ ] **Step 1: Write failing matrix-completeness tests**

```python
from risk.phase_model.experiment_matrix import build_experiment_matrix


def test_matrix_contains_fixed_ablations_seeds_and_outer_subjects():
    matrix = build_experiment_matrix(outer_subjects=("s1", "s2", "s3", "s4"), seeds=(41, 42, 43))
    assert {run.variant for run in matrix} == {
        "fall_only", "phase", "phase_transition", "reliability",
        "full_rgpc", "full_rgpc_teacher", "feature_exclusion",
    }
    assert {run.seed for run in matrix} == {41, 42, 43}
    assert {run.outer_subject for run in matrix} == {"s1", "s2", "s3", "s4"}
    assert len(matrix) == 7 * 3 * 4


def test_cross_domain_runs_never_train_on_held_out_dataset():
    matrix = build_experiment_matrix(
        outer_subjects=("s1",), seeds=(42,),
        train_datasets=("GMDCSA24", "URFD"), held_out_dataset="OmniFall",
    )
    assert all("OmniFall" not in run.train_datasets for run in matrix)
    assert all(run.held_out_dataset == "OmniFall" for run in matrix)
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/risk/phase_model/test_experiment_matrix.py -q`.

- [ ] **Step 3: Implement immutable run specifications and real-artifact aggregation**

Define `ExperimentRun` with `variant`, `seed`, `outer_subject`, `train_datasets`, `held_out_dataset`, `config_overrides`, and deterministic `run_id`. Variant overrides are fixed:

```python
VARIANTS = {
    "fall_only": {"phase_weight": 0.0, "transition_weight": 0.0, "reliability_weight": 0.0, "teacher_weight": 0.0},
    "phase": {"phase_weight": 0.5, "transition_weight": 0.0, "reliability_weight": 0.0, "teacher_weight": 0.0},
    "phase_transition": {"phase_weight": 0.5, "transition_weight": 0.1, "reliability_weight": 0.0, "teacher_weight": 0.0},
    "reliability": {"phase_weight": 0.0, "transition_weight": 0.0, "reliability_weight": 0.3, "teacher_weight": 0.0},
    "full_rgpc": {"phase_weight": 0.5, "transition_weight": 0.1, "reliability_weight": 0.3, "teacher_weight": 0.0},
    "full_rgpc_teacher": {"phase_weight": 0.5, "transition_weight": 0.1, "reliability_weight": 0.3, "teacher_weight": 0.2},
    "feature_exclusion": {"phase_weight": 0.5, "transition_weight": 0.1, "reliability_weight": 0.3, "teacher_weight": 0.0, "exclude_rule_source_features": True},
}
```

The runner invokes the existing training, nested LOSO evaluation, and continuous replay entry points for each run. It skips a run only when its output manifest has matching input/config hashes and a success status. Aggregation rejects missing seeds, missing subjects, synthetic/demo flags, mismatched hashes, or manually supplied metrics. It writes per-subject, per-seed, macro, standard-deviation, cross-domain, corruption, and latency summaries.

- [ ] **Step 4: Verify GREEN with a controlled tiny runner**

The integration test uses an injected local runner that writes complete deterministic JSON artifacts for two variants, one seed, and one subject; assertions target aggregation and resume-by-hash behavior, not mock call counts.

```powershell
python -m pytest tests/risk/phase_model/test_experiment_matrix.py tests/integration/test_rg_pcnet_experiment_matrix.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add risk/phase_model/experiment_matrix.py scripts/run_rg_pcnet_experiment_matrix.py tests/risk/phase_model/test_experiment_matrix.py tests/integration/test_rg_pcnet_experiment_matrix.py
git commit -m "feat: run fixed rgpc ablation and domain matrix"
```

## Plan 3 Completion Gate

1. Run the full focused command in Task 7 with zero failures.
2. Replay at least one hour-equivalent chronological stream with ADL, fall, near-fall, corruption, recovery, and repeated-event sections.
3. Inspect JSON first, then plots; every plotted number must match its JSON source.
4. Verify offline and live decoders load identical timer values and release hash.
5. Run GitNexus `detect_changes`; inspect affected monitoring, dispatcher, and release flows.
6. Keep the candidate non-promoted if any event, calibration, coverage, hash, or seed gate fails.
7. Confirm the matrix has all 84 primary LOSO runs (`7 variants × 3 seeds × 4 subjects`) plus separately identified cross-domain runs.
8. Commit generated metrics/figures only when they are real experiment evidence, not synthetic smoke-test output.
