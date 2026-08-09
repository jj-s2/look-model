# RG-PCNet Calibration and Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce subject-safe OOF calibration, constrained threshold/reliability selection, and a single versioned release configuration consumed identically by evaluation and inference.

**Architecture:** Repair current selection failure modes first, then add a bounded calibration artifact and nested subject-OOF selector for RG-PCNet. Package model, calibration, reliability gate, event parameters, hashes, and provenance in one validated release manifest. Add a separate RG-PCNet predictor and change the service to respect the predictor’s release decision instead of a hard-coded probability.

**Tech Stack:** Python 3.10, NumPy, PyTorch, pytest, existing JSON release infrastructure.

## Global Constraints

- Outer test subjects are never used for temperature, thresholds, reliability gate, state-machine parameters, or model choice.
- Temperature is constrained to `[0.5, 5.0]` and falls back to `1.0` for single-class, non-finite, or non-improving calibration data.
- Threshold selection must enforce recall floor, false-positive ceiling, and minimum coverage; absence of a feasible threshold is an explicit non-promotion result.
- Reliability is a decision gate, not a multiplier applied to calibrated fall probability.
- Evaluation and inference load one `release_config.json`; no service-level threshold literal may override it.
- A candidate with unavailable class metrics, degenerate folds, missing hashes, or failed continuous evaluation cannot be promoted.
- Existing PA-DTSF release loading remains backward compatible.
- Preserve all unrelated uncommitted changes; stage only task files.
- Before modifying each existing symbol, run GitNexus upstream impact and inspect direct callers.
- Every production change follows RED → GREEN → REFACTOR.

---

## File Structure

- Modify `scripts/optimize_phase_threshold.py`: validate aligned inputs before threshold search.
- Modify `scripts/run_phase_f1_experiment.py`: return auditable non-promotion for degenerate confirmation folds.
- Create `risk/phase_model/rg_calibration.py`: bounded calibration and selective threshold search.
- Create `risk/phase_model/nested_selection.py`: subject-disjoint OOF fitting.
- Create `risk/phase_model/release_config.py`: immutable release decision contract and hash validation.
- Create `risk/phase_model/rg_predictor.py`: RG-PCNet checkpoint predictor using release configuration.
- Modify `risk/phase_model/service.py`: consume `fall_decision`, including abstention.
- Add focused unit and integration tests.

### Task 1: Repair current threshold-search error semantics

**Files:**
- Modify: `scripts/optimize_phase_threshold.py:123-210`
- Modify: `scripts/run_phase_f1_experiment.py:55-115`
- Test: `tests/integration/test_threshold_search.py`
- Test: `tests/integration/test_f1_experiment.py`

**Interfaces:**
- Consumes: aligned logits, labels, subjects.
- Produces: deterministic validation errors or a result with `promoted=False` and a machine-readable reason.

- [ ] **Step 1: Add or retain failing regression tests**

Name the breaks: silently truncating mismatched sequences and raising on a single-class held-out fold must fail.

```python
def test_inner_search_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        inner_threshold_search([1.0, -1.0], [1], ["s1", "s2"])
```

```python
def test_evaluate_and_promote_rejects_single_class_confirmation_without_crashing():
    result = evaluate_and_promote(
        validation_logits=[2.0, -2.0, 1.0, -1.0],
        validation_labels=[1, 0, 1, 0],
        validation_subjects=["s1", "s1", "s2", "s2"],
        test_logits=[1.0, 2.0], test_labels=[1, 1], test_subjects=["s3", "s3"],
        baseline={"inner_mean_f1": 0.5, "brier": 0.25}, config=_Config,
    )
    assert result["promoted"] is False
    assert result["reason"] == "confirmation fold requires both classes"
```

- [ ] **Step 2: Verify RED against the current implementation**

Run:

```powershell
python -m pytest tests/integration/test_threshold_search.py::test_inner_search_rejects_mismatched_lengths tests/integration/test_f1_experiment.py::test_evaluate_and_promote_rejects_single_class_confirmation_without_crashing -q
```

Expected: first test reports a recall-floor error instead of a length error; second raises during evaluation or calibration.

- [ ] **Step 3: Add minimal validation and graceful rejection**

At the start of `_build_records` and `inner_threshold_search`, use:

```python
if not logits or len(logits) != len(labels) or len(logits) != len(subjects):
    raise ValueError("logits, labels, and subjects must be non-empty and have the same length")
```

At the start of `evaluate_and_promote`, validate test lengths and add:

```python
if len({int(value) for value in test_labels}) < 2:
    return {
        "promoted": False,
        "reason": "confirmation fold requires both classes",
        "temperature": 1.0,
        "threshold": None,
        "inner_metrics": None,
        "confirmation_metrics": None,
        "candidate": None,
        "baseline": dict(baseline),
        "fold_thresholds": None,
        "fold_temperatures": None,
    }
```

- [ ] **Step 4: Verify GREEN and current selection tests**

```powershell
python -m pytest tests/integration/test_threshold_search.py tests/integration/test_f1_experiment.py tests/risk/phase_model/test_selection.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add scripts/optimize_phase_threshold.py scripts/run_phase_f1_experiment.py tests/integration/test_threshold_search.py tests/integration/test_f1_experiment.py
git commit -m "fix: reject degenerate threshold folds safely"
```

### Task 2: Implement bounded, evidence-gated temperature calibration

**Files:**
- Create: `risk/phase_model/rg_calibration.py`
- Test: `tests/risk/phase_model/test_rg_calibration.py`

**Interfaces:**
- Consumes: validation logits/labels and optional split hash.
- Produces: `CalibrationArtifact` and calibrated probabilities.

- [ ] **Step 1: Write failing calibration tests with hand-checked fallbacks**

```python
import pytest

from risk.phase_model.rg_calibration import fit_bounded_temperature


def test_single_class_calibration_falls_back_to_identity():
    result = fit_bounded_temperature([2.0, 3.0], [1, 1], split_hash="abc")
    assert result.temperature == 1.0
    assert result.enabled is False
    assert result.reason == "calibration requires both classes"
    assert result.class_counts == {"0": 0, "1": 2}


def test_temperature_is_bounded_and_only_enabled_when_nll_improves():
    result = fit_bounded_temperature([4.0, -4.0, -2.0, 2.0], [1, 0, 1, 0], split_hash="abc")
    assert 0.5 <= result.temperature <= 5.0
    if result.enabled:
        assert result.nll_after < result.nll_before
    else:
        assert result.temperature == 1.0
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/risk/phase_model/test_rg_calibration.py -q`.

- [ ] **Step 3: Implement the immutable artifact and bounded grid fit**

Use these fields exactly:

```python
@dataclass(frozen=True)
class CalibrationArtifact:
    temperature: float
    enabled: bool
    reason: str
    sample_count: int
    class_counts: dict[str, int]
    nll_before: float
    nll_after: float
    brier_before: float
    brier_after: float
    split_hash: str

    def calibrate(self, logits: Sequence[float]) -> list[float]:
        return [_sigmoid(float(value) / self.temperature) for value in logits]
```

`fit_bounded_temperature` validates lengths, finiteness, binary labels, and split hash. Search 181 log-spaced candidates from `0.5` through `5.0`. Enable the best candidate only if `nll_after < nll_before - 1e-6` or `brier_after < brier_before - 1e-6`; otherwise return temperature `1.0` with reason `"calibration did not improve nll or brier"`. All fallback metrics remain finite.

- [ ] **Step 4: Verify GREEN**

```powershell
python -m pytest tests/risk/phase_model/test_rg_calibration.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add risk/phase_model/rg_calibration.py tests/risk/phase_model/test_rg_calibration.py
git commit -m "feat: add bounded validation calibration"
```

### Task 3: Select fall and reliability gates from subject-disjoint OOF records

**Files:**
- Create: `risk/phase_model/nested_selection.py`
- Test: `tests/risk/phase_model/test_nested_selection.py`

**Interfaces:**
- Consumes: `OOFRecord(subject_id, label, fall_logit, reliability)`.
- Produces: `SelectiveThreshold(temperature, fall_threshold, reliability_threshold, coverage, recall, f1, fpr, aurc, feasible, reason)`.

- [ ] **Step 1: Write failing constraint tests**

```python
from risk.phase_model.nested_selection import OOFRecord, select_oof_thresholds


def test_oof_selection_enforces_recall_fpr_and_coverage():
    records = [
        OOFRecord("s1", 1, 3.0, 0.9), OOFRecord("s1", 0, -2.0, 0.8),
        OOFRecord("s2", 1, 2.0, 0.7), OOFRecord("s2", 0, 1.0, 0.2),
    ]
    result = select_oof_thresholds(records, recall_floor=1.0, fpr_ceiling=0.0, coverage_floor=0.5)
    assert result.feasible
    assert result.recall == 1.0
    assert result.fpr == 0.0
    assert result.coverage >= 0.5


def test_no_feasible_oof_gate_returns_explicit_failure():
    records = [OOFRecord("s1", 1, -3.0, 0.1), OOFRecord("s2", 0, 3.0, 0.9)]
    result = select_oof_thresholds(records, recall_floor=1.0, fpr_ceiling=0.0, coverage_floor=1.0)
    assert result.feasible is False
    assert result.reason == "no threshold pair satisfies recall, fpr, and coverage constraints"
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/risk/phase_model/test_nested_selection.py -q`.

- [ ] **Step 3: Implement the two-dimensional constrained search**

Define fall thresholds `0.20..0.80` in `0.01` steps and reliability thresholds `0.00..0.90` in `0.05` steps. Calibrate all logits once with `fit_bounded_temperature`. For each pair, abstain where reliability is below its threshold; compute coverage over all records and binary rates over selected records. Reject pairs below any constraint. Rank feasible pairs by `(subject_macro_f1, worst_subject_f1, -fpr, coverage, -fall_threshold)`.

Compute risk-coverage points by sorting records descending by reliability and taking every prefix. Risk is classification error at the chosen fall threshold; AURC is trapezoidal area over coverage. Persist every evaluated constraint and the count of feasible pairs.

- [ ] **Step 4: Verify GREEN**

```powershell
python -m pytest tests/risk/phase_model/test_nested_selection.py tests/risk/phase_model/test_rg_calibration.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add risk/phase_model/nested_selection.py tests/risk/phase_model/test_nested_selection.py
git commit -m "feat: select calibrated reliability gates from oof data"
```

### Task 4: Define one validated release decision contract

**Files:**
- Create: `risk/phase_model/release_config.py`
- Test: `tests/risk/phase_model/test_release_config.py`

**Interfaces:**
- Consumes: release metadata and artifact paths.
- Produces: `RGPCReleaseConfig`, `load_release_config(path)`, `write_release_config(config, path)`.

- [ ] **Step 1: Write failing round-trip and hash tests**

```python
import pytest

from risk.phase_model.release_config import RGPCReleaseConfig, load_release_config, write_release_config


def _config():
    return RGPCReleaseConfig(
        schema_version="rgpc.release.v1", release_id="r1", model_sha256="a" * 64,
        dataset_sha256="b" * 64, split_sha256="c" * 64, temperature=1.5,
        fall_threshold=0.6, reliability_threshold=0.4, confirm_seconds=0.8,
        recovery_seconds=2.0, cooldown_seconds=10.0, minimum_coverage=0.6,
    )


def test_release_config_round_trip_is_exact(tmp_path):
    path = tmp_path / "release_config.json"
    write_release_config(_config(), path)
    assert load_release_config(path) == _config()


def test_release_config_rejects_invalid_hash():
    with pytest.raises(ValueError, match="model_sha256"):
        RGPCReleaseConfig(**{**_config().__dict__, "model_sha256": "short"})
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/risk/phase_model/test_release_config.py -q`.

- [ ] **Step 3: Implement strict validation and canonical JSON**

Validate schema exactly, non-empty release ID, lowercase 64-character hexadecimal hashes, temperature `[0.5,5.0]`, probability fields `[0,1]`, non-negative seconds, and `confirm_seconds > 0`. Serialize with `ensure_ascii=False`, `sort_keys=True`, two-space indentation, and one terminal newline. Unknown or missing JSON fields raise `ValueError` rather than being ignored.

- [ ] **Step 4: Verify GREEN**

```powershell
python -m pytest tests/risk/phase_model/test_release_config.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add risk/phase_model/release_config.py tests/risk/phase_model/test_release_config.py
git commit -m "feat: version rgpc release decision config"
```

### Task 5: Load RG-PCNet and apply reliability abstention in inference

**Files:**
- Create: `risk/phase_model/rg_predictor.py`
- Modify: `risk/phase_model/service.py:37-77`
- Test: `tests/risk/phase_model/test_rg_predictor.py`
- Test: `tests/risk/phase_model/test_service.py`

**Interfaces:**
- Consumes: RG-PCNet checkpoint, `release_config.json`, `DualWindow`.
- Produces: backward-compatible `PhaseModelOutput`; `fall_decision=None` means abstain.

- [ ] **Step 1: Write failing decision-boundary tests**

```python
def test_low_reliability_prediction_abstains_without_multiplying_probability(rg_release, dual_window):
    predictor = RGPredictor.from_release(rg_release, device="cpu")
    predictor._predict_tensors = lambda *_: (0.8, (0.1, 0.8, 0.1), 0.2)
    output = predictor.predict(dual_window)
    assert output.fall_event_prob == pytest.approx(0.8)
    assert output.fall_decision is None


def test_service_emits_no_fall_forecast_for_abstained_prediction(service_with_abstaining_predictor, observation_sequence):
    events = tuple(event for item in observation_sequence for event in service_with_abstaining_predictor.observe(item))
    assert not any(event.event_type is EventType.FALL_FORECAST for event in events)
    assert any(event.payload.get("quality_mode") == "abstained" for event in events)
```

- [ ] **Step 2: Verify RED**

```powershell
python -m pytest tests/risk/phase_model/test_rg_predictor.py tests/risk/phase_model/test_service.py -q
```

- [ ] **Step 3: Implement the predictor adapter and remove the service literal**

`RGPredictor.from_release` loads `checkpoint.pt` and `release_config.json`, verifies the checkpoint SHA-256 before `torch.load`, constructs `RGPCNet`, and uses `build_temporal_features` on the 64-frame window. It maps three coarse probabilities into the existing six-phase schema as:

```python
phase_probs = (
    coarse[0], 0.0, coarse[1], 0.0, coarse[2], 0.0,
)
```

It returns calibrated `fall_event_prob` unchanged by reliability. Set `fall_decision=None` when reliability is below the release threshold; otherwise use `int(fall_probability >= fall_threshold)`. Store reliability in `quality_score`.

In `PhaseRiskService.observe`, replace `if output.fall_event_prob >= 0.3:` with `if output.fall_decision == 1:`. Before event creation, if `fall_decision is None`, return a POSE event with `quality_mode="abstained"` and reason `"model_reliability_gate"`.

- [ ] **Step 4: Verify GREEN and legacy compatibility**

```powershell
python -m pytest tests/risk/phase_model/test_rg_predictor.py tests/risk/phase_model/test_service.py tests/risk/phase_model/test_torch_predictor.py tests/integration/test_phase_f1_release.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add risk/phase_model/rg_predictor.py risk/phase_model/service.py tests/risk/phase_model/test_rg_predictor.py tests/risk/phase_model/test_service.py
git commit -m "feat: gate rgpc inference with release reliability"
```

### Task 6: Build a nested-LOSO release artifact without test-fold tuning

**Files:**
- Create: `scripts/evaluate_rg_pcnet_loso.py`
- Test: `tests/integration/test_rg_pcnet_loso_release.py`

**Interfaces:**
- Consumes: outer-fold prediction JSONL files, checkpoint, dataset/split manifests, constraints.
- Produces: per-fold calibration, outer-fold metrics, aggregate metrics, and optional non-promoted/promoted release.

- [ ] **Step 1: Write failing leakage and artifact tests**

The fixture contains inner OOF records for subjects `s1,s2,s3` and outer records for `s4`. Assert:

```python
assert result["outer_subject"] == "s4"
assert "s4" not in result["calibration_subjects"]
assert result["release_config"]["fall_threshold"] == result["selection"]["fall_threshold"]
assert result["promoted"] is False
assert (output / "outer_predictions.jsonl").exists()
assert (output / "calibration.json").exists()
```

Add a second test that places `s4` in OOF records and expects `ValueError("outer subject leaked into calibration")`.

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/integration/test_rg_pcnet_loso_release.py -q`.

- [ ] **Step 3: Implement orchestration and promotion inputs**

The script validates subject disjointness, calls `select_oof_thresholds`, applies its calibration and decision gates once to the outer records, and writes canonical JSON/JSONL. It never recalculates thresholds from outer results. A release is promotable only when the selection is feasible, both outer classes exist, coverage meets the configured floor, hashes validate, and the continuous-event gate from Plan 3 is present and passing. Until Plan 3 supplies that artifact, set `promoted=False` with reason `"continuous event evaluation is required"`.

- [ ] **Step 4: Verify GREEN and the complete calibration suite**

```powershell
python -m pytest tests/risk/phase_model/test_rg_calibration.py tests/risk/phase_model/test_nested_selection.py tests/risk/phase_model/test_release_config.py tests/risk/phase_model/test_rg_predictor.py tests/integration/test_rg_pcnet_loso_release.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add scripts/evaluate_rg_pcnet_loso.py tests/integration/test_rg_pcnet_loso_release.py
git commit -m "feat: evaluate rgpcnet with nested loso calibration"
```

## Plan 2 Completion Gate

1. Re-run the two originally failing tests and record their exact passing node IDs.
2. Run all `tests/risk/phase_model` plus RG-PCNet integration tests with zero failures.
3. Inspect a generated calibration artifact: temperature is bounded, subjects are listed, fall/reliability thresholds are explicit, and fall probability is not multiplied by reliability.
4. Verify the service has no probability threshold literal controlling RG-PCNet alerts.
5. Run GitNexus `detect_changes`; inspect direct callers of selection, service, and release symbols.
6. Commit and push only after the release remains non-promoted without Plan 3’s continuous-event artifact.
