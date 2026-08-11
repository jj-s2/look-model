# PACE-WB Mental Warning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe, low-burden, personalized wellbeing screening lane integrated with the elderly fall-monitoring project while keeping learned mental-health models research-only until elderly external validation exists.

**Architecture:** A rules-first PACE-Behavior lane validates daily aggregates, maintains per-feature effective-day baselines, detects sustained change, and emits only local invitation candidates. A separate PACE-Voluntary shadow lane evaluates consented short check-ins with MacBERT/eGeMAPS research components and reliability/abstention diagnostics. A typed delivery contract prevents ordinary wellbeing events from entering network/family/EZVIZ dispatch.

**Tech Stack:** Existing Python 3.10+ modules, frozen dataclasses, pytest, NumPy/scikit-learn/joblib where available, optional Transformers/openSMILE research dependencies behind explicit CLI checks, existing `SensorEvent`, `DecisionEngine`, `LiveMonitoringService`, `AlertDispatcher`, and JSONL audit artifacts.

## Global Constraints

- `is_diagnosis` is always `false` for model and UI outputs.
- Audio/text is processed only after explicit voluntary consent; no continuous ambient recording.
- Passive trends can create at most a local short-check-in candidate; they cannot create a critical event or external delivery.
- Short-checkin invitations are limited to one per 7 days; GDS-15 invitations are limited to one per 28 days; quiet hours remain 21:00–08:00.
- Fewer than 14 effective baseline days cannot produce an operational passive invitation candidate.
- Missing, invalid, low-quality, out-of-domain, conflicting, or version-incompatible evidence must abstain rather than be imputed as zero.
- Self-harm candidates are separate human-review events and cannot be auto-sent to family, EZVIZ, webhook, SMS, email, or emergency services.
- Preserve all unrelated dirty files; stage only files listed in each task.
- Run GitNexus upstream impact before editing existing symbols and `detect_changes` before every commit.

---

### Task 1: Freeze contracts

**Files:** Create `mental/contracts.py`; create `tests/mental/test_contracts.py`; modify `mental/__init__.py`.

**Interfaces:** Frozen `DailyWellbeingObservation`, `VoluntaryCheckin`, `ModalityEvidence`, `WellbeingAssessmentEvent`, and `SensitiveExpressionCandidate`; `WellbeingAssessmentEvent.to_sensor_event()` preserves `delivery_scope`, `model_mode`, `abstained`, and `is_diagnosis`.

- [ ] Write RED tests for timezone-aware timestamps, finite bounded values, invalid enums, explicit consent, detached mappings, `is_diagnosis=false`, and `external_forbidden` defaults.
- [ ] Run `python -m pytest tests/mental/test_contracts.py -q`; verify expected missing-contract failures.
- [ ] Implement strict frozen dataclasses and compatibility conversion.
- [ ] Run focused and existing mental tests.
- [ ] Run GitNexus impact for `WellbeingAssessmentEvent`, stage only the three files, run staged `detect_changes`, and commit `feat: add wellbeing evidence contracts`.

### Task 2: Harden trend and interaction policy

**Files:** Modify `mental/trend.py` and `mental/interaction_policy.py`; test `tests/mental/test_trend.py` and `tests/mental/test_interaction_policy.py`.

**Interfaces:** Preserve public classes/fields; add `cold_start`, `provisional`, `operational`, per-feature effective-day counts and abstention reasons; cooldown expiry alone must not invite GDS-15.

- [ ] Write RED tests for idle GDS eligibility, empty readiness, invalid numbers, same-domain 2-of-3 persistence, cross-domain isolation, 14-day readiness and existing cooldowns.
- [ ] Run focused tests and capture expected failures.
- [ ] Implement strict validation, effective-day readiness, median/MAD baseline state and same-domain persistence.
- [ ] Run focused, mental and fusion regression tests.
- [ ] Run GitNexus impact for `WellbeingTrendAnalyzer.update` and `InteractionPolicy.evaluate`; stage exact files, run `detect_changes`, commit `fix: enforce safe wellbeing trend readiness`.

### Task 3: Implement PACE-Behavior

**Files:** Create `mental/pace_behavior.py`; create `tests/mental/test_pace_behavior.py`; modify `mental/__init__.py`.

**Interfaces:** `PACEBehaviorConfig(min_effective_days=14, provisional_days=7, persistence_window=3, persistence_required=2)` and `PACEBehaviorModel.update(DailyWellbeingObservation) -> WellbeingAssessmentEvent`; never emit `critical` or external delivery.

- [ ] Write RED tests for cold/provisional/operational states, invalid/low-quality observations, travel exclusion, baseline freeze on anomaly and deterministic updates.
- [ ] Run the focused file and verify missing-module failures.
- [ ] Implement robust per-feature baseline, EWMA/CUSUM persistence and detached evidence.
- [ ] Run focused and mental tests; run impact for changed symbols; stage exact files, run `detect_changes`, commit `feat: add personalized wellbeing trend gate`.

### Task 4: Add voluntary shadow and safety gate

**Files:** Create `mental/voluntary_shadow.py`, `mental/safety_gate.py`, their tests, `scripts/train_wellbeing_shadow.py`, and `scripts/evaluate_wellbeing_shadow.py`.

**Interfaces:** `ShadowModelConfig`; `VoluntaryShadowModel.assess(VoluntaryCheckin)`; `SafetyGate.evaluate(evidence, behavior_event)`; absent optional dependencies or licensed data cause explicit failure/abstention, never synthetic clinical weights.

- [ ] Write RED tests for consent false, missing modalities, short answers, low ASR confidence, version mismatch, conflict, deletion flip and shadow mode.
- [ ] Run focused tests and verify expected failures.
- [ ] Implement optional research adapters, reliability gate and `promoted=false` metadata.
- [ ] Train only with explicit data/output paths, subject-pure splits and OOF calibration; report AUPRC, F1, Brier, ECE, coverage-risk and missingness slices.
- [ ] Run focused tests, `python -m py_compile` and regressions; stage exact files, run `detect_changes`, commit `feat: add voluntary wellbeing shadow lane`.

### Task 5: Add routing and delivery firewall

**Files:** Modify `fusion/decision_engine.py`, `pipeline/live_service.py`, `alerts/dispatcher.py`; create `tests/integration/test_wellbeing_delivery_scope.py`; update affected tests.

**Interfaces:** Preserve fall `RiskDecision`; unavailable/low-confidence wellbeing abstains; `_should_dispatch` rejects all wellbeing delivery scopes; human escalation is a separate authorized record.

- [ ] Run GitNexus impact for `DecisionEngine._wellbeing_change`, `DecisionEngine.evaluate`, `LiveMonitoringService._should_dispatch`, and `AlertDispatcher.dispatch`; warn on HIGH/CRITICAL.
- [ ] Write RED tests for low-quality abstention, local-only wellbeing, self-harm internal review, no EZVIZ/Webhook dispatch and unchanged fall delivery.
- [ ] Run tests to capture expected failures.
- [ ] Implement minimal quality checks, scope firewall and internal review records.
- [ ] Run fusion/pipeline/alert/UI/mental regressions; stage exact files, run `detect_changes`, commit `feat: isolate wellbeing actions from external alerts`.

### Task 6: Integrate local prompts and UI

**Files:** Modify `pipeline/live_service.py` and `ui/dashboard.py`; create `mental/checkin_prompts.py` and `tests/ui/test_wellbeing_interaction.py`; update dashboard tests.

**Interfaces:** `build_short_checkin_prompt()` returns 2–3 plain-language prompts; UI shows Chinese state/evidence/uncertainty; GDS payload never exposes `risk_answer`.

- [ ] Write RED tests for local-only prompts, large-text labels, decline/stop persistence, quiet hours and no risk-answer exposure.
- [ ] Run focused tests and capture expected failures.
- [ ] Implement prompt/view-model changes without continuous microphone capture or network calls.
- [ ] Run UI/integration/mental/pipeline tests; stage exact files, run `detect_changes`, commit `feat: add low-burden wellbeing interaction surface`.

### Task 7: Release and promotion gates

**Files:** Create `configs/screening/pace_wb_v1.json`, `scripts/evaluate_wellbeing_release.py`, `tests/integration/test_wellbeing_release.py`; update the design spec with measured evidence only.

**Interfaces:** Strict schema/model-mode/hashes/thresholds/readiness/prompt-budget/external-delivery config; canonical JSON/JSONL artifacts with SHA-256; `promoted=false` unless elderly external gates pass.

- [ ] Write RED tests for mismatches, missing elderly validation, invalid thresholds, duplicates, deterministic order and refusal.
- [ ] Run focused tests and capture expected failures.
- [ ] Implement strict config and atomic audit artifacts preserving prior output on failure.
- [ ] Run full tests/compile; stage exact files, run `detect_changes`, commit `feat: add pace-wb release and audit gates`.

### Task 8: Evidence report and final verification

**Files:** Create `docs/superpowers/handoff/2026-08-12-pace-wb-validation-report.md`, `outputs/mental/pace-wb-validation/metrics.json`, and `README.md`.

- [ ] Run complete mental/fusion/pipeline/alert/UI/integration suites, compileall and `git diff --check`.
- [ ] Record provenance, subject/time splits, missingness tests, model mode, metrics, limitations and unchanged fall metrics.
- [ ] Verify no raw media, credentials, device IDs or licensed datasets are committed.
- [ ] Run final GitNexus compare/detect_changes and stage only evidence files with real measurements; keep `promoted=false` when evidence is missing.
