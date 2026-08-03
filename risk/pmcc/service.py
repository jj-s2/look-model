"""Offline orchestration for the uncertainty-aware PMCC forecast path.

This module deliberately has no device-client imports.  It consumes already
validated daily observations and leaves confirmed fall-event escalation to the
existing event state machine.
"""
from __future__ import annotations

from datetime import date, datetime, time
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Sequence

from .baseline import BaselineManager
from .chains import TemporalChainBuilder
from .changes import detect_changes
from .decisions import make_forecast_decision
from .features import FeatureWindow, build_feature_window
from .feedback import FeedbackStore
from .schema import BaselineState, DailyObservation, DecisionBand, EvidenceTier, OutcomeFeedback, PMCCForecast, TemporalChain
from .survival import SurvivalRiskModel, cumulative_risk_for_horizons, hazards_to_cumulative
from .uncertainty import UncertaintyGate


_NON_RELEASE_TIERS = {EvidenceTier.SYNTHETIC_RESEARCH, EvidenceTier.OFFLINE_FIXTURE}


class PMCCService:
    """Store daily records and produce deterministic, local PMCC forecasts."""

    def __init__(
        self,
        *,
        model: SurvivalRiskModel | None = None,
        feedback_path: str | Path | None = None,
        observations: Sequence[DailyObservation] = (),
        population_priors: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._model = model
        self._feedback = FeedbackStore(feedback_path) if feedback_path is not None else None
        self._priors = dict(population_priors or {})
        self._observations: dict[str, dict[date, DailyObservation]] = {}
        self._baselines: dict[str, BaselineManager] = {}
        for observation in observations:
            self.observe(observation)

    def observe(self, observation: DailyObservation) -> None:
        """Validate and retain the newest record for an observation's local day."""
        if not isinstance(observation, DailyObservation):
            raise ValueError("observation must be a DailyObservation")
        subject = observation.subject_id
        records = self._observations.setdefault(subject, {})
        previous = records.get(observation.observed_at.date())
        if previous is None or observation.observed_at >= previous.observed_at:
            records[observation.observed_at.date()] = observation
            # BaselineManager intentionally supports replacement corrections.
            baseline = self._baselines.setdefault(subject, BaselineManager(self._priors))
            baseline.add(observation)

    def observations(self, subject_id: str | None = None) -> tuple[DailyObservation, ...]:
        """Expose an immutable chronological snapshot for local integrations/tests."""
        if subject_id is None:
            return tuple(record for subject in sorted(self._observations) for _, record in sorted(self._observations[subject].items()))
        return tuple(record for _, record in sorted(self._records_for(subject_id).items()))

    def forecast(self, subject_id: str, as_of: date, release_mode: bool = False) -> PMCCForecast:
        if not isinstance(subject_id, str) or not subject_id:
            raise ValueError("subject_id must be a non-empty string")
        if isinstance(as_of, datetime) or not isinstance(as_of, date):
            raise ValueError("as_of must be a date, not a datetime")
        if not isinstance(release_mode, bool):
            raise ValueError("release_mode must be a boolean")
        records = tuple(record for _, record in sorted(self._records_for(subject_id).items()) if record.observed_at.date() <= as_of)
        if not records:
            raise ValueError("subject has no observations on or before as_of")
        baseline = self._baselines[subject_id]
        changes = detect_changes(records, baseline)
        chains = TemporalChainBuilder().build(changes)
        window = build_feature_window(records, baseline, chains, as_of)
        hazards, model_name, degradation = self._predict_hazards(window)
        cumulative = hazards_to_cumulative(hazards)
        risk = cumulative_risk_for_horizons(hazards)
        coverage, quality = _window_reliability(window)
        evidence_tier, promoted = _evidence(records)
        uncertainty = UncertaintyGate().evaluate((cumulative,) * 5, coverage, quality, evidence_tier, release_mode)
        state = _overall_baseline_state(baseline, records)
        decision = make_forecast_decision(risk, uncertainty, state, _evidence_groups(records), promoted)
        best_chain = _best_chain(chains)
        forecast_at = datetime.combine(as_of, time.min, tzinfo=records[-1].observed_at.tzinfo)
        reasons = tuple(dict.fromkeys((*degradation, *uncertainty.reasons, *decision.reasons)))
        provenance = {
            "forecast_kind": "fall_forecast",
            "risk": risk,
            "hazards": hazards,
            "model": model_name,
            "degradation_reasons": degradation,
            "coverage": coverage,
            "quality": quality,
            "evidence_tier": evidence_tier.value,
            "promoted": promoted,
            "abstained": decision.abstained,
            "reasons": reasons,
            "decision": decision.decision,
            "forecast_band": decision.forecast_band,
            "feature_names": window.feature_names,
            "association_only": True,
        }
        return PMCCForecast(
            subject_id=subject_id,
            forecast_at=forecast_at,
            horizon_days=7,
            risk_score=risk["72h"],
            decision_band=DecisionBand(decision.forecast_band),
            confidence=0.0 if decision.abstained else max(0.0, 1.0 - uncertainty.width_72h),
            baseline_state=state,
            temporal_chain=best_chain,
            provenance=provenance,
        )

    def record_feedback(self, feedback: OutcomeFeedback) -> None:
        if not isinstance(feedback, OutcomeFeedback):
            raise ValueError("feedback must be an OutcomeFeedback")
        if self._feedback is not None:
            self._feedback.append(feedback)

    def feedback_records(self, forecast_id: str | None = None) -> tuple[OutcomeFeedback, ...]:
        return () if self._feedback is None else self._feedback.read(forecast_id)

    def _records_for(self, subject_id: str) -> dict[date, DailyObservation]:
        records = self._observations.get(subject_id)
        if records is None:
            raise ValueError("subject has no observations")
        return records

    def _predict_hazards(self, window: FeatureWindow) -> tuple[tuple[float, ...], str, tuple[str, ...]]:
        if self._model is None:
            return _cpu_rule_hazards(window), "cpu_rule", ()
        try:
            return _validate_hazards(self._model.predict_hazards(window)), "configured_model", ()
        except Exception:
            # A failed optional/model load must never make a monitoring loop
            # contact hardware or produce an unbounded score.
            return _cpu_rule_hazards(window), "cpu_rule_fallback", ("model_inference_failed",)


def _cpu_rule_hazards(window: FeatureWindow) -> tuple[float, ...]:
    """Small deterministic local fallback, not a clinical performance claim."""
    chain_index = window.feature_names.index("chain_strength")
    observed = [row[chain_index] for row, mask in zip(window.values, window.missing_mask) if not mask[chain_index]]
    chain_strength = max(observed, default=0.0)
    coverage, quality = _window_reliability(window)
    base = min(0.25, 0.01 + 0.12 * chain_strength + 0.03 * (1.0 - coverage) + 0.02 * (1.0 - quality))
    return tuple(min(0.5, base * factor) for factor in (1.0, 0.95, 0.9, 0.8, 0.7, 0.65, 0.6))


def _validate_hazards(values: Sequence[float]) -> tuple[float, ...]:
    hazards = tuple(values)
    if len(hazards) != 7:
        raise ValueError("model must return seven hazards")
    if any(isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or not 0 <= float(value) <= 1 for value in hazards):
        raise ValueError("model hazards must be finite probabilities")
    return tuple(float(value) for value in hazards)


def _window_reliability(window: FeatureWindow) -> tuple[float, float]:
    raw_indices = [index for index, name in enumerate(window.feature_names) if name.endswith(":raw")]
    if not raw_indices:
        return 0.0, 0.0
    observed = [not row[index] for row in window.missing_mask for index in raw_indices]
    qualities = [row[index] for row, mask in zip(window.quality, window.missing_mask) for index in raw_indices if not mask[index]]
    return sum(observed) / len(observed), sum(qualities) / len(qualities) if qualities else 0.0


def _evidence(records: Sequence[DailyObservation]) -> tuple[EvidenceTier, bool]:
    tiers: list[EvidenceTier] = []
    promoted = True
    for record in records:
        value = record.provenance.get("evidence_tier", EvidenceTier.OFFLINE_FIXTURE.value)
        try:
            tiers.append(EvidenceTier(value))
        except (TypeError, ValueError):
            tiers.append(EvidenceTier.OFFLINE_FIXTURE)
        promoted = promoted and record.provenance.get("promoted") is True
    tier = max(tiers, key=lambda item: (item in _NON_RELEASE_TIERS, item.value))
    return tier, promoted and tier not in _NON_RELEASE_TIERS


def _overall_baseline_state(baseline: BaselineManager, records: Sequence[DailyObservation]) -> BaselineState:
    states = [baseline.state(feature) for record in records for feature in record.features]
    if any(state is BaselineState.PERSONAL for state in states):
        return BaselineState.PERSONAL
    if any(state is BaselineState.BLENDED for state in states):
        return BaselineState.BLENDED
    return BaselineState.POPULATION_ONLY


def _evidence_groups(records: Sequence[DailyObservation]) -> int:
    groups: set[str] = set()
    for record in records:
        for feature, value in record.features.items():
            if value is None or record.availability.get(feature) is False:
                continue
            name = feature.lower()
            if any(token in name for token in ("heart", "spo2", "pressure", "respir", "physiology")):
                groups.add("physiology")
            elif any(token in name for token in ("step", "gait", "sway", "stride", "sit", "activity", "balance")):
                groups.add("vision_activity")
            else:
                groups.add("other")
    return len(groups)


def _best_chain(chains: Sequence[TemporalChain]) -> TemporalChain | None:
    if not chains:
        return None
    return max(chains, key=lambda chain: float(chain.provenance.get("score", 0.0)))
