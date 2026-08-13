"""Conservative PMCC forecast decisions, separate from fall-event emergencies."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Mapping

from .schema import BaselineState
from .uncertainty import UncertaintySummary


@dataclass(frozen=True)
class DecisionSummary:
    decision: str
    forecast_band: str
    abstained: bool
    reasons: tuple[str, ...]


def make_forecast_decision(
    risk: Mapping[str, float],
    uncertainty: UncertaintySummary,
    baseline_state: BaselineState,
    evidence_groups: int,
    promoted: bool,
) -> DecisionSummary:
    """Return only info/watch/warning; critical remains owned by fall_event."""
    risk_72h = _risk_72h(risk)
    if not isinstance(uncertainty, UncertaintySummary):
        raise ValueError("uncertainty must be an UncertaintySummary")
    if not isinstance(baseline_state, BaselineState):
        try:
            baseline_state = BaselineState(baseline_state)
        except (TypeError, ValueError) as error:
            raise ValueError("baseline_state must be a BaselineState") from error
    if isinstance(evidence_groups, bool) or not isinstance(evidence_groups, int) or evidence_groups < 0:
        raise ValueError("evidence_groups must be a non-negative integer")
    if not isinstance(promoted, bool):
        raise ValueError("promoted must be a boolean")

    band = _forecast_band(risk_72h)
    reasons = list(uncertainty.reasons)
    if evidence_groups < 2:
        reasons.append("missing_visual_evidence_group")
    if not promoted:
        reasons.append("not_promoted")
    if reasons:
        return DecisionSummary("info", band, True, tuple(dict.fromkeys(reasons)))

    decision = "info" if band == "low" else "watch" if band == "elevated" else "warning"
    if baseline_state is BaselineState.POPULATION_ONLY and decision == "warning":
        decision = "watch"
        reasons.append("population_only_cap")
    return DecisionSummary(decision, band, False, tuple(reasons))


def _risk_72h(risk: Mapping[str, float]) -> float:
    if not isinstance(risk, Mapping) or "72h" not in risk:
        raise ValueError("risk must contain a 72h probability")
    value = risk["72h"]
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ValueError("risk[72h] must be a finite probability in [0, 1]")
    return float(value)


def _forecast_band(risk_72h: float) -> str:
    if risk_72h < 0.15:
        return "low"
    if risk_72h < 0.35:
        return "elevated"
    if risk_72h < 0.6:
        return "high"
    return "very_high"
