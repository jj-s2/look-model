from __future__ import annotations

from risk.pmcc.decisions import make_forecast_decision
from risk.pmcc.schema import BaselineState, EvidenceTier
from risk.pmcc.uncertainty import UncertaintyGate


def _uncertainty():
    return UncertaintyGate().evaluate(tuple((0.8,) * 7 for _ in range(5)), 1.0, 1.0, EvidenceTier.REAL_PUBLIC, False)


def test_decision_abstains_without_visual_evidence_group() -> None:
    result = make_forecast_decision({"72h": 0.8}, _uncertainty(), BaselineState.PERSONAL, 1, True)
    assert result.abstained is True
    assert "missing_visual_evidence_group" in result.reasons
    assert result.decision == "info"


def test_population_only_never_escalates_past_watch() -> None:
    result = make_forecast_decision({"72h": 0.8}, _uncertainty(), BaselineState.POPULATION_ONLY, 2, True)
    assert result.forecast_band == "very_high"
    assert result.decision == "watch"


def test_very_high_forecast_never_emits_critical() -> None:
    result = make_forecast_decision({"72h": 0.8}, _uncertainty(), BaselineState.PERSONAL, 2, True)
    assert result.forecast_band == "very_high"
    assert result.decision == "warning"
    assert result.decision != "critical"
