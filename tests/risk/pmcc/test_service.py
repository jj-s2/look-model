from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from risk.pmcc.schema import DailyObservation, EvidenceTier, OutcomeFeedback, OutcomeType
from risk.pmcc.service import PMCCService


UTC = timezone.utc


def _observation(day: date, *, physiology: bool = True, tier: EvidenceTier = EvidenceTier.REAL_DEVICE_LONGITUDINAL) -> DailyObservation:
    features = {"steps": 2000.0 + (day.day % 3) * 100.0, "trunk_sway": 0.2 + (day.day % 2) * 0.03}
    quality = {"steps": 0.9, "trunk_sway": 0.9}
    availability = {"steps": True, "trunk_sway": True}
    if physiology:
        features["heart_rate"] = 70.0
        quality["heart_rate"] = 0.8
        availability["heart_rate"] = True
    return DailyObservation(
        "resident-1", datetime.combine(day, datetime.min.time(), tzinfo=UTC),
        features, quality, availability,
        {"evidence_tier": tier.value, "promoted": tier not in {EvidenceTier.SYNTHETIC_RESEARCH, EvidenceTier.OFFLINE_FIXTURE}},
    )


def _service_with_history(*, physiology: bool = True, tier: EvidenceTier = EvidenceTier.REAL_DEVICE_LONGITUDINAL) -> tuple[PMCCService, date]:
    as_of = date(2026, 8, 1)
    service = PMCCService()
    for offset in range(14):
        service.observe(_observation(as_of - timedelta(days=13 - offset), physiology=physiology, tier=tier))
    return service, as_of


def test_repeated_offline_forecasts_are_deterministic_and_expose_all_horizons() -> None:
    service, as_of = _service_with_history()

    first = service.forecast("resident-1", as_of)
    second = service.forecast("resident-1", as_of)

    assert first == second
    assert set(first.provenance["risk"]) == {"24h", "72h", "7d"}
    assert first.provenance["risk"]["24h"] <= first.provenance["risk"]["72h"] <= first.provenance["risk"]["7d"]
    assert first.provenance["decision"] != "critical"


def test_service_keeps_vision_activity_when_physiology_is_absent() -> None:
    service, as_of = _service_with_history(physiology=False)

    forecast = service.forecast("resident-1", as_of)

    assert forecast.provenance["coverage"] > 0.5
    assert "steps:raw" in forecast.provenance["feature_names"]
    assert "heart_rate:raw" not in forecast.provenance["feature_names"]


def test_sleep_and_physiology_without_vision_activity_abstains_visual_gate() -> None:
    as_of = date(2026, 8, 1)
    service = PMCCService()
    for offset in range(14):
        day = as_of - timedelta(days=13 - offset)
        service.observe(DailyObservation(
            "resident-1", datetime.combine(day, datetime.min.time(), tzinfo=UTC),
            {"sleep_duration_minutes": 420.0, "heart_rate": 70.0},
            {"sleep_duration_minutes": 0.9, "heart_rate": 0.9},
            {"sleep_duration_minutes": True, "heart_rate": True},
            {"evidence_tier": "real_device_longitudinal", "promoted": True},
        ))

    forecast = service.forecast("resident-1", as_of)

    assert forecast.provenance["abstained"] is True
    assert "missing_visual_evidence_group" in forecast.provenance["reasons"]


def test_historical_forecast_ignores_observations_added_after_as_of():
    class ZModel:
        def predict_hazards(self, features: object) -> tuple[float, ...]:
            window = features
            index = window.feature_names.index("steps:directional_z")
            z_score = abs(window.values[-1][index])
            return (min(0.95, 0.01 + 0.1 * z_score),) * 7

    as_of = date(2026, 8, 1)
    service = PMCCService(model=ZModel())
    for offset in range(14):
        day = as_of - timedelta(days=13 - offset)
        service.observe(DailyObservation(
            "resident-1", datetime.combine(day, datetime.min.time(), tzinfo=UTC),
            {"steps": float(100 + offset)}, {"steps": 0.9}, {"steps": True},
            {"evidence_tier": "real_device_longitudinal", "promoted": True},
        ))
    before = service.forecast("resident-1", as_of)

    for offset in range(30):
        day = as_of + timedelta(days=offset + 1)
        service.observe(DailyObservation(
            "resident-1", datetime.combine(day, datetime.min.time(), tzinfo=UTC),
            {"steps": 10000.0}, {"steps": 0.9}, {"steps": True},
            {"evidence_tier": "real_device_longitudinal", "promoted": True},
        ))
    after = service.forecast("resident-1", as_of)

    assert after == before


def test_model_error_degrades_to_cpu_rule_path() -> None:
    class BrokenModel:
        def predict_hazards(self, _features: object) -> tuple[float, ...]:
            raise RuntimeError("model unavailable")

    service, as_of = _service_with_history()
    service = PMCCService(model=BrokenModel(), observations=service.observations())

    forecast = service.forecast("resident-1", as_of)

    assert forecast.provenance["model"] == "cpu_rule_fallback"
    assert "model_inference_failed" in forecast.provenance["degradation_reasons"]


def test_insufficient_independent_uncertainty_members_abstains_instead_of_zero_width() -> None:
    class UnstableModel:
        def predict_hazards(self, _features: object) -> tuple[float, ...]:
            return (0.1,) * 7

        def predict_hazard_members(self, _features: object) -> tuple[tuple[float, ...], ...]:
            return ((0.0,) * 7, (1.0,) * 7)

    service, as_of = _service_with_history()
    service = PMCCService(model=UnstableModel(), observations=service.observations())

    forecast = service.forecast("resident-1", as_of)

    assert forecast.provenance["abstained"] is True
    assert "insufficient_uncertainty_members" in forecast.provenance["reasons"]
    assert forecast.provenance["uncertainty_width_72h"] is None


def test_synthetic_evidence_refuses_release_mode() -> None:
    service, as_of = _service_with_history(tier=EvidenceTier.SYNTHETIC_RESEARCH)

    forecast = service.forecast("resident-1", as_of, release_mode=True)

    assert forecast.provenance["abstained"] is True
    assert "non_release_evidence" in forecast.provenance["reasons"]
    assert forecast.provenance["promoted"] is False


def test_record_feedback_is_a_noop_when_storage_is_disabled() -> None:
    service = PMCCService()
    feedback = OutcomeFeedback("resident-1", datetime(2026, 8, 1, tzinfo=UTC), OutcomeType.NORMAL_ADL, 0.9)

    assert service.record_feedback(feedback) is None
    assert service.feedback_records() == ()


def test_service_rejects_unknown_subject() -> None:
    service, as_of = _service_with_history()

    with pytest.raises(ValueError, match="subject"):
        service.forecast("unknown", as_of)
