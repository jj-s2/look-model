from datetime import date

import pytest

from mental.contracts import DailyWellbeingObservation, DeliveryScope
from mental.pace_behavior import PACEBehaviorConfig, PACEBehaviorModel


def observation(day: int, *, steps: float = 4000.0, quality: float = 0.9, excluded_reason: str | None = None) -> DailyWellbeingObservation:
    return DailyWellbeingObservation(
        subject_alias="senior-1",
        local_day=date(2026, 8, day),
        sleep_hours=8.0,
        sleep_regularity=0.8,
        steps=steps,
        feature_quality={"sleep_hours": quality, "sleep_regularity": quality, "steps": quality},
        excluded_reason=excluded_reason,
    )


def build_model() -> PACEBehaviorModel:
    return PACEBehaviorModel("senior-1")


def test_config_has_conservative_defaults_and_rejects_unsafe_values() -> None:
    config = PACEBehaviorConfig()
    assert (config.min_effective_days, config.provisional_days, config.persistence_window, config.persistence_required) == (14, 7, 3, 2)
    with pytest.raises(ValueError, match="min_effective_days"):
        PACEBehaviorConfig(min_effective_days=6)
    with pytest.raises(ValueError, match="persistence_required"):
        PACEBehaviorConfig(persistence_window=3, persistence_required=4)


def test_cold_provisional_and_operational_states_are_explicit() -> None:
    model = build_model()
    states = [model.update(observation(day)).baseline_state for day in range(1, 15)]
    assert states[0] == "cold_start"
    assert states[6] == "provisional"
    assert states[13] == "operational"
    assert model.update(observation(15)).state == "observe"


def test_same_domain_two_of_three_anomalies_create_local_invitation_only() -> None:
    model = build_model()
    for day in range(1, 15):
        model.update(observation(day))
    first = model.update(observation(15, steps=1000.0))
    second = model.update(observation(16, steps=1000.0))
    assert first.state == "observe"
    assert second.state == "invite_candidate"
    assert second.action == "local_invite_short_checkin"
    assert second.delivery_scope is DeliveryScope.EXTERNAL_FORBIDDEN
    assert second.is_diagnosis is False
    assert second.fall_critical_eligible is False
    assert "activity" in second.evidence_domains


def test_cross_domain_anomalies_do_not_form_one_persistence_streak() -> None:
    model = build_model()
    for day in range(1, 15):
        model.update(observation(day))
    model.update(observation(15, steps=1000.0))
    event = model.update(
        DailyWellbeingObservation(
            "senior-1", date(2026, 8, 16), sleep_hours=5.0, sleep_regularity=0.8,
            steps=4000.0, feature_quality={"sleep_hours": 0.9, "sleep_regularity": 0.9, "steps": 0.9}
        )
    )
    assert event.state == "observe"
    assert event.action == "local_record_only"


def test_invalid_low_quality_and_travel_days_abstain_without_polluting_baseline() -> None:
    model = build_model()
    invalid = model.update(observation(1, quality=0.2))
    travel = model.update(observation(2, excluded_reason="travel"))
    assert invalid.state == "abstained"
    assert travel.state == "abstained"
    assert "low_quality" in invalid.abstention_reasons
    assert "excluded:travel" in travel.abstention_reasons
    assert invalid.coverage == 0.0
    assert model.effective_days == {}


def test_anomaly_day_is_frozen_out_and_event_ids_are_deterministic() -> None:
    model = build_model()
    for day in range(1, 15):
        model.update(observation(day))
    anomaly = model.update(observation(15, steps=1000.0))
    assert "baseline_frozen_on_anomaly" in anomaly.evidence_codes
    assert model.effective_days["steps"] == 14
    other = build_model()
    for day in range(1, 15):
        other.update(observation(day))
    assert other.update(observation(15, steps=1000.0)).event_id == anomaly.event_id
