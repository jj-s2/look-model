from datetime import datetime, timezone

import pytest

from risk.pmcc.schema import DailyObservation


def make_observation(**overrides):
    values = {
        "subject_id": "resident-7",
        "observed_at": datetime(2026, 8, 3, 9, 30, tzinfo=timezone.utc),
        "features": {"gait_speed": 0.82, "sleep_hours": None},
        "quality": {"vision": 0.95, "radar": 0.8},
        "availability": {"vision": True, "radar": True},
        "provenance": {"evidence_tier": "measured", "source": "daily-summary"},
    }
    values.update(overrides)
    return DailyObservation(**values)


def test_daily_observation_rejects_naive_timestamp():
    with pytest.raises(ValueError, match="timezone"):
        make_observation(observed_at=datetime(2026, 8, 3, 9, 30))


def test_daily_observation_rejects_empty_subject_id():
    with pytest.raises(ValueError, match="subject_id"):
        make_observation(subject_id="  ")


@pytest.mark.parametrize("quality", [{"vision": -0.01}, {"vision": 1.01}])
def test_daily_observation_rejects_quality_outside_unit_interval(quality):
    with pytest.raises(ValueError, match="quality"):
        make_observation(quality=quality, availability={"vision": True})


def test_daily_observation_rejects_boolean_quality():
    with pytest.raises(ValueError, match="quality"):
        make_observation(quality={"vision": True}, availability={"vision": True})


def test_daily_observation_rejects_quality_for_unavailable_source():
    with pytest.raises(ValueError, match="quality"):
        make_observation(quality={"vision": 0.2}, availability={"vision": False})


def test_daily_observation_keeps_missing_features_as_none():
    observation = make_observation()

    assert observation.features["sleep_hours"] is None
    assert observation.to_dict()["features"]["sleep_hours"] is None


def test_round_trip_preserves_synthetic_provenance():
    original = make_observation(
        provenance={"evidence_tier": "synthetic_research", "study_id": "R-12"}
    )

    assert DailyObservation.from_dict(original.to_dict()) == original
