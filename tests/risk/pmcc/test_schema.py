import json
from datetime import datetime, timezone
from math import nan

import pytest

from risk.pmcc.schema import (
    BaselineState,
    DailyObservation,
    DecisionBand,
    EvidenceTier,
    OutcomeType,
)


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


def test_provenance_is_recursively_normalized_for_json_serialization():
    observation = make_observation(
        provenance={
            "evidence_tier": EvidenceTier.REAL_DEVICE_LONGITUDINAL,
            "captured_at": datetime(2026, 8, 3, 9, 31, tzinfo=timezone.utc),
            "nested": {"labels": ("daily", "verified")},
        }
    )

    encoded = observation.to_dict()

    assert encoded["provenance"]["evidence_tier"] == "real_device_longitudinal"
    assert encoded["provenance"]["captured_at"] == "2026-08-03T09:31:00+00:00"
    assert encoded["provenance"]["nested"]["labels"] == ["daily", "verified"]
    json.dumps(encoded)


@pytest.mark.parametrize("provenance", [{"labels": {"unsupported"}}, {"score": nan}])
def test_daily_observation_rejects_non_json_safe_provenance(provenance):
    with pytest.raises(ValueError, match="provenance"):
        make_observation(provenance=provenance)


@pytest.mark.parametrize("timestamp", [None, 42])
def test_from_dict_rejects_non_string_timestamp_with_value_error(timestamp):
    data = make_observation().to_dict()
    data["observed_at"] = timestamp

    with pytest.raises(ValueError, match="observed_at"):
        DailyObservation.from_dict(data)


@pytest.mark.parametrize("field", ["features", "quality", "availability"])
def test_from_dict_rejects_missing_required_mappings(field):
    data = make_observation().to_dict()
    data.pop(field)

    with pytest.raises(ValueError, match=field):
        DailyObservation.from_dict(data)


def test_pmcc_enums_use_specified_wire_values():
    assert {item.value for item in EvidenceTier} == {
        "real_device_longitudinal", "real_public", "synthetic_research", "offline_fixture"
    }
    assert {item.value for item in BaselineState} == {"population_only", "blended", "personal"}
    assert {item.value for item in DecisionBand} == {"low", "elevated", "high", "very_high"}
    assert {item.value for item in OutcomeType} == {
        "confirmed_fall", "near_fall", "normal_adl", "false_alarm", "unknown"
    }
