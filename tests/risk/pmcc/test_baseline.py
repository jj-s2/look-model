from datetime import datetime, timedelta, timezone

import pytest

from risk.pmcc.baseline import BaselineManager
from risk.pmcc.schema import BaselineState, DailyObservation


START = datetime(2026, 7, 1, 9, tzinfo=timezone.utc)


def observation(day_offset, features, *, quality=0.9, available=True, provenance=None):
    names = set(features)
    return DailyObservation(
        subject_id="resident-7",
        observed_at=START + timedelta(days=day_offset),
        features=features,
        quality={name: quality for name in names},
        availability={name: available for name in names},
        provenance=provenance or {},
    )


def manager_with_days(count, feature="sleep_duration_minutes"):
    manager = BaselineManager(
        population_priors={feature: (480.0, 60.0)},
        feature_floors={feature: 1.0},
    )
    for day in range(count):
        manager.add(observation(day, {feature: 480.0 + day}))
    return manager


@pytest.mark.parametrize(
    ("days", "expected"),
    [(0, BaselineState.POPULATION_ONLY), (6, BaselineState.POPULATION_ONLY),
     (7, BaselineState.BLENDED), (13, BaselineState.BLENDED),
     (14, BaselineState.PERSONAL)],
)
def test_cold_start_state_uses_effective_days(days, expected):
    assert manager_with_days(days).state("sleep_duration_minutes") is expected


def test_each_feature_has_its_own_cold_start_state():
    manager = manager_with_days(14)
    manager.add(observation(14, {"trunk_sway": 0.2}))

    assert manager.state("sleep_duration_minutes") is BaselineState.PERSONAL
    assert manager.state("trunk_sway") is not BaselineState.PERSONAL


def test_directional_z_uses_median_mad_and_floor_when_mad_is_zero():
    manager = BaselineManager(
        population_priors={"sleep_duration_minutes": (480.0, 60.0)},
        feature_floors={"sleep_duration_minutes": 2.0},
    )
    for day in range(14):
        manager.add(observation(day, {"sleep_duration_minutes": 480.0}))

    assert manager.directional_z("sleep_duration_minutes", 484.0) == pytest.approx(2.0)


@pytest.mark.parametrize("default_floor", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_default_floor(default_floor):
    with pytest.raises(ValueError, match="default_floor"):
        BaselineManager(default_floor=default_floor)


@pytest.mark.parametrize(
    ("provenance", "quality", "available", "reason"),
    [
        ({"outcome": "confirmed_fall"}, 0.9, True, "confirmed_fall"),
        ({"outcome": "near_fall"}, 0.9, True, "near_fall"),
        ({}, 0.49, True, "low_quality"),
        ({"offline": True}, 0.9, True, "offline"),
        ({"environment_change": True}, 0.9, True, "environment_change"),
        ({"composite_anomaly": True}, 0.9, True, "composite_anomaly"),
    ],
)
def test_invalid_days_are_excluded_and_explained(provenance, quality, available, reason):
    manager = BaselineManager()
    item = observation(0, {"gait_speed": 0.8}, quality=quality, available=available, provenance=provenance)

    manager.add(item)

    assert manager.state("gait_speed") is BaselineState.POPULATION_ONLY
    assert manager.explain_exclusion(item.observed_at.date()) == reason


def test_fall_day_excludes_the_following_three_days():
    manager = BaselineManager()
    manager.add(observation(0, {"gait_speed": 0.8}, provenance={"outcome": "confirmed_fall"}))
    for day in range(1, 4):
        manager.add(observation(day, {"gait_speed": 0.8}))
    manager.add(observation(4, {"gait_speed": 0.8}))

    assert manager.state("gait_speed") is BaselineState.POPULATION_ONLY
    assert manager.explain_exclusion((START + timedelta(days=3)).date()) == "post_fall_cooldown"
    assert manager.quality("gait_speed") == pytest.approx(1 / 14)


def test_later_replacement_with_invalid_day_removes_prior_baseline_value():
    manager = BaselineManager()
    manager.add(observation(0, {"gait_speed": 0.8}))
    manager.add(observation(0, {"gait_speed": 0.1}, provenance={"composite_anomaly": True}))

    assert manager.state("gait_speed") is BaselineState.POPULATION_ONLY
    assert manager.directional_z("gait_speed", 0.8) is None
    assert manager.explain_exclusion(START.date()) == "composite_anomaly"


def test_valid_window_keeps_only_most_recent_thirty_valid_days():
    manager = BaselineManager(feature_floors={"gait_speed": 0.01})
    for day in range(31):
        manager.add(observation(day, {"gait_speed": float(day)}))

    assert manager.directional_z("gait_speed", 30.0) == pytest.approx((30.0 - 15.5) / (1.4826 * 7.5))
