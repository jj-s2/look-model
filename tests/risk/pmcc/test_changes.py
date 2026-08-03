from datetime import datetime, timedelta, timezone

from risk.pmcc.baseline import BaselineManager
from risk.pmcc.changes import detect_changes
from risk.pmcc.schema import DailyObservation


START = datetime(2026, 7, 1, 9, tzinfo=timezone.utc)


def observation(offset, features, *, quality=0.9):
    return DailyObservation(
        subject_id="resident-3",
        observed_at=START + timedelta(days=offset),
        features=features,
        quality={name: quality for name in features},
        availability={name: True for name in features},
    )


def test_detect_changes_requires_two_abnormal_observations_in_three_days():
    baseline = BaselineManager(
        population_priors={"sleep_duration_minutes": (480.0, 10.0)},
    )
    days = (
        observation(0, {"sleep_duration_minutes": 460.0}),
        observation(1, {"sleep_duration_minutes": 430.0}),
    )

    events = detect_changes(days, baseline)

    assert len(events) == 1
    assert events[0].occurred_at == days[1].observed_at
    assert events[0].provenance["persistence"] == 2 / 3
    assert events[0].provenance["severity"] == "severe"


def test_missing_day_never_counts_as_normal_for_persistence():
    baseline = BaselineManager(
        population_priors={"sleep_duration_minutes": (480.0, 10.0)},
    )
    days = (
        observation(0, {"sleep_duration_minutes": 460.0}),
        observation(2, {"sleep_duration_minutes": 460.0}),
    )

    events = detect_changes(days, baseline)

    assert len(events) == 1
    assert events[0].provenance["persistence"] == 2 / 3
