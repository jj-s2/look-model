from datetime import date, datetime, timedelta, timezone
from math import isnan

import pytest

from risk.pmcc.baseline import BaselineManager
from risk.pmcc.features import FeatureWindow, build_feature_window
from risk.pmcc.schema import ChangeEvent, DailyObservation, TemporalChain


UTC = timezone.utc


def observation(day: date, *, steps: float | None = 100.0, quality: float = 0.9, **provenance: object) -> DailyObservation:
    return DailyObservation(
        subject_id="resident-1",
        observed_at=datetime.combine(day, datetime.min.time(), tzinfo=UTC),
        features={"steps": steps},
        quality={"steps": quality if steps is not None else 0.0},
        availability={"steps": steps is not None},
        provenance=provenance,
    )


def test_feature_window_is_fourteen_days_oldest_to_newest_with_explicit_schema():
    as_of = date(2026, 8, 1)
    observations = [observation(as_of - timedelta(days=1), steps=120.0), observation(as_of, steps=130.0)]
    baseline = BaselineManager(population_priors={"steps": (100.0, 10.0)})
    for item in observations:
        baseline.add(item)

    window = build_feature_window(observations, baseline, (), as_of)

    assert len(window.values) == len(window.missing_mask) == len(window.quality) == 14
    assert window.feature_names == (
        "steps:directional_z",
        "steps:raw",
        "chain_strength",
        "confirmed_fall_count",
        "near_fall_count",
        "false_alarm_count",
        "baseline_personal_count",
    )
    assert window.missing_mask[0][:2] == (True, True)
    assert window.missing_mask[-1][:2] == (False, False)
    assert window.values[-1][1] == 130.0


def test_missing_values_are_nan_with_mask_instead_of_zero_normal_imputation():
    as_of = date(2026, 8, 1)
    item = observation(as_of, steps=None)
    baseline = BaselineManager(population_priors={"steps": (100.0, 10.0)})
    baseline.add(item)

    window = build_feature_window((item,), baseline, (), as_of)

    raw_index = window.feature_names.index("steps:raw")
    z_index = window.feature_names.index("steps:directional_z")
    assert window.missing_mask[-1][raw_index] is True
    assert window.missing_mask[-1][z_index] is True
    assert isnan(window.values[-1][raw_index])
    assert isnan(window.values[-1][z_index])
    assert window.quality[-1][raw_index] == 0.0


def test_missing_observation_days_mask_derived_features_instead_of_zero_normal_imputation():
    as_of = date(2026, 8, 1)
    item = observation(as_of, steps=130.0)
    baseline = BaselineManager(population_priors={"steps": (100.0, 10.0)})
    baseline.add(item)

    window = build_feature_window((item,), baseline, (), as_of)

    for name in ("chain_strength", "confirmed_fall_count", "near_fall_count", "false_alarm_count", "baseline_personal_count"):
        index = window.feature_names.index(name)
        assert window.missing_mask[-2][index] is True
        assert isnan(window.values[-2][index])
        assert window.quality[-2][index] == 0.0


def test_chain_strength_and_feedback_counts_are_retained_on_their_local_day():
    as_of = date(2026, 8, 1)
    item = observation(as_of, outcome="confirmed_fall")
    event = ChangeEvent("resident-1", item.observed_at, "sleep_duration", 480.0, 300.0, 0.9)
    chain = TemporalChain(
        subject_id="resident-1",
        created_at=item.observed_at,
        changes=(event,),
        provenance={"association_only": True, "score": 0.7},
    )
    baseline = BaselineManager(population_priors={"steps": (100.0, 10.0)})
    baseline.add(item)

    window = build_feature_window((item,), baseline, (chain,), as_of)

    assert window.values[-1][window.feature_names.index("chain_strength")] == 0.7
    assert window.values[-1][window.feature_names.index("confirmed_fall_count")] == 1.0


def test_feature_window_rejects_duplicate_or_ambiguous_local_days():
    as_of = date(2026, 8, 1)
    first = observation(as_of)
    second = DailyObservation(
        subject_id="resident-1",
        observed_at=datetime(2026, 8, 1, 12, tzinfo=timezone(timedelta(hours=8))),
        features={"steps": 120.0}, quality={"steps": 0.9}, availability={"steps": True},
    )
    baseline = BaselineManager(population_priors={"steps": (100.0, 10.0)})

    with pytest.raises(ValueError, match="local day|timezone"):
        build_feature_window((first, second), baseline, (), as_of)


def test_feature_window_validates_its_fixed_matrix_shapes():
    with pytest.raises(ValueError, match="same number"):
        FeatureWindow(
            values=((1.0,),) * 14,
            missing_mask=((False,),) * 14,
            quality=(),
            feature_names=("x",),
        )


def test_feature_window_requires_exactly_fourteen_daily_rows():
    with pytest.raises(ValueError, match="14"):
        FeatureWindow(
            values=((1.0,),) * 13,
            missing_mask=((False,),) * 13,
            quality=((1.0,),) * 13,
            feature_names=("x",),
        )


@pytest.mark.parametrize("value", (float("inf"), float("-inf")))
def test_feature_window_rejects_infinite_values_instead_of_marking_them_missing(value):
    with pytest.raises(ValueError, match="NaN|finite"):
        FeatureWindow(
            values=((value,),) * 14,
            missing_mask=((True,),) * 14,
            quality=((0.0,),) * 14,
            feature_names=("x",),
        )
