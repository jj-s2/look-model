import dataclasses
import math

import pytest

from risk.phase_model.event_evaluation import (
    Alert,
    ContinuousMetrics,
    TruthEvent,
    evaluate_continuous_events,
)


def test_event_metrics_are_hand_auditable():
    truth = [TruthEvent("t1", 10.0, 15.0), TruthEvent("t2", 40.0, 45.0)]
    alerts = [Alert("a1", 11.0), Alert("a2", 12.0), Alert("a3", 70.0)]

    result = evaluate_continuous_events(
        truth, alerts, duration_seconds=3600.0, tolerance_seconds=5.0
    )

    assert result.true_events == 2
    assert result.matched_events == 1
    assert result.alerts == 3
    assert result.false_alerts == 1
    assert result.duplicate_alerts == 1
    assert result.event_recall == 0.5
    assert result.event_precision == pytest.approx(1 / 3)
    assert result.false_alerts_per_hour == 2.0
    assert result.duplicate_alert_rate == pytest.approx(0.5)
    assert result.median_delay_seconds == 1.0
    assert result.p90_delay_seconds == 1.0
    assert result.monitoring_hours == 1.0


def test_alert_matching_is_one_to_one_and_chronological():
    truth = [TruthEvent("t2", 40.0, 45.0), TruthEvent("t1", 10.0, 15.0)]
    alerts = [Alert("a2", 12.0), Alert("a1", 11.0)]

    result = evaluate_continuous_events(
        truth, alerts, duration_seconds=100.0, tolerance_seconds=0.0
    )

    assert result.matched_events == 1
    assert result.duplicate_alerts == 1
    assert result.false_alerts == 0


def test_matching_accepts_tolerance_and_preserves_negative_delay():
    result = evaluate_continuous_events(
        [TruthEvent("fall", 10.0, 15.0)],
        [Alert("early", 8.0)],
        duration_seconds=60.0,
        tolerance_seconds=2.0,
    )

    assert result.matched_events == 1
    assert result.median_delay_seconds == -2.0
    assert result.p90_delay_seconds == -2.0


def test_duplicate_is_only_inside_a_previously_matched_interval():
    result = evaluate_continuous_events(
        [TruthEvent("fall", 10.0, 15.0)],
        [Alert("match", 15.0), Alert("duplicate", 20.0), Alert("false", 21.0)],
        duration_seconds=3600.0,
        tolerance_seconds=5.0,
    )

    assert result.matched_events == 1
    assert result.duplicate_alerts == 1
    assert result.false_alerts == 1


def test_empty_matches_have_numeric_rates_and_unavailable_delays():
    result = evaluate_continuous_events(
        [], [Alert("a", 1.0)], duration_seconds=7200.0, tolerance_seconds=0.0
    )

    assert result.event_recall == 0.0
    assert result.event_precision == 0.0
    assert result.false_alerts_per_hour == 0.5
    assert result.duplicate_alert_rate == 0.0
    assert result.median_delay_seconds == "unavailable"
    assert result.p90_delay_seconds == "unavailable"


def test_continuous_metrics_is_frozen_and_to_dict_has_exact_schema():
    result = evaluate_continuous_events([], [], duration_seconds=1.0, tolerance_seconds=0.0)
    assert dataclasses.is_dataclass(result)
    assert dataclasses.is_dataclass(Alert("a", 0.0))
    assert dataclasses.is_dataclass(TruthEvent("t", 0.0, 1.0))
    assert dataclasses.is_dataclass(ContinuousMetrics)
    assert getattr(ContinuousMetrics, "__dataclass_params__").frozen is True
    assert set(result.to_dict()) == {
        "true_events",
        "matched_events",
        "alerts",
        "false_alerts",
        "duplicate_alerts",
        "event_recall",
        "event_precision",
        "false_alerts_per_hour",
        "duplicate_alert_rate",
        "median_delay_seconds",
        "p90_delay_seconds",
        "monitoring_hours",
    }


@pytest.mark.parametrize(
    "duration,tolerance",
    [(0.0, 0.0), (-1.0, 0.0), (1.0, -1.0), (math.inf, 0.0), (1.0, math.inf)],
)
def test_duration_and_tolerance_require_finite_valid_ranges(duration, tolerance):
    with pytest.raises((TypeError, ValueError)):
        evaluate_continuous_events([], [], duration_seconds=duration, tolerance_seconds=tolerance)


def test_strict_types_and_finite_times_are_validated():
    with pytest.raises(TypeError):
        TruthEvent("t", True, 1.0)
    with pytest.raises(ValueError):
        TruthEvent("t", 2.0, 1.0)
    with pytest.raises(TypeError):
        Alert("a", True)
    with pytest.raises(ValueError):
        Alert("a", math.nan)
    with pytest.raises(TypeError):
        evaluate_continuous_events(tuple(), [], duration_seconds=1.0, tolerance_seconds=0.0)


def test_inputs_are_not_mutated_and_ids_are_strict_strings():
    truth = [TruthEvent("t", 0.0, 1.0)]
    alerts = [Alert("a", 0.5)]
    truth_before = list(truth)
    alerts_before = list(alerts)
    with pytest.raises(TypeError):
        TruthEvent(1, 0.0, 1.0)
    evaluate_continuous_events(truth, alerts, duration_seconds=1.0, tolerance_seconds=0.0)
    assert truth == truth_before
    assert alerts == alerts_before
