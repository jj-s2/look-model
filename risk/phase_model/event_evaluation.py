"""Timestamp-based event matching metrics for continuous fall monitoring."""

from __future__ import annotations

import dataclasses
import math
from numbers import Real
from typing import Any


def _real(value: Any, name: str) -> float:
    """Return a finite real value, rejecting bools and string coercion."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


@dataclasses.dataclass(frozen=True)
class Alert:
    """A single emitted alert at an absolute monitoring timestamp."""

    alert_id: str
    timestamp: float

    def __post_init__(self) -> None:
        _identifier(self.alert_id, "alert_id")
        _real(self.timestamp, "timestamp")


@dataclasses.dataclass(frozen=True)
class TruthEvent:
    """A ground-truth event interval in absolute monitoring time."""

    event_id: str
    start: float
    end: float

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        start = _real(self.start, "start")
        end = _real(self.end, "end")
        if end < start:
            raise ValueError("end must be greater than or equal to start")


@dataclasses.dataclass(frozen=True)
class ContinuousMetrics:
    """Immutable summary of continuous event matching."""

    true_events: int
    matched_events: int
    alerts: int
    false_alerts: int
    duplicate_alerts: int
    event_recall: float
    event_precision: float
    false_alerts_per_hour: float
    duplicate_alert_rate: float
    median_delay_seconds: float | str
    p90_delay_seconds: float | str
    monitoring_hours: float

    def to_dict(self) -> dict[str, object]:
        return {
            "true_events": self.true_events,
            "matched_events": self.matched_events,
            "alerts": self.alerts,
            "false_alerts": self.false_alerts,
            "duplicate_alerts": self.duplicate_alerts,
            "event_recall": self.event_recall,
            "event_precision": self.event_precision,
            "false_alerts_per_hour": self.false_alerts_per_hour,
            "duplicate_alert_rate": self.duplicate_alert_rate,
            "median_delay_seconds": self.median_delay_seconds,
            "p90_delay_seconds": self.p90_delay_seconds,
            "monitoring_hours": self.monitoring_hours,
        }


def _percentile(values: list[float], quantile: float) -> float | str:
    if not values:
        return "unavailable"
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def evaluate_continuous_events(
    truth_events: list[TruthEvent],
    alerts: list[Alert],
    *,
    duration_seconds: float,
    tolerance_seconds: float,
) -> ContinuousMetrics:
    """Match alerts to event intervals and calculate auditable metrics.

    Alerts are processed chronologically.  Each alert can match at most one
    truth interval, choosing the earliest still-unmatched eligible interval.
    Later alerts inside a matched interval are duplicates; alerts outside every
    matched interval are false alerts.
    """

    duration = _real(duration_seconds, "duration_seconds")
    tolerance = _real(tolerance_seconds, "tolerance_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    if tolerance < 0:
        raise ValueError("tolerance_seconds must be non-negative")
    if type(truth_events) is not list or type(alerts) is not list:
        raise TypeError("truth_events and alerts must be lists")
    for item in truth_events:
        if type(item) is not TruthEvent:
            raise TypeError("truth_events must contain TruthEvent instances")
    for item in alerts:
        if type(item) is not Alert:
            raise TypeError("alerts must contain Alert instances")

    ordered_truth = sorted(truth_events, key=lambda event: (float(event.start), event.event_id))
    ordered_alerts = sorted(alerts, key=lambda alert: (float(alert.timestamp), alert.alert_id))
    matched: set[int] = set()
    delays: list[float] = []
    duplicate_count = 0
    false_count = 0

    for alert in ordered_alerts:
        timestamp = float(alert.timestamp)
        eligible = [
            index
            for index, event in enumerate(ordered_truth)
            if index not in matched
            and float(event.start) - tolerance <= timestamp <= float(event.end) + tolerance
        ]
        if eligible:
            index = eligible[0]
            matched.add(index)
            delays.append(timestamp - float(ordered_truth[index].start))
            continue

        duplicate = any(
            float(ordered_truth[index].start) - tolerance
            <= timestamp
            <= float(ordered_truth[index].end) + tolerance
            for index in matched
        )
        if duplicate:
            duplicate_count += 1
        else:
            false_count += 1

    true_count = len(ordered_truth)
    alert_count = len(ordered_alerts)
    matched_count = len(matched)
    monitoring_hours = duration / 3600.0
    duplicate_denominator = matched_count + duplicate_count
    return ContinuousMetrics(
        true_events=true_count,
        matched_events=matched_count,
        alerts=alert_count,
        false_alerts=false_count,
        duplicate_alerts=duplicate_count,
        event_recall=(matched_count / true_count) if true_count else 0.0,
        event_precision=(matched_count / alert_count) if alert_count else 0.0,
        # Every non-unique alert consumes operator attention.  Consequently
        # duplicates contribute to the false-alarm burden even though they are
        # reported separately from alerts outside all truth intervals.
        false_alerts_per_hour=(false_count + duplicate_count) / monitoring_hours,
        duplicate_alert_rate=(duplicate_count / duplicate_denominator)
        if duplicate_denominator
        else 0.0,
        median_delay_seconds=_percentile(delays, 0.5),
        p90_delay_seconds=_percentile(delays, 0.9),
        monitoring_hours=monitoring_hours,
    )
