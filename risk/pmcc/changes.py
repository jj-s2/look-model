"""Directional, persistence-aware PMCC change event detection."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from math import isfinite
from typing import Sequence

from .baseline import BaselineManager
from .schema import ChangeEvent, DailyObservation


ORDINARY_Z = 1.5
SEVERE_Z = 2.5
_PERSISTENCE_WINDOW = timedelta(days=3)


def detect_changes(
    observations: Sequence[DailyObservation], baseline: BaselineManager
) -> tuple[ChangeEvent, ...]:
    """Return changes that occur on at least two observed days in 72 hours.

    An absent feature is deliberately ignored rather than converted into a
    normal reading.  This makes a missing day unable to erase a concerning
    sequence of observations.
    """
    if not isinstance(baseline, BaselineManager):
        raise ValueError("baseline must be a BaselineManager")
    ordered = tuple(observations)
    if not all(isinstance(item, DailyObservation) for item in ordered):
        raise ValueError("observations must contain DailyObservation records")
    _require_strictly_chronological(
        tuple(item.observed_at for item in ordered), "observations"
    )
    if not ordered:
        return ()
    subject_id = ordered[0].subject_id
    if any(item.subject_id != subject_id for item in ordered):
        raise ValueError("observations must belong to one subject")

    candidates: dict[str, list[tuple[DailyObservation, float]]] = defaultdict(list)
    for observation in ordered:
        for feature, value in observation.features.items():
            if value is None or observation.availability.get(feature) is False:
                continue
            quality = _quality(observation, feature)
            if quality < 0.5:
                continue
            z_score = baseline.directional_z(feature, value)
            if z_score is not None and abs(z_score) >= ORDINARY_Z:
                candidates[feature].append((observation, z_score))

    events: list[ChangeEvent] = []
    for feature, readings in candidates.items():
        for index, (observation, z_score) in enumerate(readings):
            window = [
                item for item in readings[: index + 1]
                if observation.observed_at - item[0].observed_at <= _PERSISTENCE_WINDOW
            ]
            if len(window) < 2:
                continue
            persistence = min(1.0, len(window) / 3.0)
            previous = window[-2][0].features[feature]
            quality = _quality(observation, feature)
            severity = "severe" if abs(z_score) >= SEVERE_Z else "ordinary"
            events.append(ChangeEvent(
                subject_id=observation.subject_id,
                occurred_at=observation.observed_at,
                feature=feature,
                previous_value=previous,
                current_value=observation.features[feature],
                confidence=min(1.0, abs(z_score) / SEVERE_Z) * quality,
                provenance={
                    "event_id": f"{feature}:{observation.observed_at.isoformat()}",
                    "directional_z": z_score,
                    "severity": severity,
                    "persistence": persistence,
                    "quality": quality,
                    "association_only": True,
                },
            ))
    return tuple(events)


def _quality(observation: DailyObservation, feature: str) -> float:
    value = observation.quality.get(feature)
    if value is None:
        available = [
            score for source, score in observation.quality.items()
            if observation.availability.get(source, True)
        ]
        value = max(available, default=0.0)
    if not isfinite(value):  # schema already rejects this; retain a defensive boundary.
        return 0.0
    return float(value)


def _require_strictly_chronological(timestamps: tuple[datetime, ...], name: str) -> None:
    if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{name} must be supplied in strictly chronological order")
