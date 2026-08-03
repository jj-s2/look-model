"""Fixed-shape, missingness-preserving inputs for PMCC survival models."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import isfinite, nan
from numbers import Real
from typing import Sequence

from .baseline import BaselineManager
from .schema import DailyObservation, TemporalChain


WINDOW_DAYS = 14
_OUTCOMES = ("confirmed_fall", "near_fall", "false_alarm")
_DERIVED_FEATURES = (
    "chain_strength",
    "confirmed_fall_count",
    "near_fall_count",
    "false_alarm_count",
    "baseline_personal_count",
)


@dataclass(frozen=True)
class FeatureWindow:
    """A chronological 14-day feature matrix with a separate missing mask."""

    values: tuple[tuple[float, ...], ...]
    missing_mask: tuple[tuple[bool, ...], ...]
    quality: tuple[tuple[float, ...], ...]
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        names = tuple(self.feature_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("feature_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        values = tuple(tuple(row) for row in self.values)
        masks = tuple(tuple(row) for row in self.missing_mask)
        qualities = tuple(tuple(row) for row in self.quality)
        if len(values) != WINDOW_DAYS:
            raise ValueError("FeatureWindow must contain exactly 14 daily rows")
        if not (len(values) == len(masks) == len(qualities)):
            raise ValueError("values, missing_mask, and quality must have the same number of rows")
        for row, mask, row_quality in zip(values, masks, qualities):
            if not (len(row) == len(mask) == len(row_quality) == len(names)):
                raise ValueError("feature rows must match feature_names")
            for value, missing, score in zip(row, mask, row_quality):
                if not isinstance(missing, bool):
                    raise ValueError("missing_mask values must be booleans")
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError("feature values must be numeric or NaN for missing data")
                if missing != (not isfinite(float(value))):
                    raise ValueError("missing feature values must be NaN and finite values must not be masked")
                if isinstance(score, bool) or not isinstance(score, Real) or not isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
                    raise ValueError("feature quality must be finite within [0, 1]")
                if missing and float(score) != 0.0:
                    raise ValueError("missing feature quality must be zero")
        object.__setattr__(self, "values", tuple(tuple(float(value) for value in row) for row in values))
        object.__setattr__(self, "missing_mask", masks)
        object.__setattr__(self, "quality", tuple(tuple(float(score) for score in row) for row in qualities))
        object.__setattr__(self, "feature_names", names)


def build_feature_window(
    observations: Sequence[DailyObservation],
    baseline: BaselineManager,
    chains: Sequence[TemporalChain],
    as_of: date,
) -> FeatureWindow:
    """Build a locally ordered 14-day window without treating absence as normal.

    Input records belong to one resident and one local timezone.  The output is
    oldest-to-newest; days outside the available range and unavailable values
    are represented by ``NaN`` plus ``missing_mask=True`` rather than zero.
    """
    if isinstance(as_of, datetime) or not isinstance(as_of, date):
        raise ValueError("as_of must be a date, not a datetime")
    if not isinstance(baseline, BaselineManager):
        raise ValueError("baseline must be a BaselineManager")
    records = tuple(observations)
    if not all(isinstance(item, DailyObservation) for item in records):
        raise ValueError("observations must contain DailyObservation records")
    if not all(isinstance(chain, TemporalChain) for chain in chains):
        raise ValueError("chains must contain TemporalChain records")
    _validate_calendar_ownership(records, tuple(chains))
    by_day = {item.observed_at.date(): item for item in records}
    feature_names = _feature_names(records)
    values: list[tuple[float, ...]] = []
    masks: list[tuple[bool, ...]] = []
    qualities: list[tuple[float, ...]] = []
    for offset in range(WINDOW_DAYS - 1, -1, -1):
        day = as_of - timedelta(days=offset)
        row, mask, score = _row_for_day(by_day.get(day), baseline, chains, day, feature_names)
        values.append(row)
        masks.append(mask)
        qualities.append(score)
    return FeatureWindow(tuple(values), tuple(masks), tuple(qualities), feature_names)


def _feature_names(records: Sequence[DailyObservation]) -> tuple[str, ...]:
    raw_names = sorted({name for record in records for name in record.features})
    return tuple(name for raw in raw_names for name in (f"{raw}:directional_z", f"{raw}:raw")) + _DERIVED_FEATURES


def _row_for_day(
    observation: DailyObservation | None,
    baseline: BaselineManager,
    chains: Sequence[TemporalChain],
    day: date,
    feature_names: tuple[str, ...],
) -> tuple[tuple[float, ...], tuple[bool, ...], tuple[float, ...]]:
    feature_bases = tuple(name[:-len(":directional_z")] for name in feature_names if name.endswith(":directional_z"))
    row: list[float] = []
    mask: list[bool] = []
    scores: list[float] = []
    for feature in feature_bases:
        value, available, quality = _daily_value(observation, feature)
        z_value = baseline.directional_z(feature, value) if available else None
        z_quality = quality * baseline.quality(feature) if z_value is not None else 0.0
        _append(row, mask, scores, z_value, z_quality)
        _append(row, mask, scores, value if available else None, quality if available else 0.0)
    chain_score = _chain_strength(chains, day)
    _append(row, mask, scores, chain_score, 1.0)
    outcomes = _outcome_counts(observation)
    for outcome in _OUTCOMES:
        _append(row, mask, scores, outcomes[outcome], 1.0)
    personal_count = sum(baseline.state(feature).value == "personal" for feature in feature_bases)
    _append(row, mask, scores, float(personal_count), 1.0)
    return tuple(row), tuple(mask), tuple(scores)


def _append(row: list[float], mask: list[bool], scores: list[float], value: float | None, quality: float) -> None:
    if value is None:
        row.append(nan)
        mask.append(True)
        scores.append(0.0)
    else:
        row.append(float(value))
        mask.append(False)
        scores.append(float(quality))


def _daily_value(observation: DailyObservation | None, feature: str) -> tuple[float | None, bool, float]:
    if observation is None or observation.availability.get(feature) is False:
        return None, False, 0.0
    value = observation.features.get(feature)
    if value is None:
        return None, False, 0.0
    if feature in observation.quality:
        return value, True, observation.quality[feature]
    source_scores = [score for source, score in observation.quality.items() if observation.availability.get(source, True)]
    return value, bool(source_scores), max(source_scores, default=0.0)


def _chain_strength(chains: Sequence[TemporalChain], day: date) -> float:
    total = 0.0
    for chain in chains:
        if chain.created_at.date() != day:
            continue
        score = chain.provenance.get("score", 0.0)
        if isinstance(score, bool) or not isinstance(score, Real) or not isfinite(float(score)):
            raise ValueError("temporal chain score must be a finite number")
        total += float(score)
    return min(1.0, max(0.0, total))


def _outcome_counts(observation: DailyObservation | None) -> dict[str, float]:
    counts = {outcome: 0.0 for outcome in _OUTCOMES}
    if observation is None:
        return counts
    outcome = observation.provenance.get("outcome")
    if outcome in counts:
        counts[outcome] = 1.0
    return counts


def _validate_calendar_ownership(records: Sequence[DailyObservation], chains: Sequence[TemporalChain]) -> None:
    subjects = {record.subject_id for record in records}
    if len(subjects) > 1:
        raise ValueError("observations must belong to one subject")
    if subjects and any(chain.subject_id not in subjects for chain in chains):
        raise ValueError("chains must belong to the observation subject")
    timestamps = [record.observed_at for record in records] + [chain.created_at for chain in chains]
    if timestamps:
        first = timestamps[0]
        for timestamp in timestamps[1:]:
            if timestamp.tzinfo != first.tzinfo:
                raise ValueError("timezone mismatch makes local day ownership ambiguous")
    days = [record.observed_at.date() for record in records]
    if len(days) != len(set(days)):
        raise ValueError("at most one observation is allowed for each local day")
