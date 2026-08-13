"""Robust, staged baselines for daily PMCC observations.

The manager intentionally keeps the raw daily records and derives the eligible
reference set on demand.  A later correction that marks a previously accepted
day invalid therefore removes it from every feature reference automatically.
"""
from __future__ import annotations

from datetime import date
from math import isfinite
from statistics import median
from typing import Mapping

from .schema import BaselineState, DailyObservation


class BaselineManager:
    """Maintain per-feature robust references over the latest 30 valid days."""

    _BLENDED_START = 7
    _PERSONAL_START = 14
    _WINDOW_SIZE = 30
    _QUALITY_THRESHOLD = 0.5

    def __init__(
        self,
        population_priors: Mapping[str, tuple[float, float]] | None = None,
        feature_floors: Mapping[str, float] | None = None,
        default_floor: float = 1e-6,
    ) -> None:
        if (
            not isinstance(default_floor, (int, float))
            or isinstance(default_floor, bool)
            or not isfinite(default_floor)
            or default_floor <= 0
        ):
            raise ValueError("default_floor must be a positive number")
        self._default_floor = float(default_floor)
        self._priors = self._clean_priors(population_priors or {})
        self._floors = self._clean_floors(feature_floors or {})
        self._observations: dict[date, DailyObservation] = {}
        self._subject_id: str | None = None

    def add(self, observation: DailyObservation) -> None:
        """Store or replace one local-calendar daily observation.

        A manager represents one resident.  Replacing a day is deliberate: it
        supports late quality/outcome corrections without retaining stale data.
        """
        if not isinstance(observation, DailyObservation):
            raise ValueError("observation must be a DailyObservation")
        if self._subject_id is None:
            self._subject_id = observation.subject_id
        elif self._subject_id != observation.subject_id:
            raise ValueError("BaselineManager accepts observations for one subject")
        day = observation.observed_at.date()
        current = self._observations.get(day)
        if current is None or observation.observed_at >= current.observed_at:
            self._observations[day] = observation

    def state(self, feature: str) -> BaselineState:
        count = len(self._valid_feature_values(feature))
        if count >= self._PERSONAL_START:
            return BaselineState.PERSONAL
        if count >= self._BLENDED_START:
            return BaselineState.BLENDED
        return BaselineState.POPULATION_ONLY

    def directional_z(self, feature: str, value: float) -> float | None:
        """Return a signed robust deviation; ``None`` means no usable reference."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("value must be a finite numeric value")
        values = self._valid_feature_values(feature)
        reference = self._reference(feature, values)
        if reference is None:
            return None
        center, spread = reference
        return (float(value) - center) / spread

    def quality(self, feature: str) -> float:
        """Return feature baseline readiness in [0, 1], capped at 14 valid days."""
        return min(1.0, len(self._valid_feature_values(feature)) / self._PERSONAL_START)

    def explain_exclusion(self, day: date) -> str | None:
        if not isinstance(day, date):
            raise ValueError("day must be a date")
        return self._exclusion_reasons().get(day)

    def _valid_feature_values(self, feature: str) -> list[float]:
        if not isinstance(feature, str) or not feature:
            raise ValueError("feature must be a non-empty string")
        reasons = self._exclusion_reasons()
        values: list[tuple[date, float]] = []
        for day, observation in self._observations.items():
            if day in reasons:
                continue
            value = observation.features.get(feature)
            if value is None or not self._feature_is_usable(observation, feature):
                continue
            values.append((day, value))
        values.sort(key=lambda item: item[0])
        return [value for _, value in values[-self._WINDOW_SIZE:]]

    def _reference(self, feature: str, values: list[float]) -> tuple[float, float] | None:
        count = len(values)
        personal = self._robust_reference(feature, values) if values else None
        prior = self._priors.get(feature)
        if count >= self._PERSONAL_START:
            return personal
        if count >= self._BLENDED_START:
            if prior is None:
                return personal
            assert personal is not None
            weight = (count - (self._BLENDED_START - 1)) / (
                self._PERSONAL_START - (self._BLENDED_START - 1)
            )
            return (
                (1.0 - weight) * prior[0] + weight * personal[0],
                max(self._floor(feature), (1.0 - weight) * prior[1] + weight * personal[1]),
            )
        return prior

    def _robust_reference(self, feature: str, values: list[float]) -> tuple[float, float]:
        center = float(median(values))
        mad = float(median([abs(value - center) for value in values]))
        return center, max(self._floor(feature), 1.4826 * mad)

    def _floor(self, feature: str) -> float:
        return self._floors.get(feature, self._default_floor)

    def _feature_is_usable(self, observation: DailyObservation, feature: str) -> bool:
        if observation.availability.get(feature) is False:
            return False
        if feature in observation.quality:
            return observation.quality[feature] >= self._QUALITY_THRESHOLD
        source_qualities = [
            score
            for source, score in observation.quality.items()
            if observation.availability.get(source, True)
        ]
        return bool(source_qualities) and max(source_qualities) >= self._QUALITY_THRESHOLD

    def _exclusion_reasons(self) -> dict[date, str]:
        reasons: dict[date, str] = {}
        fall_days: list[date] = []
        for day, observation in self._observations.items():
            reason = self._intrinsic_exclusion(observation)
            if reason is not None:
                reasons[day] = reason
            if reason in {"confirmed_fall", "near_fall"}:
                fall_days.append(day)
        for fall_day in fall_days:
            for offset in range(1, 4):
                cooldown_day = date.fromordinal(fall_day.toordinal() + offset)
                if cooldown_day in self._observations and cooldown_day not in reasons:
                    reasons[cooldown_day] = "post_fall_cooldown"
        return reasons

    def _intrinsic_exclusion(self, observation: DailyObservation) -> str | None:
        provenance = observation.provenance
        outcome = str(provenance.get("outcome", ""))
        if outcome in {"confirmed_fall", "near_fall"}:
            return outcome
        if provenance.get("offline") is True or (observation.availability and not any(observation.availability.values())):
            return "offline"
        if provenance.get("environment_change") is True:
            return "environment_change"
        if provenance.get("composite_anomaly") is True:
            return "composite_anomaly"
        usable_quality = [
            quality for source, quality in observation.quality.items() if observation.availability.get(source, True)
        ]
        if not usable_quality or max(usable_quality) < self._QUALITY_THRESHOLD:
            return "low_quality"
        return None

    @staticmethod
    def _clean_priors(priors: Mapping[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        clean: dict[str, tuple[float, float]] = {}
        for feature, pair in priors.items():
            if not isinstance(feature, str) or not feature or not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("population_priors must map names to (center, spread) tuples")
            center, spread = pair
            if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not isfinite(item) for item in pair) or spread <= 0:
                raise ValueError("population prior center/spread must be finite and spread positive")
            clean[feature] = (float(center), float(spread))
        return clean

    @staticmethod
    def _clean_floors(floors: Mapping[str, float]) -> dict[str, float]:
        clean: dict[str, float] = {}
        for feature, floor in floors.items():
            if not isinstance(feature, str) or not feature or isinstance(floor, bool) or not isinstance(floor, (int, float)) or not isfinite(floor) or floor <= 0:
                raise ValueError("feature floors must be positive finite numbers")
            clean[feature] = float(floor)
        return clean
