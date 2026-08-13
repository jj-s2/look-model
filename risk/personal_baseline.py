"""Seven-day robust personal gait baseline without external dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite
from statistics import median
from typing import Mapping


@dataclass(frozen=True)
class BaselineScore:
    baseline_ready: bool
    score: float
    deviations: Mapping[str, float]
    unready_features: tuple[str, ...] = ()


class RobustPersonalBaseline:
    """Maintains median/IQR references from distinct valid natural days.

    Invalid (including pre-labelled anomalous) days are deliberately discarded,
    so they cannot contaminate the reference distribution.
    """

    def __init__(self, min_valid_days: int = 7, iqr_floor: float = 0.01):
        self.min_valid_days = max(int(min_valid_days), 1)
        self.iqr_floor = max(float(iqr_floor), 1e-9)
        self._days: dict[date, dict[str, float]] = {}

    @property
    def ready(self) -> bool:
        counts = self.feature_day_counts
        return bool(counts) and all(count >= self.min_valid_days for count in counts.values())

    @property
    def valid_day_count(self) -> int:
        return len(self._days)

    @property
    def feature_day_counts(self) -> dict[str, int]:
        return {name: len(values) for name, values in self._values_by_feature().items()}

    @property
    def reference(self) -> dict[str, float]:
        return {name: median(values) for name, values in self._values_by_feature().items()}

    @property
    def iqr(self) -> dict[str, float]:
        return {name: max(self._quartile(values, .75) - self._quartile(values, .25), self.iqr_floor)
                for name, values in self._values_by_feature().items()}

    def add_day(self, day: date, features: Mapping[str, float], valid: bool) -> None:
        if not valid or not isinstance(day, date):
            return
        clean = {str(name): float(value) for name, value in features.items()
                 if isinstance(value, (int, float)) and isfinite(float(value))}
        if clean:
            self._days[day] = clean

    def score(self, features: Mapping[str, float]) -> BaselineScore:
        reference, spread, deviations = self.reference, self.iqr, {}
        unready = []
        for name, value in features.items():
            if not isinstance(value, (int, float)) or not isfinite(float(value)):
                continue
            if self.feature_day_counts.get(name, 0) < self.min_valid_days:
                unready.append(str(name))
            elif name in reference:
                deviations[name] = abs(float(value) - reference[name]) / spread[name]
        if unready or not deviations:
            return BaselineScore(False, 0.0, deviations, tuple(unready))
        average = sum(deviations.values()) / len(deviations) if deviations else 0.0
        return BaselineScore(True, min(1.0, average / 3.0), deviations, ())

    def _values_by_feature(self) -> dict[str, list[float]]:
        grouped: dict[str, list[float]] = {}
        for values in self._days.values():
            for name, value in values.items():
                grouped.setdefault(name, []).append(value)
        return grouped

    @staticmethod
    def _quartile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * fraction
        lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
