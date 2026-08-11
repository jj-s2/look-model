"""Conservative, offline trend evidence for voluntary wellbeing check-ins."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from math import isfinite
from numbers import Real
from statistics import median
from typing import Mapping


@dataclass(frozen=True)
class DailyPhysiologySummary:
    sleep_hours: float | None = None
    sleep_regularity: float | None = None


@dataclass(frozen=True)
class ActivitySummary:
    steps: float | None = None


@dataclass(frozen=True)
class CheckinResult:
    wellbeing: float | None = None
    self_harm_expression: bool = False


@dataclass(frozen=True)
class HumanReviewEvent:
    priority: str
    reason: str
    day: date
    requires_human_review: bool = True


@dataclass(frozen=True)
class TrendResult:
    baseline_ready: bool
    observation_evidence: tuple[str, ...]
    sustained_change: bool
    invite_short_checkin: bool
    human_review_event: HumanReviewEvent | None = None
    baseline_state: str = "cold_start"
    effective_days: Mapping[str, int] = field(default_factory=dict)
    abstention_reasons: tuple[str, ...] = ()


class WellbeingTrendAnalyzer:
    """Uses a seven-day personal baseline; a single unusual day never prompts."""

    def __init__(self, baseline_days: int = 14):
        self.baseline_days = max(14, int(baseline_days))
        self.provisional_days = 7
        self._history: list[tuple[date, dict[str, float]]] = []
        self._deviation_days: list[set[str]] = []
        self._last_day: date | None = None

    def update(
        self,
        day: date,
        physiology: DailyPhysiologySummary | None,
        activity: ActivitySummary | None,
        checkin: CheckinResult | None,
    ) -> TrendResult:
        if not isinstance(day, date):
            raise ValueError("day must be a date")
        if self._last_day is not None and day < self._last_day:
            raise ValueError("updates must be chronological by natural day")
        review = self._self_harm_event(day, checkin)
        values = self._values(physiology, activity, checkin)
        baseline = self._baseline()
        evidence = self._deviations(values, baseline) if len(self._history) >= self.baseline_days else ()
        is_new_day = day != self._last_day
        if is_new_day and self._last_day is not None and day != self._last_day + timedelta(days=1):
            self._deviation_days.clear()
        if is_new_day:
            self._deviation_days.append(set(evidence))
            self._deviation_days = self._deviation_days[-3:]
        self._upsert(day, values)
        self._last_day = day
        effective_days = self._effective_days()
        baseline_state = self._baseline_state(effective_days)
        sustained_keys = {
            key for key in {item for day_evidence in self._deviation_days for item in day_evidence}
            if sum(key in day_evidence for day_evidence in self._deviation_days) >= 2
        }
        sustained = baseline_state == "operational" and bool(sustained_keys)
        abstention_reasons = self._abstention_reasons(values, effective_days, baseline_state)
        return TrendResult(
            baseline_ready=baseline_state != "cold_start",
            observation_evidence=evidence,
            sustained_change=sustained,
            invite_short_checkin=sustained and review is None,
            human_review_event=review,
            baseline_state=baseline_state,
            effective_days=effective_days,
            abstention_reasons=abstention_reasons,
        )

    def _baseline(self) -> dict[str, float]:
        reference = self._history
        values_by_key: dict[str, list[float]] = {}
        for _, values in reference:
            for key, value in values.items():
                values_by_key.setdefault(key, []).append(value)
        return {
            key: median(values[: self.baseline_days])
            for key, values in values_by_key.items()
            if len(values) >= self.baseline_days
        }

    @staticmethod
    def _values(physiology, activity, checkin) -> dict[str, float]:
        def numeric(value, key: str, *, lower: float | None = None, upper: float | None = None) -> float | None:
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{key} must be numeric")
            result = float(value)
            if not isfinite(result):
                raise ValueError(f"{key} must be finite")
            if lower is not None and result < lower:
                raise ValueError(f"{key} must be >= {lower:g}")
            if upper is not None and result > upper:
                raise ValueError(f"{key} must be <= {upper:g}")
            return result

        candidates = {
            "sleep_hours": numeric(physiology.sleep_hours if physiology else None, "sleep_hours", lower=0.0, upper=24.0),
            "sleep_regularity": numeric(physiology.sleep_regularity if physiology else None, "sleep_regularity", lower=0.0, upper=1.0),
            "steps": numeric(activity.steps if activity else None, "steps", lower=0.0),
            "wellbeing": numeric(checkin.wellbeing if checkin else None, "wellbeing", lower=0.0, upper=10.0),
        }
        return {key: value for key, value in candidates.items() if value is not None}

    def _effective_days(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, values in self._history:
            for key in values:
                counts[key] = counts.get(key, 0) + 1
        return counts

    def _baseline_state(self, effective_days: Mapping[str, int]) -> str:
        if not effective_days:
            return "cold_start"
        active = [count for count in effective_days.values() if count > 0]
        if active and min(active) >= self.baseline_days:
            return "operational"
        if max(active) >= self.provisional_days:
            return "provisional"
        return "cold_start"

    @staticmethod
    def _abstention_reasons(values: Mapping[str, float], effective_days: Mapping[str, int], baseline_state: str) -> tuple[str, ...]:
        reasons: list[str] = []
        if not values:
            reasons.append("no_valid_observation")
        if baseline_state != "operational":
            reasons.append("baseline_not_operational")
        if any(count == 0 for count in effective_days.values()):
            reasons.append("feature_missing_history")
        return tuple(dict.fromkeys(reasons))

    @staticmethod
    def _deviations(values: dict[str, float], baseline: dict[str, float]) -> tuple[str, ...]:
        evidence = []
        if "steps" in values and baseline.get("steps", 0) > 0 and values["steps"] < baseline["steps"] * .7:
            evidence.append("activity_lower_than_baseline")
        if "sleep_hours" in values and "sleep_hours" in baseline and abs(values["sleep_hours"] - baseline["sleep_hours"]) >= 1.5:
            evidence.append("sleep_change_from_baseline")
        if "sleep_regularity" in values and "sleep_regularity" in baseline and values["sleep_regularity"] < baseline["sleep_regularity"] - .2:
            evidence.append("sleep_regularity_lower_than_baseline")
        if "wellbeing" in values and "wellbeing" in baseline and values["wellbeing"] <= baseline["wellbeing"] - 1:
            evidence.append("voluntary_wellbeing_lower_than_baseline")
        return tuple(evidence)

    def _upsert(self, day: date, values: dict[str, float]) -> None:
        self._history = [(saved_day, saved) for saved_day, saved in self._history if saved_day != day]
        self._history.append((day, values))
        self._history.sort(key=lambda entry: entry[0])

    @staticmethod
    def _self_harm_event(day: date, checkin: CheckinResult | None) -> HumanReviewEvent | None:
        if checkin is not None and checkin.self_harm_expression:
            return HumanReviewEvent(priority="highest", reason="self_harm_expression", day=day)
        return None
