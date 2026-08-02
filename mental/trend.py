"""Conservative, offline trend evidence for voluntary wellbeing check-ins."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import median


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


class WellbeingTrendAnalyzer:
    """Uses a seven-day personal baseline; a single unusual day never prompts."""

    def __init__(self, baseline_days: int = 7):
        self.baseline_days = max(7, int(baseline_days))
        self._history: list[tuple[date, dict[str, float]]] = []
        self._deviation_streak = 0

    def update(
        self,
        day: date,
        physiology: DailyPhysiologySummary | None,
        activity: ActivitySummary | None,
        checkin: CheckinResult | None,
    ) -> TrendResult:
        if not isinstance(day, date):
            raise ValueError("day must be a date")
        review = self._self_harm_event(day, checkin)
        values = self._values(physiology, activity, checkin)
        baseline = self._baseline()
        evidence = self._deviations(values, baseline) if len(self._history) >= self.baseline_days else ()
        if evidence:
            self._deviation_streak += 1
        else:
            self._deviation_streak = 0
        self._upsert(day, values)
        sustained = len(self._history) > self.baseline_days and self._deviation_streak >= 2
        return TrendResult(
            baseline_ready=len(self._history) >= self.baseline_days,
            observation_evidence=evidence,
            sustained_change=sustained,
            invite_short_checkin=sustained and review is None,
            human_review_event=review,
        )

    def _baseline(self) -> dict[str, float]:
        reference = self._history[: self.baseline_days]
        keys = {key for _, values in reference for key in values}
        return {key: median([values[key] for _, values in reference if key in values]) for key in keys}

    @staticmethod
    def _values(physiology, activity, checkin) -> dict[str, float]:
        candidates = {
            "sleep_hours": physiology.sleep_hours if physiology else None,
            "sleep_regularity": physiology.sleep_regularity if physiology else None,
            "steps": activity.steps if activity else None,
            "wellbeing": checkin.wellbeing if checkin else None,
        }
        return {key: float(value) for key, value in candidates.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}

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
