"""Conservative, explainable late fusion of typed sensor events.

The engine never turns physiology into a diagnosis.  It only attaches available
physiology evidence to an existing wellbeing-change observation, and it keeps
fall events, pre-fall forecasts, and wellbeing changes as separate decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from numbers import Real
from typing import Literal, Sequence

from core.events import EventType, SensorEvent, Source


RiskLevel = Literal["info", "watch", "warning", "critical"]
DecisionQuality = Literal["multimodal", "vision_only", "screening_only", "degraded"]


@dataclass(frozen=True)
class RiskDecision:
    kind: Literal["fall_event", "fall_forecast", "wellbeing_change"]
    level: RiskLevel
    score: float
    reasons: tuple[str, ...]
    quality: DecisionQuality
    recommended_action: str
    subject_id: str = "unknown"
    timestamp: datetime | None = None
    recovery_confirmed: bool = False


class DecisionEngine:
    """Produce independently actionable, human-readable risk decisions."""

    def evaluate(self, events: Sequence[SensorEvent], now: datetime) -> list[RiskDecision]:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        decisions: list[RiskDecision] = []
        for event in events:
            if event.event_type is EventType.FALL_EVENT:
                decisions.append(self._fall_event(event, self._has_matching_radar(event, events)))
            elif event.event_type is EventType.FALL_FORECAST:
                decisions.append(self._fall_forecast(event, self._has_matching_radar(event, events)))
            elif event.event_type is EventType.WELLBEING_CHANGE:
                decisions.append(self._wellbeing_change(event, self._has_abnormal_physiology(event, events)))
        return decisions

    @staticmethod
    def _subject(event: SensorEvent) -> str:
        subject = event.payload.get("subject_id", event.payload.get("tracking_id", "unknown"))
        return str(subject)

    def _has_matching_radar(self, decision_event: SensorEvent, events: Sequence[SensorEvent]) -> bool:
        """Require actual, timely radar evidence for the same tracked person."""
        return any(
            candidate.source is Source.RADAR
            and candidate.event_type is not EventType.AVAILABILITY
            and candidate.quality.available
            and self._subject(candidate) == self._subject(decision_event)
            and abs(candidate.timestamp - decision_event.timestamp) <= self._FUSION_WINDOW
            for candidate in events
        )

    def _has_abnormal_physiology(self, decision_event: SensorEvent, events: Sequence[SensorEvent]) -> bool:
        return any(
            candidate.event_type is EventType.PHYSIOLOGY
            and candidate.quality.available
            and bool(candidate.payload.get("abnormal") or candidate.payload.get("anomaly"))
            and self._subject(candidate) == self._subject(decision_event)
            and abs(candidate.timestamp - decision_event.timestamp) <= self._FUSION_WINDOW
            for candidate in events
        )

    def _fall_event(self, event: SensorEvent, radar_matched: bool) -> RiskDecision:
        recovered = bool(event.payload.get("recovered") or event.payload.get("recovery_confirmed"))
        confirmed = bool(event.payload.get("confirmed"))
        quality: DecisionQuality = "multimodal" if radar_matched else "vision_only"
        quality_reason = "timely radar evidence corroborates this vision event" if radar_matched else "no timely matching radar evidence; vision evidence remains actionable"
        if recovered:
            return RiskDecision(
                "fall_event", "info", 0.0,
                ("confirmed recovery after prior fall", quality_reason), quality,
                "resume routine monitoring", self._subject(event), event.timestamp, True,
            )
        if confirmed:
            return RiskDecision(
                "fall_event", "critical", 1.0,
                ("confirmed fall evidence from vision", quality_reason), quality,
                "contact emergency responder and check the person immediately", self._subject(event), event.timestamp,
            )
        return RiskDecision(
            "fall_event", "warning", 0.6,
            ("unconfirmed fall event requires human check", quality_reason), quality,
            "check the person and continue monitoring", self._subject(event), event.timestamp,
        )

    def _fall_forecast(self, event: SensorEvent, radar_matched: bool) -> RiskDecision:
        raw_score = event.payload.get("score", event.payload.get("risk_score", 0.0))
        score = float(raw_score) if isinstance(raw_score, Real) and not isinstance(raw_score, bool) else 0.0
        score = max(0.0, min(1.0, score))
        level: RiskLevel = "info" if score < 0.3 else "watch" if score < 0.6 else "warning" if score < 0.9 else "critical"
        quality: DecisionQuality = "multimodal" if radar_matched else "vision_only"
        quality_reason = "timely radar evidence corroborates this forecast" if radar_matched else "no timely matching radar evidence; forecast is vision-only"
        action = "continue routine monitoring" if level == "info" else "check mobility and reduce fall hazards"
        return RiskDecision(
            "fall_forecast", level, score,
            (f"vision pre-fall score is {score:.2f}", quality_reason), quality,
            action, self._subject(event), event.timestamp,
        )

    def _wellbeing_change(self, event: SensorEvent, physiology_abnormal: bool) -> RiskDecision:
        sustained = bool(event.payload.get("sustained_change"))
        level: RiskLevel = "warning" if sustained else "watch"
        score = 0.7 if sustained else 0.4
        physiology_reason = (
            "physiology is supplemental context only; it is not a mental-health diagnosis"
            if physiology_abnormal
            else "no physiology context is available; no diagnosis is inferred"
        )
        action = "invite a voluntary short wellbeing check-in" if sustained else "observe trend and avoid diagnostic labels"
        return RiskDecision(
            "wellbeing_change", level, score,
            ("sustained wellbeing trend change" if sustained else "wellbeing trend change", physiology_reason),
            "screening_only", action, self._subject(event), event.timestamp,
        )
    _FUSION_WINDOW = timedelta(minutes=15)
