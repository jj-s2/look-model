"""Local phase-risk inference orchestration without device or network side effects."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Protocol

from core.events import DataQuality, EventType, SensorEvent, Source

from .quality import assess_window_quality
from .schema import PhaseModelOutput, PoseObservation
from .transitions import PhaseSmoother


class PhasePredictor(Protocol):
    def predict(self, window: object) -> PhaseModelOutput: ...


class PhaseRiskService:
    def __init__(self, predictor: PhasePredictor, buffer: object, smoother: PhaseSmoother | None = None, clock: Callable[[], datetime] | None = None) -> None:
        self.predictor = predictor
        self.buffer = buffer
        self.smoother = smoother or PhaseSmoother()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def observe(self, observation: PoseObservation) -> tuple[SensorEvent, ...]:
        timestamp = observation.timestamp
        tracking_id = observation.tracking_id
        window = self.buffer.append(observation)
        if window is None:
            score = min(observation.scores) if observation.scores else 0.0
            return (SensorEvent(
                timestamp=timestamp, source=Source.VISION, event_type=EventType.POSE,
                payload={"tracking_id": tracking_id, "quality_mode": "abstained"},
                quality=DataQuality(False, max(0.0, min(1.0, score)), False, "insufficient_window"),
            ),)
        assessment = assess_window_quality(window)
        if assessment.mode == "abstained":
            return (SensorEvent(
                timestamp=timestamp, source=Source.VISION, event_type=EventType.POSE,
                payload={"tracking_id": tracking_id, "quality_mode": assessment.mode},
                quality=DataQuality(False, assessment.score, False, ";".join(assessment.reasons)),
            ),)
        output = self.predictor.predict(window)
        if output.fall_decision is None:
            return (SensorEvent(
                timestamp=timestamp, source=Source.VISION, event_type=EventType.POSE,
                payload={"tracking_id": tracking_id, "quality_mode": "abstained", "reason": "model_reliability_gate"},
                quality=DataQuality(False, min(assessment.score, output.quality_score), False, "model_reliability_gate"),
            ),)
        confidence = min(assessment.score, output.quality_score)
        quality = DataQuality(True, confidence, False, None if confidence >= 0.55 else "degraded_pose_quality")
        common_payload = {
            "subject_id": tracking_id,
            "tracking_id": tracking_id,
            "phase": output.phase.value if output.phase else None,
            "quality_score": assessment.score,
        }
        events: list[SensorEvent] = [SensorEvent(
            timestamp=timestamp, source=Source.VISION, event_type=EventType.POSE,
            payload=common_payload, quality=quality,
        )]
        smoothed = self.smoother.update(output, timestamp)
        if smoothed.prefall_warning:
            events.append(SensorEvent(
                timestamp=timestamp, source=Source.VISION, event_type=EventType.PREFALL_WARNING,
                payload={**common_payload, "score": output.prefall_prob, "confirmed": True}, quality=quality,
            ))
        if output.fall_decision == 1:
            events.append(SensorEvent(
                timestamp=timestamp, source=Source.VISION, event_type=EventType.FALL_FORECAST,
                payload={**common_payload, "score": output.fall_event_prob}, quality=quality,
            ))
        if smoothed.confirmed_fall:
            events.append(SensorEvent(
                timestamp=timestamp, source=Source.VISION, event_type=EventType.FALL_EVENT,
                payload={**common_payload, "confirmed": True}, quality=quality,
            ))
        if smoothed.recovery_confirmed:
            events.append(SensorEvent(
                timestamp=timestamp, source=Source.VISION, event_type=EventType.FALL_EVENT,
                payload={**common_payload, "recovered": True, "recovery_confirmed": True}, quality=quality,
            ))
        return tuple(events)
