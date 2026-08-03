"""One-cycle C6c frame source feeding pose observations into phase risk."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from core.events import DataQuality, EventType, SensorEvent, Source
from risk.phase_model.schema import PoseObservation

from .live_service import SourceBatch


class VisionPhaseSource:
    name = "vision"

    def __init__(self, stream: Any, pose_pipeline: Any, tracker: Any, phase_service: Any, *, clock=None) -> None:
        self.stream = stream
        self.pose_pipeline = pose_pipeline
        self.tracker = tracker
        self.phase_service = phase_service
        self.clock = clock

    def poll(self, now: datetime) -> SourceBatch:
        raw = self.stream.read_frame()
        if isinstance(raw, tuple) and len(raw) == 2:
            ok, frame = raw
        else:
            ok, frame = raw is not None, raw
        if not ok or frame is None:
            return SourceBatch((SensorEvent(
                timestamp=now, source=Source.VISION, event_type=EventType.AVAILABILITY,
                payload={"modality": "vision"}, quality=DataQuality(False, 0.0, False, "frame_unavailable"),
            ),))
        result = self.pose_pipeline.process(frame, now)
        if not result.bboxes:
            return SourceBatch((SensorEvent(
                timestamp=now, source=Source.VISION, event_type=EventType.POSE,
                payload={"no_person": True}, quality=DataQuality(True, 1.0, False, "no_person"),
            ),), frame=frame, frame_timestamp=now)
        track_items = self._tracks(result)
        events: list[SensorEvent] = []
        observations = []
        for index, bbox in enumerate(result.bboxes):
            points = self._person_value(result.keypoints, index)
            scores = self._person_value(result.keypoint_scores, index)
            if not points:
                continue
            points = tuple((float(point[0]), float(point[1])) for point in points)
            scores = tuple(float(value) for value in (scores or [0.0] * len(points)))
            visible = tuple(score >= 0.5 for score in scores)
            tracking_id = self._tracking_id(track_items, index)
            width, height = self._frame_size(frame, bbox)
            observation = PoseObservation(
                timestamp=now, tracking_id=tracking_id, keypoints=points, scores=scores,
                visible_mask=visible, bbox=tuple(float(value) for value in bbox),
                frame_size=(width, height), stream_fresh=True,
            )
            observations.append(observation)
            service_events = tuple(self.phase_service.observe(observation))
            events.extend(service_events)
            if not service_events:
                events.append(SensorEvent(
                    timestamp=now, source=Source.VISION, event_type=EventType.POSE,
                    payload={"tracking_id": tracking_id},
                    quality=DataQuality(True, sum(scores) / len(scores), False, None),
                ))
        if not observations and not events:
            events.append(SensorEvent(
                timestamp=now, source=Source.VISION, event_type=EventType.POSE,
                payload={"no_reliable_pose": True}, quality=DataQuality(False, 0.0, False, "no_reliable_pose"),
            ))
        return SourceBatch(tuple(events), frame=frame, frame_timestamp=now)

    def _tracks(self, result: Any) -> Any:
        try:
            return self.tracker.track(result)
        except TypeError:
            return self.tracker.track(result.bboxes, result.keypoints)

    @staticmethod
    def _person_value(values: Any, index: int) -> list[Any]:
        if values is None:
            return []
        try:
            candidate = values[index]
        except (IndexError, TypeError):
            return []
        return candidate.tolist() if hasattr(candidate, "tolist") else list(candidate)

    @staticmethod
    def _tracking_id(tracks: Any, index: int) -> str:
        try:
            item = tracks[index]
        except (IndexError, TypeError):
            return f"person-{index}"
        if isinstance(item, dict):
            return str(item.get("tracking_id", item.get("id", f"person-{index}")))
        return str(getattr(item, "tracking_id", getattr(item, "id", f"person-{index}")))

    @staticmethod
    def _frame_size(frame: Any, bbox: Any) -> tuple[int, int]:
        shape = getattr(frame, "shape", None)
        if shape is not None and len(shape) >= 2:
            return max(1, int(shape[1])), max(1, int(shape[0]))
        return max(1, int(float(bbox[2]) if len(bbox) > 2 else 1)), max(1, int(float(bbox[3]) if len(bbox) > 3 else 1))
