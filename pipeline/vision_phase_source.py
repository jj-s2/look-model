"""One-cycle C6c frame source feeding pose observations into phase risk."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from core.events import DataQuality, EventType, SensorEvent, Source
from risk.phase_model.event_state import EventDecoder, EventDecoderConfig, FrameDecision
from risk.phase_model.release_config import RGPCReleaseConfig, load_release_config
from risk.phase_model.rg_predictor import RGPredictor
from risk.phase_model.schema import Phase, PhaseModelOutput, PoseObservation
from risk.phase_model.windows import DualTimescaleBuffer

from .live_service import SourceBatch


class _RGPCPhaseService:
    """Adapt one release predictor and decoder to the live pose contract."""

    def __init__(self, predictor: Any, decoder: EventDecoder, *, buffer: Any | None = None) -> None:
        self.predictor = predictor
        self.decoder = decoder
        self.buffer = buffer or DualTimescaleBuffer(
            short_frames=48, short_fps=10.0, long_frames=64, long_fps=2.0,
        )

    def observe(self, observation: PoseObservation) -> tuple[SensorEvent, ...]:
        window = self.buffer.append(observation)
        if window is None:
            score = min(observation.scores) if observation.scores else 0.0
            return (SensorEvent(
                timestamp=observation.timestamp, source=Source.VISION, event_type=EventType.POSE,
                payload={"tracking_id": observation.tracking_id, "quality_mode": "abstained"},
                quality=DataQuality(False, max(0.0, min(1.0, score)), False, "insufficient_window"),
            ),)
        output = self.predictor.predict(window)
        if not isinstance(output, PhaseModelOutput):
            raise TypeError("RG-PCNet predictor must return PhaseModelOutput")
        reliable = output.fall_decision is not None
        confidence = max(0.0, min(1.0, float(output.quality_score)))
        phase = self._decoder_phase(output.phase)
        transition = self.decoder.update(FrameDecision(
            timestamp=observation.timestamp.timestamp(),
            fall_probability=output.fall_event_prob,
            phase=phase,
            reliable=reliable,
        ))
        common_payload = {
            "subject_id": observation.tracking_id,
            "tracking_id": observation.tracking_id,
            "phase": phase,
            "event_state": transition.state,
        }
        if not reliable:
            return (SensorEvent(
                timestamp=observation.timestamp, source=Source.VISION, event_type=EventType.POSE,
                payload={**common_payload, "quality_mode": "abstained"},
                quality=DataQuality(False, confidence, False, "model_reliability_gate"),
            ),)
        quality = DataQuality(True, confidence, False, None)
        events: list[SensorEvent] = [SensorEvent(
            timestamp=observation.timestamp, source=Source.VISION, event_type=EventType.POSE,
            payload={**common_payload, "fall_probability": output.fall_event_prob}, quality=quality,
        )]
        if transition.emitted in {"fall_confirmed", "fall_recovered"}:
            recovered = transition.emitted == "fall_recovered"
            events.append(SensorEvent(
                timestamp=observation.timestamp, source=Source.VISION, event_type=EventType.FALL_EVENT,
                payload={
                    **common_payload,
                    "event_id": transition.event_id,
                    "confirmed": not recovered,
                    "recovered": recovered,
                    "recovery_confirmed": recovered,
                }, quality=quality,
            ))
        return tuple(events)

    @staticmethod
    def _decoder_phase(phase: Phase | None) -> str:
        if phase in {Phase.DESCENDING, Phase.IMPACT}:
            return "descent_or_impact"
        if phase in {Phase.FALLEN, Phase.RECOVERING}:
            return "postfall_or_recovery"
        if phase is Phase.NORMAL_ADL:
            return "normal"
        return "prefall"


class VisionPhaseSource:
    name = "vision"

    def __init__(self, stream: Any, pose_pipeline: Any, tracker: Any, phase_service: Any, *, clock=None, release_config: RGPCReleaseConfig | None = None) -> None:
        self.stream = stream
        self.pose_pipeline = pose_pipeline
        self.tracker = tracker
        self.phase_service = phase_service
        self.clock = clock
        self.decoder = getattr(phase_service, "decoder", None)
        self.release_config = release_config

    @classmethod
    def from_rgpc_release(
        cls,
        release_dir: str | Path,
        *,
        device: str = "auto",
        stream: Any = None,
        pose_pipeline: Any = None,
        tracker: Any = None,
        clock=None,
    ) -> "VisionPhaseSource":
        directory = Path(release_dir)
        config = load_release_config(directory / "release_config.json")
        predictor = RGPredictor.from_release(directory, device=device)
        decoder = EventDecoder(EventDecoderConfig(
            fall_threshold=config.fall_threshold,
            confirm_seconds=config.confirm_seconds,
            recovery_seconds=config.recovery_seconds,
            cooldown_seconds=config.cooldown_seconds,
        ))
        buffer = DualTimescaleBuffer(min_coverage=config.minimum_coverage)
        phase_service = _RGPCPhaseService(predictor, decoder, buffer=buffer)
        return cls(
            stream, pose_pipeline, tracker, phase_service,
            clock=clock, release_config=config,
        )

    def poll(self, now: datetime) -> SourceBatch:
        if self.stream is None:
            return SourceBatch((SensorEvent(
                timestamp=now, source=Source.VISION, event_type=EventType.AVAILABILITY,
                payload={"modality": "vision"},
                quality=DataQuality(False, 0.0, False, "stream_not_configured"),
            ),))
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
