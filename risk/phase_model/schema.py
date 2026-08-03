"""Dependency-light contracts for the PA-DTSF phase model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import math
from numbers import Real
from typing import Mapping


class Phase(str, Enum):
    NORMAL_ADL = "normal_adl"
    PREFALL_ABNORMAL = "prefall_abnormal"
    DESCENDING = "descending"
    IMPACT = "impact"
    FALLEN = "fallen"
    RECOVERING = "recovering"


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _tuple_floats(values: object, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)) or len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    result = tuple(_probability(v, name) for v in values)
    return result


@dataclass(frozen=True)
class PoseObservation:
    timestamp: datetime
    tracking_id: str
    keypoints: tuple[tuple[float, float], ...]
    scores: tuple[float, ...]
    visible_mask: tuple[bool, ...]
    bbox: tuple[float, float, float, float] | None
    frame_size: tuple[int, int]
    stream_fresh: bool

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if not isinstance(self.tracking_id, str) or not self.tracking_id.strip():
            raise ValueError("tracking_id must be non-empty")
        if not isinstance(self.keypoints, (tuple, list)) or not self.keypoints:
            raise ValueError("keypoints must be a non-empty sequence")
        if len(self.scores) != len(self.keypoints) or len(self.visible_mask) != len(self.keypoints):
            raise ValueError("scores and visible_mask must match keypoints")
        for point in self.keypoints:
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise ValueError("each keypoint must be an x/y pair")
            if any(isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(float(item)) for item in point):
                raise ValueError("keypoint coordinates must be finite numbers")
        for score in self.scores:
            _probability(score, "score")
        if any(not isinstance(item, bool) for item in self.visible_mask):
            raise ValueError("visible_mask must contain booleans")
        if self.bbox is not None:
            if len(self.bbox) != 4 or any(isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(float(item)) for item in self.bbox):
                raise ValueError("bbox must contain four finite numbers")
        if len(self.frame_size) != 2 or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in self.frame_size):
            raise ValueError("frame_size must contain two positive integers")
        if not isinstance(self.stream_fresh, bool):
            raise ValueError("stream_fresh must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "tracking_id": self.tracking_id,
            "keypoints": [list(point) for point in self.keypoints],
            "scores": list(self.scores),
            "visible_mask": list(self.visible_mask),
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "frame_size": list(self.frame_size),
            "stream_fresh": self.stream_fresh,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PoseObservation":
        try:
            return cls(
                timestamp=datetime.fromisoformat(str(data["timestamp"])),
                tracking_id=str(data["tracking_id"]),
                keypoints=tuple(tuple(float(value) for value in point) for point in data["keypoints"]),  # type: ignore[union-attr]
                scores=tuple(float(value) for value in data["scores"]),  # type: ignore[union-attr]
                visible_mask=tuple(bool(value) for value in data["visible_mask"]),  # type: ignore[union-attr]
                bbox=tuple(float(value) for value in data["bbox"]) if data.get("bbox") is not None else None,  # type: ignore[union-attr]
                frame_size=tuple(int(value) for value in data["frame_size"]),  # type: ignore[union-attr]
                stream_fresh=bool(data["stream_fresh"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid pose observation") from error


@dataclass(frozen=True)
class PhaseModelOutput:
    phase_probs: tuple[float, ...]
    fall_event_prob: float
    prefall_prob: float
    recovery_prob: float
    quality_score: float
    embedding_version: str
    model_version: str
    phase: Phase | None = None

    def __post_init__(self) -> None:
        if len(self.phase_probs) != len(Phase):
            raise ValueError("phase_probs must contain one probability for each phase")
        probabilities = tuple(_probability(value, "phase_probs") for value in self.phase_probs)
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("phase_probs must sum to 1")
        object.__setattr__(self, "phase_probs", probabilities)
        for name in ("fall_event_prob", "prefall_prob", "recovery_prob", "quality_score"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        for name in ("embedding_version", "model_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.phase is not None and not isinstance(self.phase, Phase):
            raise ValueError("phase must be a Phase")
        if self.phase is None:
            object.__setattr__(self, "phase", tuple(Phase)[probabilities.index(max(probabilities))])

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "padtfs.phase_output.v1",
            "phase_probs": list(self.phase_probs),
            "fall_event_prob": self.fall_event_prob,
            "prefall_prob": self.prefall_prob,
            "recovery_prob": self.recovery_prob,
            "quality_score": self.quality_score,
            "embedding_version": self.embedding_version,
            "model_version": self.model_version,
            "phase": self.phase.value if self.phase else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PhaseModelOutput":
        if data.get("schema_version") != "padtfs.phase_output.v1":
            raise ValueError("unsupported phase output schema")
        try:
            phase = data.get("phase")
            return cls(
                phase_probs=tuple(float(value) for value in data["phase_probs"]),  # type: ignore[union-attr]
                fall_event_prob=data["fall_event_prob"],  # type: ignore[arg-type]
                prefall_prob=data["prefall_prob"],  # type: ignore[arg-type]
                recovery_prob=data["recovery_prob"],  # type: ignore[arg-type]
                quality_score=data["quality_score"],  # type: ignore[arg-type]
                embedding_version=str(data["embedding_version"]),
                model_version=str(data["model_version"]),
                phase=Phase(str(phase)) if phase is not None else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid phase model output") from error
