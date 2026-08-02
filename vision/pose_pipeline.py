"""Testable, dependency-injected person detection and pose inference pipeline.

The module deliberately does not import NumPy, Torch, or OpenMMLab at import
time.  Production adapters for those packages live at the script boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only; NumPy stays optional here.
    import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    bbox: Sequence[float]


@dataclass(frozen=True)
class PoseInferenceOutput:
    """Normalized pose-model output plus optional model-native payload."""

    keypoints: Sequence[Any]
    keypoint_scores: Sequence[Any]
    payload: Any = None


@dataclass(frozen=True)
class PoseFrameResult:
    frame_index: int
    timestamp: datetime
    bboxes: list[list[float]]
    keypoints: list[Any]
    keypoint_scores: list[Any]
    payload: Any = None


class PersonDetector(Protocol):
    def detect(self, frame: "np.ndarray") -> Sequence[Detection]: ...


class PoseEstimator(Protocol):
    def estimate(
        self, frame: "np.ndarray", boxes: Sequence[Sequence[float]]
    ) -> PoseInferenceOutput: ...


class PosePipeline:
    """Filter person detections then submit only those boxes to pose inference."""

    def __init__(
        self,
        detector: PersonDetector,
        pose_estimator: PoseEstimator,
        *,
        person_threshold: float = 0.3,
    ) -> None:
        self._detector = detector
        self._pose_estimator = pose_estimator
        self._person_threshold = person_threshold
        self._frame_index = 0

    def process(self, frame: "np.ndarray", timestamp: datetime) -> PoseFrameResult:
        self._frame_index += 1
        boxes = [
            list(detection.bbox)
            for detection in self._detector.detect(frame)
            if detection.label == "person" and detection.score >= self._person_threshold
        ]
        inference = self._pose_estimator.estimate(frame, boxes)
        return PoseFrameResult(
            frame_index=self._frame_index,
            timestamp=timestamp,
            bboxes=boxes,
            keypoints=list(inference.keypoints),
            keypoint_scores=list(inference.keypoint_scores),
            payload=inference.payload,
        )


__all__ = [
    "Detection",
    "PersonDetector",
    "PoseEstimator",
    "PoseFrameResult",
    "PoseInferenceOutput",
    "PosePipeline",
]
