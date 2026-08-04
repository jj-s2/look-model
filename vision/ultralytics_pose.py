"""Lazy Ultralytics YOLO11-pose adapter for the live monitoring path."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .pose_pipeline import PoseFrameResult


class UltralyticsPosePipeline:
    """Convert one BGR frame into the existing pose-frame contract."""

    def __init__(
        self,
        model: Any | None = None,
        *,
        model_path: str = "yolo11n-pose.pt",
        device: str = "auto",
        confidence: float = 0.25,
        imgsz: int = 512,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if imgsz <= 0:
            raise ValueError("imgsz must be positive")
        self.device = _resolve_device(device)
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self._model = model or self._load_model(model_path)
        self._frame_index = 0

    @staticmethod
    def _load_model(model_path: str) -> Any:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as error:  # pragma: no cover - dependency guard
            raise RuntimeError("Ultralytics is required for the live pose pipeline") from error
        return YOLO(model_path)

    def process(self, frame: Any, timestamp: datetime) -> PoseFrameResult:
        self._frame_index += 1
        try:
            results = self._model.predict(
                source=frame,
                device=self.device,
                verbose=False,
                imgsz=self.imgsz,
                conf=self.confidence,
            )
            result = next(iter(results), None)
            return self._convert(result, timestamp)
        except Exception:
            return PoseFrameResult(
                frame_index=self._frame_index,
                timestamp=timestamp,
                bboxes=[],
                keypoints=[],
                keypoint_scores=[],
                payload={"error": "pose_inference_failed"},
            )

    def _convert(self, result: Any, timestamp: datetime) -> PoseFrameResult:
        if result is None or getattr(result, "boxes", None) is None:
            return self._empty(timestamp)
        boxes = result.boxes
        raw_xyxy = _to_numpy(getattr(boxes, "xyxy", None))
        raw_scores = _to_numpy(getattr(boxes, "conf", None))
        raw_labels = _to_numpy(getattr(boxes, "cls", None))
        if raw_xyxy is None or raw_xyxy.ndim != 2:
            return self._empty(timestamp)
        keypoints = getattr(result, "keypoints", None)
        raw_points = _to_numpy(getattr(keypoints, "xy", None)) if keypoints is not None else None
        raw_point_scores = _to_numpy(getattr(keypoints, "conf", None)) if keypoints is not None else None
        selected_boxes: list[list[float]] = []
        selected_points: list[list[list[float]]] = []
        selected_scores: list[list[float]] = []
        for index, box in enumerate(raw_xyxy):
            label = float(raw_labels[index]) if raw_labels is not None and index < len(raw_labels) else 0.0
            score = float(raw_scores[index]) if raw_scores is not None and index < len(raw_scores) else 0.0
            if label != 0.0 or score < self.confidence:
                continue
            selected_boxes.append([float(value) for value in box[:4]])
            if raw_points is not None and index < len(raw_points):
                points = [[float(value) for value in point[:2]] for point in raw_points[index][:17]]
            else:
                points = []
            if raw_point_scores is not None and index < len(raw_point_scores):
                point_scores = [float(value) for value in raw_point_scores[index][:17]]
            else:
                point_scores = []
            selected_points.append(points)
            selected_scores.append(point_scores)
        if not selected_boxes:
            return self._empty(timestamp)
        return PoseFrameResult(
            frame_index=self._frame_index,
            timestamp=timestamp,
            bboxes=selected_boxes,
            keypoints=selected_points,
            keypoint_scores=selected_scores,
            payload=result,
        )

    def _empty(self, timestamp: datetime) -> PoseFrameResult:
        return PoseFrameResult(
            frame_index=self._frame_index,
            timestamp=timestamp,
            bboxes=[],
            keypoints=[],
            keypoint_scores=[],
            payload={},
        )


class SinglePersonTracker:
    """Deterministic index tracker for the one-resident demo path."""

    def track(self, result: PoseFrameResult) -> list[dict[str, str]]:
        return [{"tracking_id": f"person-{index}"} for index, _ in enumerate(result.bboxes)]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def _to_numpy(value: Any):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        import numpy as np
        return np.asarray(value)
    except (ImportError, TypeError, ValueError):
        return None


__all__ = ["SinglePersonTracker", "UltralyticsPosePipeline"]
