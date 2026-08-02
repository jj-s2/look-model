"""Scale-invariant gait features extracted from COCO-17 pose sequences.

The implementation deliberately uses only the Python standard library so that
the risk layer can run on an edge device before NumPy/Torch are installed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan2, degrees, hypot, isfinite, sqrt
from statistics import median
from typing import Any, Iterable, Sequence


L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16
_EPSILON = 1e-6


@dataclass(frozen=True)
class GaitFeatures:
    """Dimensionless gait measurements for a single time window."""

    sway: float = 0.0
    step_width: float = 0.0
    step_variability: float = 0.0
    step_frequency_stability: float = 0.0
    left_right_symmetry: float = 0.0
    torso_angle_change: float = 0.0
    keypoint_quality: float = 0.0

    @property
    def gait_jitter(self) -> float:
        """Compatibility name: higher values mean less regular stepping."""
        return self.step_variability


class GaitStabilityAnalyzer:
    """Extract robust, scale-independent features from a COCO-17 sequence."""

    def __init__(self, fps: float = 29.7, window_sec: float = 3.0):
        self.fps = float(fps) if fps and isfinite(float(fps)) else 29.7
        self.window = max(int(self.fps * float(window_sec)), 1)

    def extract_features(self, sequence: Any, frame_size: Sequence[float] | None) -> GaitFeatures:
        """Return gait features normalized by torso length (or frame diagonal).

        ``sequence`` accepts nested lists or array-like objects in ``(T,17,2)``
        or ``(N,T,17,2)`` form.  A third keypoint component is interpreted as
        keypoint confidence; callers can instead pass a mapping containing
        ``keypoints`` and ``keypoint_scores``.
        """
        raw, scores = self._split_sequence(sequence)
        frames = self._frames(raw)
        if not frames:
            return GaitFeatures()
        points, embedded_quality = self._point_frames(frames)
        quality = self._quality(scores, embedded_quality, len(points))
        scale = self._normalizer(points, frame_size)

        centers = [self._midpoint(frame[L_HIP], frame[R_HIP]) for frame in points]
        widths = [abs(frame[L_ANKLE][0] - frame[R_ANKLE][0]) / scale
                  for frame in points if frame[L_ANKLE] and frame[R_ANKLE]]
        sway = self._std([center[0] / scale for center in centers if center])
        step_width = self._median_or_zero(widths)
        step_variability = self._std(widths)

        angles = []
        for frame in points:
            shoulder = self._midpoint(frame[L_SHOULDER], frame[R_SHOULDER])
            hip = self._midpoint(frame[L_HIP], frame[R_HIP])
            if shoulder and hip:
                angles.append(degrees(atan2(abs(shoulder[0] - hip[0]),
                                           max(abs(shoulder[1] - hip[1]), _EPSILON))))
        angle_change = self._mean_abs_difference(angles)

        left_path = self._path_length([frame[L_ANKLE] for frame in points]) / scale
        right_path = self._path_length([frame[R_ANKLE] for frame in points]) / scale
        symmetry = 0.0
        if left_path + right_path > _EPSILON:
            symmetry = max(0.0, min(1.0, 1.0 - abs(left_path - right_path) /
                                    (left_path + right_path)))

        cadence_stability = self._cadence_stability(points)
        return GaitFeatures(
            sway=self._finite(sway),
            step_width=self._finite(step_width),
            step_variability=self._finite(step_variability),
            step_frequency_stability=self._finite(cadence_stability),
            left_right_symmetry=self._finite(symmetry),
            torso_angle_change=self._finite(angle_change),
            keypoint_quality=self._finite(quality),
        )

    def analyze(self, keypoints: Any, keypoint_scores: Any = None) -> dict[str, float]:
        """Legacy dictionary API retained for existing pre-fall callers."""
        features = self.extract_features(
            {"keypoints": keypoints, "keypoint_scores": keypoint_scores}, None)
        # These names are kept to avoid breaking the Task 6/pre-fall pipeline.
        risk_score = min(1.0, max(0.0, (
            features.sway + features.step_variability + features.torso_angle_change / 30.0
            + (1.0 - features.left_right_symmetry) + (1.0 - features.keypoint_quality)
        ) / 5.0))
        return {
            "activity_level": features.step_width,
            "activity_trend": 0.0,
            "com_height": 0.0,
            "com_vertical_drop": 0.0,
            "com_vel_y": 0.0,
            "activity_burst": 0.0,
            "com_sway": features.sway,
            "body_lean_angle": 0.0,
            "body_lean_var": features.torso_angle_change,
            "lean_trend": 0.0,
            "gait_jitter": features.step_variability,
            "confidence": features.keypoint_quality,
            "risk_score": risk_score,
            **asdict(features),
        }

    def analyze_windowed(self, keypoints: Any, keypoint_scores: Any = None, stride: int | None = None) -> list[dict[str, float]]:
        frames = self._frames(keypoints)
        step = stride or max(self.window // 2, 1)
        results = []
        for start in range(0, max(len(frames) - self.window + 1, 1), step):
            segment = frames[start:start + self.window]
            result = self.analyze(segment, None)
            result.update(frame_start=start, frame_end=start + len(segment))
            results.append(result)
        return results

    @staticmethod
    def _to_list(value: Any) -> Any:
        return value.tolist() if hasattr(value, "tolist") else value

    def _split_sequence(self, sequence: Any) -> tuple[Any, Any]:
        if isinstance(sequence, dict):
            return sequence.get("keypoints", sequence.get("points", [])), sequence.get("keypoint_scores")
        return sequence, None

    def _frames(self, sequence: Any) -> list[Any]:
        value = self._to_list(sequence)
        if not isinstance(value, (list, tuple)) or not value:
            return []
        # (N,T,17,2): choose first tracked person, matching the former API.
        if self._is_point(value[0]):
            return [value]
        if isinstance(value[0], (list, tuple)) and value[0] and self._is_point(value[0][0]):
            return list(value)
        if isinstance(value[0], (list, tuple)) and value[0] and isinstance(value[0][0], (list, tuple)):
            return list(value[0])
        return []

    @staticmethod
    def _is_point(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) >= 2 and all(
            isinstance(component, (int, float)) for component in value[:2])

    def _point_frames(self, frames: Iterable[Any]) -> tuple[list[list[tuple[float, float] | None]], list[float]]:
        result, embedded_quality = [], []
        for frame in frames:
            clean = []
            for index in range(17):
                point = frame[index] if isinstance(frame, (list, tuple)) and index < len(frame) else None
                if not self._is_point(point) or not isfinite(float(point[0])) or not isfinite(float(point[1])):
                    clean.append(None)
                    continue
                clean.append((float(point[0]), float(point[1])))
                if len(point) >= 3 and isinstance(point[2], (int, float)) and isfinite(float(point[2])):
                    embedded_quality.append(max(0.0, min(1.0, float(point[2]))))
            result.append(clean)
        return result, embedded_quality

    def _quality(self, scores: Any, embedded: list[float], frame_count: int) -> float:
        candidate = self._to_list(scores)
        values = []
        def collect(value: Any) -> None:
            if isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            elif isinstance(value, (int, float)) and isfinite(float(value)):
                values.append(max(0.0, min(1.0, float(value))))
        if candidate is not None:
            collect(candidate)
        if values:
            return sum(values) / len(values)
        if embedded:
            return sum(embedded) / len(embedded)
        return 1.0 if frame_count else 0.0

    def _normalizer(self, points: list[list[tuple[float, float] | None]], frame_size: Sequence[float] | None) -> float:
        torso_lengths = []
        for frame in points:
            shoulder = self._midpoint(frame[L_SHOULDER], frame[R_SHOULDER])
            hip = self._midpoint(frame[L_HIP], frame[R_HIP])
            if shoulder and hip:
                torso_lengths.append(hypot(shoulder[0] - hip[0], shoulder[1] - hip[1]))
        valid = [length for length in torso_lengths if length > _EPSILON]
        if valid:
            return median(valid)
        size = self._to_list(frame_size) or ()
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            diagonal = hypot(float(size[0]), float(size[1]))
            if isfinite(diagonal) and diagonal > _EPSILON:
                return diagonal
        return 1.0

    def _cadence_stability(self, points: list[list[tuple[float, float] | None]]) -> float:
        signal = []
        for frame in points:
            ankle, hip = frame[L_ANKLE], self._midpoint(frame[L_HIP], frame[R_HIP])
            if ankle and hip:
                signal.append(ankle[0] - hip[0])
        extrema = [index for index in range(1, len(signal) - 1)
                   if (signal[index] >= signal[index - 1] and signal[index] > signal[index + 1])
                   or (signal[index] <= signal[index - 1] and signal[index] < signal[index + 1])]
        intervals = [extrema[index] - extrema[index - 1] for index in range(1, len(extrema))]
        if not intervals:
            return 1.0 if len(signal) >= 3 else 0.0
        mean_interval = sum(intervals) / len(intervals)
        return 1.0 / (1.0 + self._std(intervals) / max(mean_interval, _EPSILON))

    @staticmethod
    def _midpoint(left: tuple[float, float] | None, right: tuple[float, float] | None) -> tuple[float, float] | None:
        if left is None or right is None:
            return None
        return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)

    @staticmethod
    def _path_length(points: list[tuple[float, float] | None]) -> float:
        total = 0.0
        for previous, current in zip(points, points[1:]):
            if previous and current:
                total += hypot(current[0] - previous[0], current[1] - previous[1])
        return total

    @staticmethod
    def _std(values: list[float]) -> float:
        if not values:
            return 0.0
        average = sum(values) / len(values)
        return sqrt(sum((value - average) ** 2 for value in values) / len(values))

    @staticmethod
    def _mean_abs_difference(values: list[float]) -> float:
        return sum(abs(current - previous) for previous, current in zip(values, values[1:])) / max(len(values) - 1, 1)

    @staticmethod
    def _median_or_zero(values: list[float]) -> float:
        return float(median(values)) if values else 0.0

    @staticmethod
    def _finite(value: float) -> float:
        return float(value) if isfinite(value) else 0.0
