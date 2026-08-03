"""Person-centred, scale-normalized pose features with explicit masks."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Sequence

from .schema import PoseObservation
from .windows import DualWindow


def normalize_pose_array(pose: object):
    """Normalize a COCO-17 ``(time, joints, xy+confidence)`` pose cache.

    The cache extractor emits pixel coordinates.  Person-centering on the two
    hips and scaling by shoulder-to-hip distance removes camera translation and
    subject size while retaining confidence as the third channel.
    """
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError("NumPy is required to normalize pose arrays") from error
    array = np.asarray(pose, dtype=np.float32).copy()
    if array.ndim != 3 or array.shape[1:] != (17, 3):
        raise ValueError("pose must have shape (time, 17, 3)")
    for frame in array:
        visible = frame[:, 2] >= 0.25
        hip_visible = bool(visible[11] and visible[12])
        shoulder_visible = bool(visible[5] and visible[6])
        if hip_visible:
            origin = (frame[11, :2] + frame[12, :2]) / 2.0
        elif np.any(visible):
            origin = frame[visible, :2].mean(axis=0)
        else:
            frame[:, :2] = 0.0
            continue
        if hip_visible and shoulder_visible:
            shoulder = (frame[5, :2] + frame[6, :2]) / 2.0
            scale = float(np.linalg.norm(shoulder - origin))
        else:
            spread = frame[visible, :2] - origin
            scale = float(np.sqrt(np.mean(spread * spread))) if spread.size else 0.0
        if not np.isfinite(scale) or scale <= 1e-6:
            frame[:, :2] = 0.0
            continue
        frame[:, :2] = (frame[:, :2] - origin) / scale
        frame[~visible, :2] = 0.0
    return array


@dataclass(frozen=True)
class NormalizedPoseWindow:
    coordinates: tuple[tuple[tuple[float, float], ...], ...]
    visible_mask: tuple[tuple[bool, ...], ...]
    frame_valid: tuple[bool, ...]
    observations: tuple[PoseObservation, ...]


def _frames(window: DualWindow | Sequence[PoseObservation]) -> tuple[PoseObservation, ...]:
    if isinstance(window, DualWindow):
        return window.short or window.long
    return tuple(window)


def normalize_pose_window(window: DualWindow | Sequence[PoseObservation]) -> NormalizedPoseWindow:
    observations = _frames(window)
    coordinates: list[tuple[tuple[float, float], ...]] = []
    masks: list[tuple[bool, ...]] = []
    valid: list[bool] = []
    for observation in observations:
        points = observation.keypoints
        mask = observation.visible_mask
        hip_indices = (0, 1)
        shoulder_indices = (5, 6)
        hip_visible = all(index < len(points) and mask[index] for index in hip_indices)
        shoulder_visible = all(index < len(points) and mask[index] for index in shoulder_indices)
        if hip_visible:
            origin = (
                sum(points[index][0] for index in hip_indices) / len(hip_indices),
                sum(points[index][1] for index in hip_indices) / len(hip_indices),
            )
        elif observation.bbox is not None:
            x, y, width, height = observation.bbox
            origin = (x + width / 2.0, y + height / 2.0)
        else:
            origin = (0.0, 0.0)
        if hip_visible and shoulder_visible:
            shoulder = (
                sum(points[index][0] for index in shoulder_indices) / len(shoulder_indices),
                sum(points[index][1] for index in shoulder_indices) / len(shoulder_indices),
            )
            scale = hypot(shoulder[0] - origin[0], shoulder[1] - origin[1])
        elif observation.bbox is not None:
            _, _, width, height = observation.bbox
            scale = hypot(width, height)
        else:
            scale = 0.0
        frame_is_valid = scale > 1e-6 and (hip_visible or observation.bbox is not None)
        if not frame_is_valid:
            coordinates.append(tuple((0.0, 0.0) for _ in points))
            masks.append(tuple(False for _ in points))
            valid.append(False)
            continue
        normalized = []
        for point, is_visible in zip(points, mask):
            normalized.append(
                ((point[0] - origin[0]) / scale, (point[1] - origin[1]) / scale)
                if is_visible else (0.0, 0.0)
            )
        coordinates.append(tuple(normalized))
        masks.append(tuple(mask))
        valid.append(True)
    return NormalizedPoseWindow(tuple(coordinates), tuple(masks), tuple(valid), observations)
