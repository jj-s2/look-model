"""Deterministic pose corruptions with per-frame reliability supervision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorruptedPose:
    pose: np.ndarray
    reliability_target: np.ndarray
    corruption: str
    severity: float
    timestamp_scale: np.ndarray


def corrupt_pose(
    pose: np.ndarray,
    rng: np.random.Generator,
    *,
    severity: float,
    corruption: str,
) -> CorruptedPose:
    """Return one seeded pose corruption and its auditable reliability target."""
    array = np.asarray(pose, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (17, 3) or array.shape[0] == 0:
        raise ValueError("pose must have shape (time, 17, 3) and contain frames")
    severity = float(severity)
    if not np.isfinite(severity) or not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be finite and in [0, 1]")
    allowed = {"joint_dropout", "frame_drop", "freeze", "confidence_collapse", "time_jitter"}
    if corruption not in allowed:
        raise ValueError(f"unsupported corruption: {corruption}")

    changed = array.copy()
    frames = len(changed)
    affected = np.zeros(frames, dtype=np.float32)
    timestamp_scale = np.ones(frames, dtype=np.float32)
    if severity == 0.0:
        return CorruptedPose(changed, np.ones(frames, np.float32), corruption, severity, timestamp_scale)
    if corruption == "joint_dropout":
        mask = rng.random((frames, 17)) < severity
        changed[mask] = 0.0
        affected = mask.mean(axis=1).astype(np.float32)
    elif corruption == "frame_drop":
        mask = rng.random(frames) < severity
        changed[mask] = 0.0
        affected = mask.astype(np.float32)
    elif corruption == "freeze":
        length = min(frames, max(1, round(frames * severity)))
        start = int(rng.integers(0, frames - length + 1))
        changed[start : start + length] = changed[max(0, start - 1)]
        affected[start : start + length] = 1.0
    elif corruption == "confidence_collapse":
        changed[..., 2] *= 1.0 - severity
        affected.fill(severity)
    else:
        timestamp_scale = np.clip(rng.normal(1.0, 0.1 * severity, frames), 0.5, 1.5).astype(np.float32)
        affected = np.clip(np.abs(timestamp_scale - 1.0) / 0.1, 0.0, 1.0)
    reliability = np.clip(1.0 - severity * affected, 0.0, 1.0).astype(np.float32)
    return CorruptedPose(changed, reliability, corruption, severity, timestamp_scale)


__all__ = ["CorruptedPose", "corrupt_pose"]
