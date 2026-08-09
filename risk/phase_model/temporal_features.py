"""Temporal COCO-17 pose features with explicit missing-data semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


JOINTS = 17
JOINT_BASE_DIM = JOINTS * 4       # x, y, confidence, missing
JOINT_VELOCITY_DIM = JOINTS * 2  # dx/dt, dy/dt
GLOBAL_DIM = 10
FEATURE_DIM = JOINT_BASE_DIM + JOINT_VELOCITY_DIM + GLOBAL_DIM


@dataclass(frozen=True)
class TemporalPoseFeatures:
    values: np.ndarray
    valid_mask: np.ndarray
    missing_mask: np.ndarray
    dt: np.ndarray


def build_temporal_features(pose: np.ndarray, timestamps: np.ndarray | None = None) -> TemporalPoseFeatures:
    """Build the fixed-width temporal feature representation for a pose window."""
    array = np.asarray(pose, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (JOINTS, 3):
        raise ValueError("pose must have shape (time, 17, 3)")
    frames = array.shape[0]
    if frames == 0:
        raise ValueError("pose must contain at least one frame")
    if timestamps is None:
        times = np.arange(frames, dtype=np.float32)
    else:
        times = np.asarray(timestamps, dtype=np.float32)
        if times.shape != (frames,) or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("timestamps must be finite, strictly increasing, and match time")
    dt = np.zeros(frames, dtype=np.float32)
    dt[1:] = np.diff(times)
    visible = array[..., 2] >= 0.25
    missing = ~visible
    base = np.concatenate((array[..., :2], array[..., 2:3], missing[..., None].astype(np.float32)), axis=-1)
    velocity = np.zeros((frames, JOINTS, 2), dtype=np.float32)
    valid_pair = visible[1:] & visible[:-1]
    raw_velocity = (array[1:, :, :2] - array[:-1, :, :2]) / dt[1:, None, None]
    velocity[1:] = np.where(valid_pair[..., None], raw_velocity, 0.0)
    hips = array[:, (11, 12), :2].mean(axis=1)
    shoulders = array[:, (5, 6), :2].mean(axis=1)
    root_delta = np.zeros((frames, 2), dtype=np.float32)
    root_delta[1:] = (hips[1:] - hips[:-1]) / dt[1:, None]
    torso = shoulders - hips
    torso_norm = np.linalg.norm(torso, axis=1).clip(min=1e-6)
    mins = np.where(visible[..., None], array[..., :2], np.inf).min(axis=1)
    maxs = np.where(visible[..., None], array[..., :2], -np.inf).max(axis=1)
    extent = np.where(np.isfinite(maxs - mins), maxs - mins, 0.0)
    valid_ratio = visible.mean(axis=1).astype(np.float32)
    mean_conf = np.where(visible, array[..., 2], 0.0).sum(axis=1) / visible.sum(axis=1).clip(min=1)
    global_features = np.column_stack((
        dt, root_delta[:, 0], root_delta[:, 1], np.linalg.norm(root_delta, axis=1),
        torso[:, 0] / torso_norm, torso[:, 1] / torso_norm,
        extent[:, 0], extent[:, 1], valid_ratio, mean_conf,
    )).astype(np.float32)
    values = np.concatenate((base.reshape(frames, -1), velocity.reshape(frames, -1), global_features), axis=1)
    return TemporalPoseFeatures(values, visible.any(axis=1), missing, dt)
