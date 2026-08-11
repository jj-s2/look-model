"""Leakage-safe construction of pre-fall tabular samples from pose caches."""
from __future__ import annotations

from dataclasses import asdict
from math import ceil, floor, isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from risk.gait_stability import GaitStabilityAnalyzer


FEATURE_COLUMNS = (
    "sway", "step_width", "step_variability", "step_frequency_stability",
    "left_right_symmetry", "torso_angle_change", "keypoint_quality",
    "activity_level", "activity_trend", "com_vertical_drop", "com_vel_y",
    "activity_burst", "com_sway", "body_lean_angle", "body_lean_var",
    "lean_trend", "gait_jitter",
)

METADATA_COLUMNS = (
    "subject_id", "clip_id", "label", "window_start_frame", "window_end_frame",
    "window_start_sec", "window_end_sec",
)


def duration_by_media_from_metadata(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Map GMDCSA24 metadata rows to their cached pose artifact names."""
    durations: dict[str, float] = {}
    for row in rows:
        try:
            subject = int(str(row["subject_id"]).strip())
            category = str(row["category"]).strip().lower()
            video = int(str(row["video_id"]).strip())
            duration = float(row["duration_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if category not in {"fall", "adl"} or not isfinite(duration) or duration <= 0:
            continue
        durations[f"subject-{subject}_{category}_{video:02d}.npz"] = duration
    return durations


def build_prefall_rows(
    clips: Iterable[Mapping[str, Any]], duration_by_media: Mapping[str, float], cache_dir: Path,
    *, horizon_sec: float = 3.0, guard_sec: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    """Return pose-feature rows without any post-onset frames in positive samples.

    `duration_by_media` is the original video duration, not the fall interval
    duration. Positive windows end at least `guard_sec` before the annotated
    fall onset; ADL samples use their full available pose cache.
    """
    if not isfinite(horizon_sec) or horizon_sec <= 0:
        raise ValueError("horizon_sec must be a positive finite number")
    if not isfinite(guard_sec) or guard_sec < 0:
        raise ValueError("guard_sec must be a non-negative finite number")
    audit: dict[str, int | float] = {
        "horizon_sec": float(horizon_sec), "guard_sec": float(guard_sec),
        "input_clips": 0, "positive_rows": 0, "negative_rows": 0,
        "excluded_missing_duration": 0, "excluded_missing_cache": 0,
        "excluded_no_window": 0, "excluded_invalid_event": 0,
    }
    rows: list[dict[str, Any]] = []
    for clip in clips:
        audit["input_clips"] = int(audit["input_clips"]) + 1
        media_path = str(clip.get("media_path", ""))
        duration = duration_by_media.get(media_path)
        if not isinstance(duration, (int, float)) or not isfinite(float(duration)) or float(duration) <= 0:
            audit["excluded_missing_duration"] = int(audit["excluded_missing_duration"]) + 1
            continue
        cache_path = cache_dir / media_path
        if not cache_path.is_file():
            audit["excluded_missing_cache"] = int(audit["excluded_missing_cache"]) + 1
            continue
        event = str(clip.get("coarse_event", ""))
        if event not in {"fall", "adl"}:
            audit["excluded_invalid_event"] = int(audit["excluded_invalid_event"]) + 1
            continue
        pose = _load_pose(cache_path)
        if pose is None:
            audit["excluded_missing_cache"] = int(audit["excluded_missing_cache"]) + 1
            continue
        if event == "fall":
            onset = clip.get("start_sec")
            if not isinstance(onset, (int, float)) or not isfinite(float(onset)):
                audit["excluded_no_window"] = int(audit["excluded_no_window"]) + 1
                continue
            window_end_sec = float(onset) - float(guard_sec)
            window_start_sec = max(0.0, window_end_sec - float(horizon_sec))
            start_frame = max(0, int(floor(window_start_sec / float(duration) * len(pose))))
            end_frame = min(len(pose), int(ceil(window_end_sec / float(duration) * len(pose))))
            label = 1
        else:
            window_start_sec, window_end_sec = 0.0, float(duration)
            start_frame, end_frame, label = 0, len(pose), 0
        window = pose[start_frame:end_frame]
        if len(window) < 3:
            audit["excluded_no_window"] = int(audit["excluded_no_window"]) + 1
            continue
        analyzer = GaitStabilityAnalyzer(fps=len(pose) / float(duration), window_sec=float(horizon_sec))
        gait = asdict(analyzer.extract_features(window, None))
        legacy = analyzer.analyze(window)
        row = {
            "subject_id": str(clip.get("subject_id", "")),
            "clip_id": str(clip.get("event_group_id", media_path)),
            "label": label,
            "window_start_frame": start_frame,
            "window_end_frame": end_frame,
            "window_start_sec": window_start_sec,
            "window_end_sec": window_end_sec,
            **{name: float(gait[name]) for name in (
                "sway", "step_width", "step_variability", "step_frequency_stability",
                "left_right_symmetry", "torso_angle_change", "keypoint_quality",
            )},
            **{name: float(legacy.get(name, 0.0)) for name in FEATURE_COLUMNS[7:]},
        }
        rows.append(row)
        kind = "positive_rows" if label else "negative_rows"
        audit[kind] = int(audit[kind]) + 1
    return rows, audit


def _load_pose(path: Path) -> np.ndarray | None:
    try:
        with np.load(path) as artifact:
            pose = np.asarray(artifact["long_pose"], dtype=np.float32)
    except (KeyError, OSError, ValueError):
        return None
    if pose.ndim != 3 or pose.shape[1:] != (17, 3) or len(pose) == 0:
        return None
    return pose
