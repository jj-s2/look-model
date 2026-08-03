"""Quality scoring and abstention policy for pose evidence."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Mapping

from .schema import PoseObservation
from .windows import DualWindow


@dataclass(frozen=True)
class QualityAssessment:
    score: float
    mode: str
    max_level: str
    components: Mapping[str, float]
    reasons: tuple[str, ...] = ()


def classify_quality(score: float) -> QualityAssessment:
    score = max(0.0, min(1.0, float(score)))
    if score >= 0.70:
        mode, max_level = "normal", "critical"
    elif score >= 0.55:
        mode, max_level = "degraded", "warning"
    else:
        mode, max_level = "abstained", "warning"
    return QualityAssessment(score, mode, max_level, {})


def _observations(window: DualWindow) -> tuple[PoseObservation, ...]:
    return window.short or window.long


def assess_window_quality(window: DualWindow) -> QualityAssessment:
    observations = _observations(window)
    if not observations:
        components = {name: 0.0 for name in (
            "visible_ratio", "mean_keypoint_score", "torso_completeness",
            "tracking_continuity", "valid_frame_ratio", "stream_freshness",
        )}
        return QualityAssessment(0.0, "abstained", "warning", components, ("no_reliable_pose",))
    visible_ratio = mean(sum(item.visible_mask) / len(item.visible_mask) for item in observations)
    mean_score = mean(mean(item.scores) for item in observations)
    torso = mean(float(all(item.visible_mask[index] for index in (0, 1, 5, 6) if index < len(item.visible_mask))) for item in observations)
    continuity = sum(item.tracking_id == observations[0].tracking_id for item in observations) / len(observations)
    valid_ratio = sum(bool(item.bbox) and any(item.visible_mask) for item in observations) / len(observations)
    freshness = sum(item.stream_fresh for item in observations) / len(observations)
    components = {
        "visible_ratio": min(1.0, max(0.0, visible_ratio)),
        "mean_keypoint_score": min(1.0, max(0.0, mean_score)),
        "torso_completeness": torso,
        "tracking_continuity": continuity,
        "valid_frame_ratio": valid_ratio,
        "stream_freshness": freshness,
    }
    score = (
        0.25 * components["visible_ratio"]
        + 0.20 * components["mean_keypoint_score"]
        + 0.20 * components["torso_completeness"]
        + 0.15 * components["tracking_continuity"]
        + 0.10 * components["valid_frame_ratio"]
        + 0.10 * components["stream_freshness"]
    )
    classified = classify_quality(score)
    reasons = () if classified.mode != "abstained" else ("no_reliable_pose",)
    return QualityAssessment(score, classified.mode, classified.max_level, components, reasons)
