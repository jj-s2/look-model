from datetime import datetime, timezone

import pytest

from risk.phase_model.schema import PoseObservation
from risk.phase_model.windows import DualWindow
from risk.phase_model.quality import assess_window_quality, classify_quality


@pytest.mark.parametrize(
    "score,mode,max_level",
    [(0.71, "normal", "critical"), (0.60, "degraded", "warning"), (0.49, "abstained", "warning")],
)
def test_quality_modes(score, mode, max_level):
    result = classify_quality(score)
    assert (result.mode, result.max_level) == (mode, max_level)


def test_two_seconds_without_reliable_pose_does_not_become_fall():
    assessment = assess_window_quality(DualWindow(short=(), long=()))
    assert assessment.mode == "abstained"
    assert "no_reliable_pose" in assessment.reasons


def test_quality_exposes_components():
    observation = PoseObservation(
        timestamp=datetime.now(timezone.utc), tracking_id="p",
        keypoints=((1.0, 1.0),) * 17, scores=(0.9,) * 17,
        visible_mask=(True,) * 17, bbox=(0, 0, 10, 20),
        frame_size=(640, 480), stream_fresh=True,
    )
    assessment = assess_window_quality(DualWindow(short=(observation,), long=(observation,)))
    assert set(assessment.components) == {
        "visible_ratio", "mean_keypoint_score", "torso_completeness",
        "tracking_continuity", "valid_frame_ratio", "stream_freshness",
    }
