import math

import pytest

from risk.gait_stability import GaitStabilityAnalyzer


def _sequence(scale=1.0):
    """Small walking sequence with all COCO joints present."""
    frames = []
    for index in range(12):
        phase = -1.0 if index % 4 < 2 else 1.0
        frame = [[0.0, 0.0] for _ in range(17)]
        frame[5] = [98.0 * scale, 60.0 * scale]
        frame[6] = [102.0 * scale, 60.0 * scale]
        frame[11] = [(98.0 + 0.5 * index) * scale, 100.0 * scale]
        frame[12] = [(102.0 + 0.5 * index) * scale, 100.0 * scale]
        frame[15] = [(92.0 + index + phase) * scale, 140.0 * scale]
        frame[16] = [(108.0 + index - phase) * scale, 140.0 * scale]
        frames.append(frame)
    return frames


def test_gait_features_are_invariant_to_uniform_pixel_scaling():
    analyzer = GaitStabilityAnalyzer(fps=4)

    original = analyzer.extract_features(_sequence(), (640, 480))
    scaled = analyzer.extract_features(_sequence(2.0), (1280, 960))

    for field in (
        "sway",
        "step_width",
        "step_variability",
        "step_frequency_stability",
        "left_right_symmetry",
        "torso_angle_change",
    ):
        assert getattr(scaled, field) == pytest.approx(getattr(original, field), rel=0.05)


def test_extract_features_handles_short_or_degenerate_sequences_without_nan():
    result = GaitStabilityAnalyzer().extract_features([_sequence()[0]], (0, 0))

    assert result.keypoint_quality == 1.0
    assert all(math.isfinite(getattr(result, field)) for field in result.__dataclass_fields__)


def test_legacy_analyze_reports_downward_motion_and_higher_risk():
    stable = _sequence()
    falling = _sequence()
    for index, frame in enumerate(falling):
        downward_shift = max(0, index - 6) * 8.0
        for point in frame:
            point[1] += downward_shift
        # Increasing shoulder offset produces a real late-window lean trend.
        frame[5][0] += max(0, index - 6) * 3.0
        frame[6][0] += max(0, index - 6) * 3.0

    analyzer = GaitStabilityAnalyzer(fps=4)
    stable_result = analyzer.analyze(stable)
    result = analyzer.analyze(falling)

    assert result["com_vertical_drop"] > 0.0
    assert result["com_vel_y"] > 0.0
    assert result["activity_burst"] > 1.0
    assert result["body_lean_angle"] > 0.0
    assert result["lean_trend"] > 0.0
    assert result["risk_score"] > stable_result["risk_score"]


def test_windowed_analysis_preserves_keypoint_scores():
    frames = _sequence()
    scores = [[0.25] * 17 for _ in frames]

    result = GaitStabilityAnalyzer(fps=4, window_sec=1).analyze_windowed(
        frames, scores, stride=4
    )

    assert result
    assert result[0]["confidence"] == pytest.approx(0.25)
    assert result[0]["keypoint_quality"] == pytest.approx(0.25)
