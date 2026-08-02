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
