from datetime import datetime, timedelta, timezone

import pytest

from risk.phase_model.schema import PoseObservation
from risk.phase_model.windows import DualTimescaleBuffer


def observation_at(seconds: float, tracking_id: str = "resident-1") -> PoseObservation:
    return PoseObservation(
        timestamp=datetime(2026, 8, 3, tzinfo=timezone.utc) + timedelta(seconds=seconds),
        tracking_id=tracking_id,
        keypoints=((seconds, seconds + 1.0),) * 17,
        scores=(0.95,) * 17,
        visible_mask=(True,) * 17,
        bbox=(0.0, 0.0, 100.0, 200.0),
        frame_size=(640, 480),
        stream_fresh=True,
    )


def test_sampler_uses_timestamps_not_frame_indices():
    buffer = DualTimescaleBuffer(short_frames=4, short_fps=2.0, long_frames=4, long_fps=1.0)
    seconds = [0.0, 0.52, 1.02, 1.57, 2.08, 2.56, 3.09, 3.61, 4.12]
    windows = [buffer.append(observation_at(value)) for value in seconds]
    window = next(item for item in reversed(windows) if item is not None)
    for observations, expected in ((window.short, 0.5), (window.long, 1.0)):
        gaps = [item.timestamp.timestamp() - observations[index - 1].timestamp.timestamp() for index, item in enumerate(observations) if index]
        assert gaps == pytest.approx([expected] * (len(observations) - 1), abs=0.2)


def test_out_of_order_timestamp_is_rejected():
    buffer = DualTimescaleBuffer(short_frames=2, short_fps=1, long_frames=2, long_fps=1)
    buffer.append(observation_at(2.0))
    with pytest.raises(ValueError, match="monotonic"):
        buffer.append(observation_at(1.0))


def test_low_coverage_does_not_emit_window():
    buffer = DualTimescaleBuffer(short_frames=5, short_fps=5, long_frames=5, long_fps=2, min_coverage=0.8)
    assert buffer.append(observation_at(0.0)) is None
    assert buffer.append(observation_at(2.0)) is None


def test_complete_long_window_allows_covered_short_quality_window():
    """The predictor needs an exact long window, while quality may be sparse."""
    buffer = DualTimescaleBuffer(
        short_frames=5, short_fps=5, long_frames=3, long_fps=1, min_coverage=.8
    )
    observations = [0.0, 1.0, 1.2, 1.4, 1.8, 2.0]

    windows = [buffer.append(observation_at(seconds)) for seconds in observations]
    window = windows[-1]

    assert window is not None
    assert len(window.long) == 3
    assert len(window.short) == 4


def test_long_branch_uses_adjacent_real_pose_when_target_frame_is_missed():
    """A one-interval detector gap may use a nearby real long-scale pose."""
    buffer = DualTimescaleBuffer(
        short_frames=2, short_fps=1, long_frames=3, long_fps=1, min_coverage=.5
    )

    windows = [buffer.append(observation_at(seconds)) for seconds in (0.0, .4, 2.0)]
    window = windows[-1]

    assert window is not None
    assert [item.timestamp.second for item in window.long] == [0, 0, 2]
    assert window.long[1].timestamp.microsecond == 400000
