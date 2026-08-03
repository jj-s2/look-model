from datetime import datetime, timezone

import pytest

from risk.phase_model.schema import PoseObservation
from risk.phase_model.windows import DualWindow
from risk.phase_model.normalization import normalize_pose_window


def _observation(scale=1.0, offset=(0.0, 0.0)):
    points = [(0.0, 0.0)] * 17
    points[0] = (10.0, 20.0)
    points[1] = (12.0, 20.0)
    points[5] = (10.0, 0.0)
    points[6] = (12.0, 0.0)
    points = tuple((scale * x + offset[0], scale * y + offset[1]) for x, y in points)
    return PoseObservation(
        timestamp=datetime.now(timezone.utc), tracking_id="p",
        keypoints=points, scores=(0.9,) * 17, visible_mask=(True,) * 17,
        bbox=(0, 0, 20, 30), frame_size=(640, 480), stream_fresh=True,
    )


def test_normalization_is_scale_and_translation_invariant():
    first = normalize_pose_window(DualWindow(short=(_observation(),), long=()))
    second = normalize_pose_window(DualWindow(short=(_observation(2.0, (100.0, 50.0)),), long=()))
    first_flat = [value for frame in first.coordinates for point in frame for value in point]
    second_flat = [value for frame in second.coordinates for point in frame for value in point]
    assert first_flat == pytest.approx(second_flat)
    assert first.visible_mask == second.visible_mask


def test_invisible_coordinates_are_masked_not_counted_as_real_zero():
    item = _observation()
    item = PoseObservation(
        timestamp=item.timestamp, tracking_id=item.tracking_id, keypoints=item.keypoints,
        scores=item.scores, visible_mask=(True,) * 16 + (False,), bbox=item.bbox,
        frame_size=item.frame_size, stream_fresh=item.stream_fresh,
    )
    result = normalize_pose_window(DualWindow(short=(item,), long=()))
    assert result.visible_mask[0][-1] is False
    assert result.frame_valid[0] is True
