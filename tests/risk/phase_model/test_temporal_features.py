import numpy as np
import pytest

from risk.phase_model.temporal_features import FEATURE_DIM, build_temporal_features


def _pose(frames: int = 3) -> np.ndarray:
    pose = np.zeros((frames, 17, 3), dtype=np.float32)
    pose[:, :, 2] = 1.0
    pose[:, 11, :2] = (10.0, 20.0)
    pose[:, 12, :2] = (12.0, 20.0)
    pose[:, 5, :2] = (10.0, 0.0)
    pose[:, 6, :2] = (12.0, 0.0)
    pose[1:, 11:13, 1] += np.arange(1, frames, dtype=np.float32)[:, None]
    return pose


def test_temporal_features_have_stable_shape_masks_and_real_dt():
    result = build_temporal_features(_pose(), np.array([0.0, 0.1, 0.35], dtype=np.float32))
    assert FEATURE_DIM == 112
    assert result.values.shape == (3, 112)
    assert result.valid_mask.tolist() == [True, True, True]
    assert result.missing_mask.shape == (3, 17)
    assert result.dt.tolist() == pytest.approx([0.0, 0.1, 0.25])


def test_missing_joint_is_explicit_and_does_not_create_velocity_spike():
    pose = _pose()
    pose[1, 0] = 0.0
    result = build_temporal_features(pose)
    assert result.missing_mask[1, 0]
    velocity_offset = 17 * 4
    assert result.values[1, velocity_offset : velocity_offset + 2].tolist() == [0.0, 0.0]
