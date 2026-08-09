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


def test_temporal_feature_columns_follow_the_112_value_contract():
    result = build_temporal_features(_pose(), np.array([0.0, 0.1, 0.35], dtype=np.float32))

    # Base columns are [x, y, confidence, missing] per COCO joint.
    assert result.values[1, 0:4].tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0])
    assert result.values[1, 20:24].tolist() == pytest.approx([10.0, 0.0, 1.0, 0.0])
    assert result.values[1, 44:52].tolist() == pytest.approx([
        10.0, 21.0, 1.0, 0.0, 12.0, 21.0, 1.0, 0.0,
    ])

    # Velocity columns are [dx/dt, dy/dt] per COCO joint.
    assert result.values[1, 68:70].tolist() == pytest.approx([0.0, 0.0])
    assert result.values[1, 90:94].tolist() == pytest.approx([0.0, 10.0, 0.0, 10.0])

    # Global columns 102..111 are dt, root dx, root dy, root speed, torso unit
    # x/y, visible extent x/y, valid ratio, and mean confidence, respectively.
    assert result.values[1, 102] == pytest.approx(0.1)
    assert result.values[1, 103] == pytest.approx(0.0)
    assert result.values[1, 104] == pytest.approx(10.0)
    assert result.values[1, 105] == pytest.approx(10.0)
    assert result.values[1, 106] == pytest.approx(0.0)
    assert result.values[1, 107] == pytest.approx(-1.0)
    assert result.values[1, 108] == pytest.approx(12.0)
    assert result.values[1, 109] == pytest.approx(21.0)
    assert result.values[1, 110] == pytest.approx(1.0)
    assert result.values[1, 111] == pytest.approx(1.0)
