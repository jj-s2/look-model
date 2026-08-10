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


@pytest.mark.parametrize("missing_joint", range(17))
def test_each_joint_missing_flag_uses_its_exact_interleaved_base_offset(missing_joint):
    """Break caught: any two per-joint missing columns are swapped in the base ABI."""
    pose = _pose(frames=1)
    pose[0, missing_joint] = 0.0

    result = build_temporal_features(pose)

    for joint in range(17):
        expected = 1.0 if joint == missing_joint else 0.0
        assert result.values[0, 4 * joint + 3] == expected


def test_temporal_feature_columns_follow_the_full_112_value_contract():
    pose = np.array([
        [
            (0.0, 0.0, 0.90), (0.0, 0.0, 0.90), (0.0, 0.0, 0.90),
            (0.0, 0.0, 0.90), (0.0, 0.0, 0.90), (0.0, 0.0, 0.90),
            (0.0, 0.0, 0.90), (0.0, 0.0, 0.90), (0.0, 0.0, 0.90),
            (0.0, 0.0, 0.90), (0.0, 0.0, 0.90), (0.0, 0.0, 0.90),
            (0.0, 0.0, 0.90), (0.0, 0.0, 0.90), (0.0, 0.0, 0.90),
            (0.0, 0.0, 0.90), (0.0, 0.0, 0.90),
        ],
        [
            (1.0, 2.0, 0.30), (2.0, 4.0, 0.31), (3.0, 6.0, 0.32),
            (4.0, 8.0, 0.33), (5.0, 10.0, 0.34), (6.0, 12.0, 0.35),
            (7.0, 14.0, 0.36), (8.0, 16.0, 0.37), (9.0, 18.0, 0.38),
            (10.0, 20.0, 0.39), (11.0, 22.0, 0.40), (12.0, 24.0, 0.41),
            (13.0, 26.0, 0.42), (14.0, 28.0, 0.43), (15.0, 30.0, 0.44),
            (16.0, 32.0, 0.45), (17.0, 34.0, 0.46),
        ],
    ], dtype=np.float32)

    result = build_temporal_features(pose, np.array([0.0, 1.0], dtype=np.float32))
    expected = [
        1.0, 2.0, 0.30, 0.0, 2.0, 4.0, 0.31, 0.0,
        3.0, 6.0, 0.32, 0.0, 4.0, 8.0, 0.33, 0.0,
        5.0, 10.0, 0.34, 0.0, 6.0, 12.0, 0.35, 0.0,
        7.0, 14.0, 0.36, 0.0, 8.0, 16.0, 0.37, 0.0,
        9.0, 18.0, 0.38, 0.0, 10.0, 20.0, 0.39, 0.0,
        11.0, 22.0, 0.40, 0.0, 12.0, 24.0, 0.41, 0.0,
        13.0, 26.0, 0.42, 0.0, 14.0, 28.0, 0.43, 0.0,
        15.0, 30.0, 0.44, 0.0, 16.0, 32.0, 0.45, 0.0,
        17.0, 34.0, 0.46, 0.0,
        1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0,
        5.0, 10.0, 6.0, 12.0, 7.0, 14.0, 8.0, 16.0,
        9.0, 18.0, 10.0, 20.0, 11.0, 22.0, 12.0, 24.0,
        13.0, 26.0, 14.0, 28.0, 15.0, 30.0, 16.0, 32.0,
        17.0, 34.0,
        1.0, 12.5, 25.0, 27.9508497, -0.4472136, -0.8944272,
        16.0, 32.0, 1.0, 0.38,
    ]
    assert result.values[1].tolist() == pytest.approx(expected)
