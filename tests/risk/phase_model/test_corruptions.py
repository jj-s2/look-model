import numpy as np
import pytest

from risk.phase_model.corruptions import corrupt_pose
from risk.phase_model.training_data import RGPCDataset


def test_corruption_is_seeded_and_severity_reduces_reliability():
    """Break caught: dropout no longer uses the supplied seed or supervision target."""
    pose = np.ones((16, 17, 3), dtype=np.float32)

    first = corrupt_pose(pose, np.random.default_rng(7), severity=0.8, corruption="joint_dropout")
    second = corrupt_pose(pose, np.random.default_rng(7), severity=0.8, corruption="joint_dropout")

    assert first.pose == pytest.approx(second.pose)
    assert first.reliability_target.shape == (16,)
    assert float(first.reliability_target.mean()) < 0.5


def test_clean_corruption_preserves_pose_and_full_reliability():
    """Break caught: the clean path changes pose values or reliability labels."""
    pose = np.ones((4, 17, 3), dtype=np.float32)

    result = corrupt_pose(pose, np.random.default_rng(3), severity=0.0, corruption="frame_drop")

    assert result.pose == pytest.approx(pose)
    assert result.reliability_target.tolist() == [1.0] * 4


def test_dataset_corrupts_before_building_features_and_records_metadata(tmp_path):
    """Break caught: corruptions are skipped, post-feature, or absent from artifacts."""
    pose = np.ones((64, 17, 3), dtype=np.float32)
    pose[..., 0] = np.arange(17, dtype=np.float32)
    pose[..., 1] = np.arange(17, dtype=np.float32) * 0.5
    np.save(tmp_path / "clip.npy", pose)

    sample = RGPCDataset(
        [{"clip_id": "clip-1", "subject_id": "s1", "media_path": "clip.npy", "coarse_event": "adl"}],
        tmp_path,
        corruption_probability=1.0,
        corruption_severity=1.0,
        corruption="frame_drop",
        seed=7,
    )[0]

    assert sample.corruption == "frame_drop"
    assert sample.corruption_severity == 1.0
    assert sample.reliability_target.tolist() == [0.0] * 64
    assert sample.valid_mask.tolist() == [False] * 64


def test_dataset_records_a_clean_view_when_probability_skips_corruption(tmp_path):
    """Break caught: clean draws are mislabeled as configured corruptions."""
    pose = np.ones((64, 17, 3), dtype=np.float32)
    pose[..., 0] = np.arange(17, dtype=np.float32)
    pose[..., 1] = np.arange(17, dtype=np.float32) * 0.5
    np.save(tmp_path / "clip.npy", pose)

    sample = RGPCDataset(
        [{"clip_id": "clip-1", "subject_id": "s1", "media_path": "clip.npy", "coarse_event": "adl"}],
        tmp_path,
        corruption_probability=0.5,
        corruption_severity=1.0,
        corruption="frame_drop",
        seed=7,
    )[0]

    assert sample.corruption == "clean"
    assert sample.corruption_severity == 0.0
