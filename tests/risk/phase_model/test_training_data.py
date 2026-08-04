import numpy as np
import pytest

from risk.phase_model.training_data import (
    AugmentationConfig,
    PhasePoseDataset,
    augment_pose,
    subject_folds,
)


def test_subject_folds_never_overlap_subjects():
    clips = [
        {"clip_id": "a", "subject_id": "subject-2"},
        {"clip_id": "b", "subject_id": "subject-3"},
        {"clip_id": "c", "subject_id": "subject-4"},
    ]
    folds = subject_folds(clips, ("subject-2", "subject-3", "subject-4"))
    assert len(folds) == 3
    for fold in folds:
        assert set(fold.train_subjects).isdisjoint(fold.validation_subjects)


def test_augmentation_is_seeded_and_preserves_shape():
    pose = np.ones((64, 17, 3), dtype=np.float32)
    config = AugmentationConfig(
        flip_probability=1.0,
        noise_std=0.0,
        keypoint_dropout=0.0,
        frame_mask_probability=0.0,
    )
    first = augment_pose(pose, np.random.default_rng(42), config)
    second = augment_pose(pose, np.random.default_rng(42), config)
    assert first.shape == (64, 17, 3)
    np.testing.assert_array_equal(first, second)


def test_dataset_loads_pose_cache_then_applies_train_augmentation(tmp_path):
    raw_pose = np.ones((64, 17, 3), dtype=np.float32)
    raw_pose[:, :, 0] = np.arange(17, dtype=np.float32)
    np.savez(tmp_path / "clip.npz", long_pose=raw_pose, short_embedding=np.zeros(512, dtype=np.float32))
    dataset = PhasePoseDataset(
        [{"clip_id": "clip-1", "subject_id": "subject-2", "media_path": "clip.npz"}],
        tmp_path,
        train=True,
        augmentation=AugmentationConfig(
            flip_probability=0.0,
            noise_std=0.0,
            keypoint_dropout=0.0,
            frame_mask_probability=1.0,
        ),
    )

    sample = dataset[0]

    assert sample.clip_id == "clip-1"
    assert sample.subject_id == "subject-2"
    assert sample.short_quality == 0.0
    assert sample.short_embedding is None
    assert sample.long_pose.shape == (64, 17, 3)
    masked = np.flatnonzero(np.all(sample.long_pose == 0.0, axis=(1, 2)))
    assert 2 <= len(masked) <= 6
    np.testing.assert_array_equal(masked, np.arange(masked[0], masked[0] + len(masked)))


def test_dataset_rejects_pose_cache_without_exactly_64_frames(tmp_path):
    np.savez(
        tmp_path / "wrong-length.npz",
        long_pose=np.ones((63, 17, 3), dtype=np.float32),
        short_embedding=np.zeros(512, dtype=np.float32),
    )
    dataset = PhasePoseDataset(
        [{"clip_id": "clip-1", "subject_id": "subject-2", "media_path": "wrong-length.npz"}],
        tmp_path,
    )

    with pytest.raises(ValueError, match="shape"):
        dataset[0]


def test_frame_masking_masks_one_contiguous_two_to_six_frame_interval():
    pose = np.ones((64, 17, 3), dtype=np.float32)
    augmented = augment_pose(
        pose,
        np.random.default_rng(3),
        AugmentationConfig(
            flip_probability=0.0,
            noise_std=0.0,
            keypoint_dropout=0.0,
            frame_mask_probability=1.0,
            time_scale_min=1.0,
            time_scale_max=1.0,
        ),
    )

    masked = np.flatnonzero(np.all(augmented == 0.0, axis=(1, 2)))
    assert 2 <= len(masked) <= 6
    np.testing.assert_array_equal(masked, np.arange(masked[0], masked[0] + len(masked)))


def test_noise_is_invariant_to_raw_pose_pixel_scale_after_normalization():
    pose = np.zeros((64, 17, 3), dtype=np.float32)
    pose[:, :, 0] = np.arange(17, dtype=np.float32)
    pose[:, :, 1] = np.arange(17, dtype=np.float32) * 0.5
    pose[:, :, 2] = 1.0
    config = AugmentationConfig(
        flip_probability=0.0,
        noise_std=0.03,
        keypoint_dropout=0.0,
        frame_mask_probability=0.0,
        time_scale_min=1.0,
        time_scale_max=1.0,
    )

    base = augment_pose(pose, np.random.default_rng(9), config)
    scaled = augment_pose(pose * np.array([10.0, 10.0, 1.0], dtype=np.float32), np.random.default_rng(9), config)

    from risk.phase_model.normalization import normalize_pose_array

    np.testing.assert_allclose(normalize_pose_array(base), normalize_pose_array(scaled), rtol=1e-5, atol=1e-5)
