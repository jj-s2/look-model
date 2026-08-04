import numpy as np

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
    assert sample.long_pose.shape == (64, 17, 3)
    np.testing.assert_array_equal(sample.long_pose, np.zeros((64, 17, 3), dtype=np.float32))
