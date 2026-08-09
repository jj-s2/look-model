import numpy as np
import pytest

from risk.phase_model.training_data import RGPCDataset, RGPCSample, collate_rgpc_samples, phase_targets_for_record


def test_unlabeled_fall_clip_has_no_primary_phase_supervision():
    """Break caught: treating an unlabeled fall clip as phase-supervised."""
    target, mask = phase_targets_for_record({"coarse_event": "fall"}, frames=4)
    assert target.tolist() == [-1, -1, -1, -1]
    assert mask.tolist() == [False, False, False, False]


def test_adl_clip_is_supervised_as_normal_only():
    """Break caught: removing normal-only supervision for unambiguous ADL clips."""
    target, mask = phase_targets_for_record({"coarse_event": "adl"}, frames=3)
    assert target.tolist() == [0, 0, 0]
    assert mask.tolist() == [True, True, True]


def test_collate_pads_time_and_preserves_valid_masks():
    """Break caught: padding overwrites temporal validity or phase-supervision masks."""
    a = RGPCSample("a", "s1", np.ones((2, 112), np.float32), np.array([True, True]), 0.0,
                   np.array([0, 0]), np.array([True, True]), np.ones(2, np.float32))
    b = RGPCSample("b", "s2", np.ones((3, 112), np.float32), np.array([True, True, True]), 1.0,
                   np.array([-1, -1, -1]), np.array([False, False, False]), np.ones(3, np.float32))
    batch = collate_rgpc_samples([a, b])
    assert batch.features.shape == (2, 3, 112)
    assert batch.valid_mask.tolist() == [[True, True, False], [True, True, True]]
    assert batch.phase_mask.tolist() == [[True, True, False], [False, False, False]]
    assert batch.subject_ids == ("s1", "s2")


def test_reviewed_human_phase_sequence_maps_known_labels_and_masks_unknowns():
    """Break caught: trusting unrecognized phase labels in an approved human sequence."""
    target, mask = phase_targets_for_record(
        {
            "coarse_phase_sequence": ["normal", "descent_or_impact", "unknown"],
            "label_source": "human",
            "review_status": "approved",
        },
        frames=3,
    )
    assert target.tolist() == [0, 1, -1]
    assert mask.tolist() == [True, True, False]


def test_phase_sequence_rejects_frame_count_mismatch():
    """Break caught: silently aligning a phase sequence to the wrong number of frames."""
    with pytest.raises(ValueError, match="frame count"):
        phase_targets_for_record({"coarse_phase_sequence": ["normal"]}, frames=2)


def test_dataset_builds_112_feature_fall_sample_without_phase_labels(tmp_path):
    """Break caught: a fall cache loses its temporal features or receives fake phase labels."""
    pose = np.zeros((64, 17, 3), dtype=np.float32)
    pose[..., 0] = np.arange(17, dtype=np.float32)
    pose[..., 1] = np.arange(17, dtype=np.float32) * 0.5
    pose[..., 2] = 1.0
    np.save(tmp_path / "fall.npy", pose)

    sample = RGPCDataset(
        [{"clip_id": "fall-1", "subject_id": "s1", "media_path": "fall.npy", "coarse_event": "fall"}],
        tmp_path,
    )[0]

    assert sample.features.shape == (64, 112)
    assert sample.fall_target == 1.0
    assert sample.phase_target.tolist() == [-1] * 64
    assert sample.phase_mask.tolist() == [False] * 64
    assert sample.valid_mask.tolist() == [True] * 64
