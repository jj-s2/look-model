import pickle
import os

import numpy as np
import pytest

from risk.phase_model.training_data import RGPCDataset, RGPCSample, collate_rgpc_samples, phase_targets_for_record
from risk.phase_model.rg_training import make_loader


class _WorkerIdentityDataset:
    def __len__(self):
        return 1

    def __getitem__(self, _index):
        from torch.utils.data import get_worker_info
        worker = get_worker_info()
        assert worker is not None
        identity = f"{os.getpid()}:{worker.id}"
        return RGPCSample(identity, identity, np.zeros((1, 112), np.float32), np.array([True]), np.array([0.0], np.float32), 0.0, np.array([0]), np.array([True]), np.ones(1, np.float32))


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
    a = RGPCSample("a", "s1", np.ones((2, 112), np.float32), np.array([True, True]), np.array([0.0, 2.0], np.float32), 0.0,
                   np.array([0, 0]), np.array([True, True]), np.ones(2, np.float32))
    b = RGPCSample("b", "s2", np.ones((3, 112), np.float32), np.array([True, True, True]), np.array([0.0, 0.1, 3.0], np.float32), 1.0,
                   np.array([-1, -1, -1]), np.array([False, False, False]), np.ones(3, np.float32))
    batch = collate_rgpc_samples([a, b])
    assert batch.features.shape == (2, 3, 112)
    assert batch.valid_mask.tolist() == [[True, True, False], [True, True, True]]
    assert batch.phase_mask.tolist() == [[True, True, False], [False, False, False]]
    assert batch.subject_ids == ("s1", "s2")
    np.testing.assert_allclose(batch.dt.numpy(), [[0.0, 2.0, 0.0], [0.0, 0.1, 3.0]])
    assert batch.teacher_fall_logit.tolist() == [0.0, 0.0]
    assert batch.teacher_mask.tolist() == [False, False]


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


def test_time_jitter_dt_survives_dataset_and_collation(tmp_path):
    pose = np.zeros((64, 17, 3), dtype=np.float32)
    pose[..., 0] = np.arange(17, dtype=np.float32)
    pose[..., 2] = 1.0
    np.save(tmp_path / "pose.npy", pose)
    dataset = RGPCDataset([{"clip_id": "clip", "subject_id": "s", "media_path": "pose.npy", "coarse_event": "adl"}], tmp_path, corruption_probability=1.0, corruption_severity=1.0, corruption="time_jitter", seed=3)
    sample = dataset[0]
    batch = collate_rgpc_samples([sample])
    np.testing.assert_allclose(batch.dt.numpy(), sample.dt[None, :])
    assert np.any(sample.dt[1:] > 1.0)


def test_dataset_rejects_path_escape_with_clip_id(tmp_path):
    outside = tmp_path.parent / "outside.npy"
    np.save(outside, np.zeros((64, 17, 3), dtype=np.float32))
    dataset = RGPCDataset([{"clip_id": "escape", "subject_id": "s", "media_path": "../outside.npy", "coarse_event": "adl"}], tmp_path)
    with pytest.raises(ValueError, match="escape"):
        dataset[0]


@pytest.mark.parametrize("media_path", ("/absolute.npy", "../outside.npy"))
def test_dataset_rejects_absolute_and_parent_paths(media_path, tmp_path):
    dataset = RGPCDataset([{"clip_id": "unsafe", "subject_id": "s", "media_path": media_path, "coarse_event": "adl"}], tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        dataset[0]


def test_dataset_accepts_dataset_subdirectory(tmp_path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    pose = np.zeros((64, 17, 3), dtype=np.float32)
    pose[..., 0] = np.arange(17, dtype=np.float32)
    pose[..., 2] = 1.0
    np.save(folder / "pose.npy", pose)
    assert RGPCDataset([{"clip_id": "nested", "subject_id": "s", "dataset": "dataset", "media_path": "pose.npy", "coarse_event": "adl"}], tmp_path)[0].clip_id == "nested"


def test_dataset_rejects_symlink_escape_when_supported(tmp_path):
    outside = tmp_path.parent / "outside-link.npy"
    np.save(outside, np.zeros((64, 17, 3), dtype=np.float32))
    link = tmp_path / "link.npy"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows environment")
    dataset = RGPCDataset([{"clip_id": "link", "subject_id": "s", "media_path": "link.npy", "coarse_event": "adl"}], tmp_path)
    with pytest.raises(ValueError, match="link"):
        dataset[0]


def test_loader_runs_with_one_spawn_safe_worker(tmp_path):
    pose = np.zeros((64, 17, 3), dtype=np.float32)
    pose[..., 0] = np.arange(17, dtype=np.float32)
    pose[..., 2] = 1.0
    np.save(tmp_path / "worker.npy", pose)
    dataset = RGPCDataset([{"clip_id": "worker", "subject_id": "s", "media_path": "worker.npy", "coarse_event": "adl"}], tmp_path)
    batch = next(iter(make_loader(dataset, batch_size=1, shuffle=False, seed=4, workers=1)))
    assert batch.clip_ids == ("worker",)


def test_loader_uses_a_real_child_worker_not_the_parent_process():
    batch = next(iter(make_loader(_WorkerIdentityDataset(), batch_size=1, shuffle=False, seed=4, workers=1)))
    worker_pid, worker_id = batch.clip_ids[0].split(":")
    assert int(worker_pid) != os.getpid()
    assert worker_id == "0"
