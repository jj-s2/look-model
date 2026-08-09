import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from risk.phase_model import rg_training
from risk.phase_model.training_data import RGPCDataset


def _pose(value=1.0):
    pose = np.zeros((64, 17, 3), dtype=np.float32)
    pose[..., 0] = np.arange(17, dtype=np.float32)[None, :] + value
    pose[..., 1] = np.arange(64, dtype=np.float32)[:, None] * 0.01
    pose[..., 2] = 1.0
    return pose


def _write_fixture(tmp_path):
    clips = []
    for subject, event, number in (("s1", "adl", 1), ("s1", "fall", 2), ("s2", "adl", 3), ("s2", "fall", 4), ("s3", "adl", 5), ("s3", "fall", 6)):
        clip_id = f"{subject}-{event}"
        path = f"{clip_id}.npz"
        np.savez(tmp_path / path, long_pose=_pose(number))
        clips.append({"clip_id": clip_id, "subject_id": subject, "media_path": path, "coarse_event": event})
    lock = tmp_path / "dataset_lock.json"
    split = tmp_path / "split_manifest.json"
    lock.write_text(json.dumps({"schema_version": "1.0", "demo": False, "clips": clips}), encoding="utf-8")
    split.write_text(json.dumps({"schema_version": "1.0", "release_id": "r1", "partitions": {"train": ["s1-adl", "s1-fall", "s2-adl", "s2-fall"], "validation": ["s3-adl", "s3-fall"]}}), encoding="utf-8")
    config = tmp_path / "tiny_config.py"
    config.write_text("SEED=7\nINPUT_DIM=112\nHIDDEN_DIM=8\nDROPOUT=0.0\nBATCH_SIZE=2\nEPOCHS=1\nPATIENCE=1\nLEARNING_RATE=0.001\nWEIGHT_DECAY=0.0001\nGRAD_CLIP_NORM=1.0\nNUM_WORKERS=0\nCORRUPTION_PROBABILITY=1.0\n", encoding="utf-8")
    return lock, split, config


def test_training_writes_a_non_promoted_best_checkpoint_and_provenance(tmp_path):
    lock, split, config = _write_fixture(tmp_path)
    output = tmp_path / "release"
    metrics = rg_training.train_rg_pcnet(dataset_lock=lock, split_manifest=split, data_root=tmp_path, output_dir=output, release_id="r1", config_path=config, device="cpu")
    assert metrics["batch_size"] == 2
    assert metrics["best_epoch"] == 0
    assert metrics["promoted"] is False
    assert metrics["validation_subjects"] == ["s3"]
    assert len(metrics["checkpoint_sha256"]) == 64
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["git_commit"]
    assert manifest["input_manifest_hashes"]
    assert (output / "dataset_lock.json").read_bytes() == lock.read_bytes()
    assert (output / "split_manifest.json").read_bytes() == split.read_bytes()


def test_clean_and_corrupted_views_are_clip_aligned_and_epoch_streams_change(tmp_path):
    lock, split, _ = _write_fixture(tmp_path)
    clips = json.loads(lock.read_text(encoding="utf-8"))["clips"][:2]
    clean = RGPCDataset(clips, tmp_path, seed=7)
    corrupt = RGPCDataset(clips, tmp_path, seed=7, corruption_probability=1.0, corruption_severity=0.5)
    clean.set_epoch(0)
    corrupt.set_epoch(0)
    assert [clean[i].clip_id for i in range(2)] == [corrupt[i].clip_id for i in range(2)]
    epoch_zero = corrupt[0].features.copy()
    corrupt.set_epoch(0)
    np.testing.assert_array_equal(epoch_zero, corrupt[0].features)
    corrupt.set_epoch(1)
    assert not np.array_equal(epoch_zero, corrupt[0].features)


def test_training_passes_a_corrupted_output_to_the_loss(tmp_path, monkeypatch):
    lock, split, config = _write_fixture(tmp_path)
    captured = []
    real_loss = rg_training.compute_rgpc_loss

    def recording_loss(*args, **kwargs):
        captured.append(kwargs.get("corrupted_output"))
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(rg_training, "compute_rgpc_loss", recording_loss)
    rg_training.train_rg_pcnet(dataset_lock=lock, split_manifest=split, data_root=tmp_path, output_dir=tmp_path / "release", release_id="r1", config_path=config, device="cpu")
    assert captured and all(output is not None for output in captured)


def test_training_rejects_a_clip_with_no_valid_frames_before_optimizing(tmp_path):
    lock, split, config = _write_fixture(tmp_path)
    np.savez(tmp_path / "s1-adl.npz", long_pose=np.zeros((64, 17, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="s1-adl"):
        rg_training.train_rg_pcnet(dataset_lock=lock, split_manifest=split, data_root=tmp_path, output_dir=tmp_path / "release", release_id="r1", config_path=config, device="cpu")
