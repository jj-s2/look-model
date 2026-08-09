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
    assert manifest["config_source_sha256"]
    assert list(manifest["cache_hashes"]) == sorted(manifest["cache_hashes"])
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
    assert checkpoint["best_epoch"] == metrics["best_epoch"]
    assert rg_training._sha256(output / "checkpoint.pt") == metrics["checkpoint_sha256"]
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
    captured, primary_batches = [], []
    real_loss = rg_training.compute_rgpc_loss
    real_targets = rg_training._targets

    def recording_targets(batch):
        primary_batches.append(batch)
        return real_targets(batch)

    def recording_loss(*args, **kwargs):
        loss = real_loss(*args, **kwargs)
        captured.append((args[0], args[1], kwargs.get("corrupted_output"), loss.components["consistency"].detach().cpu().item()))
        return loss

    monkeypatch.setattr(rg_training, "_targets", recording_targets)
    monkeypatch.setattr(rg_training, "compute_rgpc_loss", recording_loss)
    rg_training.train_rg_pcnet(dataset_lock=lock, split_manifest=split, data_root=tmp_path, output_dir=tmp_path / "release", release_id="r1", config_path=config, device="cpu")
    assert captured and primary_batches
    primary, targets, paired_clean, consistency = captured[0]
    corrupt_batch = primary_batches[0]
    assert paired_clean is not None
    assert torch.equal(primary.valid_mask, corrupt_batch.valid_mask)
    assert torch.equal(targets.fall_target, corrupt_batch.fall_target)
    assert torch.equal(targets.phase_target, corrupt_batch.phase_target)
    assert torch.equal(targets.phase_mask, corrupt_batch.phase_mask)
    assert torch.equal(targets.reliability_target, corrupt_batch.reliability_target)
    assert torch.equal(targets.valid_mask, corrupt_batch.valid_mask)
    assert torch.equal(targets.dt, corrupt_batch.dt)
    assert not torch.equal(targets.reliability_target, torch.ones_like(targets.reliability_target))
    assert consistency > 1e-8


def test_training_rejects_subject_overlap_and_invalid_schema(tmp_path):
    lock, split, config = _write_fixture(tmp_path)
    payload = json.loads(split.read_text(encoding="utf-8"))
    payload["partitions"]["validation"] = ["s1-adl"]
    split.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        rg_training.train_rg_pcnet(dataset_lock=lock, split_manifest=split, data_root=tmp_path, output_dir=tmp_path / "release", release_id="r1", config_path=config, device="cpu")


def test_same_seed_cpu_runs_produce_the_same_checkpoint_hash(tmp_path):
    lock, split, config = _write_fixture(tmp_path)
    first = rg_training.train_rg_pcnet(dataset_lock=lock, split_manifest=split, data_root=tmp_path, output_dir=tmp_path / "first", release_id="r1", config_path=config, device="cpu")
    second = rg_training.train_rg_pcnet(dataset_lock=lock, split_manifest=split, data_root=tmp_path, output_dir=tmp_path / "second", release_id="r1", config_path=config, device="cpu")
    assert first["checkpoint_sha256"] == second["checkpoint_sha256"]


def test_existing_output_is_rejected_before_reading_training_inputs(tmp_path):
    lock, split, config = _write_fixture(tmp_path)
    output = tmp_path / "release"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        rg_training.train_rg_pcnet(dataset_lock=lock, split_manifest=split, data_root=tmp_path, output_dir=output, release_id="r1", config_path=config, device="cpu")
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_checkpoint_restores_best_not_last_epoch(tmp_path, monkeypatch):
    lock, split, config = _write_fixture(tmp_path)
    config.write_text(config.read_text(encoding="utf-8").replace("EPOCHS=1", "EPOCHS=2").replace("PATIENCE=1", "PATIENCE=4"), encoding="utf-8")
    scores = iter((0.9, 0.1))
    last_states = []
    real_update = rg_training.EarlyStopping.update

    def fixed_score(*_args):
        return next(scores)

    def recording_update(self, score, *, epoch, model=None):
        result = real_update(self, score, epoch=epoch, model=model)
        last_states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        return result

    monkeypatch.setattr(rg_training, "_macro_f1", fixed_score)
    monkeypatch.setattr(rg_training.EarlyStopping, "update", recording_update)
    metrics = rg_training.train_rg_pcnet(dataset_lock=lock, split_manifest=split, data_root=tmp_path, output_dir=tmp_path / "release", release_id="r1", config_path=config, device="cpu")
    checkpoint = torch.load(tmp_path / "release" / "checkpoint.pt", weights_only=True)["model_state_dict"]
    assert metrics["best_epoch"] == 0
    assert all(torch.equal(checkpoint[key], last_states[0][key]) for key in checkpoint)
    assert any(not torch.equal(checkpoint[key], last_states[1][key]) for key in checkpoint)


def test_training_rejects_a_changed_pose_cache_before_publishing(tmp_path, monkeypatch):
    lock, split, config = _write_fixture(tmp_path)
    real_hashes = rg_training._cache_hashes
    calls = 0

    def changing_hashes(*args):
        nonlocal calls
        calls += 1
        result = real_hashes(*args)
        if calls > 1:
            result = dict(result)
            result[next(iter(result))] = "changed"
        return result

    monkeypatch.setattr(rg_training, "_cache_hashes", changing_hashes)
    with pytest.raises(RuntimeError, match="changed during training"):
        rg_training.train_rg_pcnet(dataset_lock=lock, split_manifest=split, data_root=tmp_path, output_dir=tmp_path / "release", release_id="r1", config_path=config, device="cpu")
    assert not (tmp_path / "release").exists()


def test_training_rejects_a_clip_with_no_valid_frames_before_optimizing(tmp_path):
    lock, split, config = _write_fixture(tmp_path)
    np.savez(tmp_path / "s1-adl.npz", long_pose=np.zeros((64, 17, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="s1-adl"):
        rg_training.train_rg_pcnet(dataset_lock=lock, split_manifest=split, data_root=tmp_path, output_dir=tmp_path / "release", release_id="r1", config_path=config, device="cpu")
