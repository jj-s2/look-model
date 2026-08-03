import json

import pytest

from scripts.train_phase_model import train_phase_model


def test_training_requires_frozen_provenance(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset_lock"):
        train_phase_model(
            dataset_lock=tmp_path / "missing-lock.json",
            split_manifest=tmp_path / "split.json",
            output_dir=tmp_path / "release",
            release_id="r1",
            demo=True,
        )


def test_demo_training_writes_checkpoint_and_metrics(tmp_path):
    lock = tmp_path / "dataset_lock.json"
    split = tmp_path / "split_manifest.json"
    lock.write_text(json.dumps({"schema_version": "1.0", "demo": True, "clips": []}), encoding="utf-8")
    split.write_text(json.dumps({"schema_version": "1.0", "partitions": {"train": [], "validation": [], "test": []}}), encoding="utf-8")
    result = train_phase_model(dataset_lock=lock, split_manifest=split, output_dir=tmp_path / "release", release_id="r1", demo=True, epochs=1)
    assert result["release_id"] == "r1"
    assert (tmp_path / "release" / "checkpoint.pt").exists()
    assert json.loads((tmp_path / "release" / "metrics.json").read_text(encoding="utf-8"))["promoted"] is False


def test_real_training_resolves_dataset_subdirectory_and_uses_clip_mask(tmp_path):
    import numpy as np

    raw_root = tmp_path / "raw"
    dataset_root = raw_root / "GMDCSA24"
    dataset_root.mkdir(parents=True)
    np.savez(
        dataset_root / "sample.npz",
        long_pose=np.zeros((64, 17, 3), dtype=np.float32),
        short_embedding=np.zeros((512,), dtype=np.float32),
    )
    clip = {
        "clip_id": "a" * 64,
        "dataset": "GMDCSA24",
        "media_path": "sample.npz",
        "phase": None,
        "coarse_event": "fall",
        "supervision_mask": ["fall_event"],
    }
    lock = tmp_path / "dataset_lock.json"
    split = tmp_path / "split_manifest.json"
    lock.write_text(json.dumps({"schema_version": "1.0", "demo": False, "clips": [clip]}), encoding="utf-8")
    split.write_text(json.dumps({"schema_version": "1.0", "partitions": {"train": [clip["clip_id"]], "validation": [], "test": []}}), encoding="utf-8")
    result = train_phase_model(
        dataset_lock=lock,
        split_manifest=split,
        output_dir=tmp_path / "release",
        release_id="r-real",
        data_root=raw_root,
        epochs=1,
    )
    assert result["demo"] is False
    assert result["promoted"] is False
