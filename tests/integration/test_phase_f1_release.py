import json
from pathlib import Path

import numpy as np
import pytest

from risk.phase_model.release import (
    ReleaseBundle,
    build_release_bundle,
    load_release_bundle,
    release_inference_config,
    write_release_bundle,
)
from risk.phase_model.torch_predictor import TorchPhasePredictor


def _minimal_checkpoint(release_id: str) -> dict:
    import torch

    from risk.phase_model.model import PhaseAwareFusionModel

    model = PhaseAwareFusionModel(short_dim=512, joints=17, hidden_dim=128)
    return {
        "release_id": release_id,
        "model": model.state_dict(),
        "short_dim": 512,
        "joints": 17,
        "hidden_dim": 128,
        "checkpoint_sha256": "a" * 64,
    }


def _write_checkpoint(checkpoint_path: Path, release_id: str) -> None:
    import torch

    torch.save(_minimal_checkpoint(release_id), checkpoint_path)


def _fake_clip(tmp_path: Path, clip_id: str, dataset: str) -> dict:
    np.savez(
        tmp_path / dataset / f"{clip_id}.npz",
        long_pose=np.zeros((64, 17, 3), dtype=np.float32),
        short_embedding=np.zeros((512,), dtype=np.float32),
    )
    return {
        "clip_id": clip_id,
        "dataset": dataset,
        "media_path": f"{clip_id}.npz",
        "subject_id": "subject-1",
        "coarse_event": "fall",
        "supervision_mask": ["fall_event"],
    }


def test_release_bundle_includes_calibration(tmp_path):
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    dataset_dir = tmp_path / "GMDCSA24"
    dataset_dir.mkdir()

    checkpoint_path = release_dir / "checkpoint.pt"
    _write_checkpoint(checkpoint_path, "r-f1")

    clip = _fake_clip(tmp_path, "clip-a", "GMDCSA24")
    lock = {"schema_version": "1.0", "clips": [clip]}
    split = {"schema_version": "1.0", "partitions": {"train": [], "validation": ["clip-a"], "test": []}}
    metrics = {"release_id": "r-f1", "promoted": True}
    calibration = {"temperature": 2.0, "threshold": 0.75}

    (release_dir / "dataset_lock.json").write_text(json.dumps(lock), encoding="utf-8")
    (release_dir / "split_manifest.json").write_text(json.dumps(split), encoding="utf-8")
    (release_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (release_dir / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")

    bundle = load_release_bundle(release_dir)
    assert bundle.calibration is not None
    assert bundle.calibration["temperature"] == 2.0
    assert bundle.calibration["threshold"] == 0.75

    config = release_inference_config(bundle)
    assert config["temperature"] == 2.0
    assert config["threshold"] == 0.75


def test_predictor_uses_release_calibration(tmp_path):
    pytest.importorskip("torch")

    release_dir = tmp_path / "release"
    release_dir.mkdir()
    dataset_dir = tmp_path / "GMDCSA24"
    dataset_dir.mkdir()

    checkpoint_path = release_dir / "checkpoint.pt"
    _write_checkpoint(checkpoint_path, "r-f1")

    clip = _fake_clip(tmp_path, "clip-a", "GMDCSA24")
    lock = {"schema_version": "1.0", "clips": [clip]}
    split = {"schema_version": "1.0", "partitions": {"train": [], "validation": ["clip-a"], "test": []}}
    metrics = {"release_id": "r-f1", "promoted": True}
    calibration = {"temperature": 2.0, "threshold": 0.75}

    for name, data in (
        ("dataset_lock.json", lock),
        ("split_manifest.json", split),
        ("metrics.json", metrics),
        ("calibration.json", calibration),
    ):
        (release_dir / name).write_text(json.dumps(data), encoding="utf-8")

    predictor = TorchPhasePredictor.from_release(release_dir, device="cpu")
    assert predictor.temperature == 2.0
    assert predictor.threshold == 0.75
    assert predictor.release_id == "r-f1"


def test_write_release_bundle_round_trip(tmp_path):
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    checkpoint_path = release_dir / "checkpoint.pt"
    _write_checkpoint(checkpoint_path, "r-f1")

    clip = {"clip_id": "clip-a", "dataset": "GMDCSA24", "media_path": "clip-a.npz", "subject_id": "subject-1", "coarse_event": "fall"}
    lock = {"schema_version": "1.0", "clips": [clip]}
    split = {"schema_version": "1.0", "partitions": {"train": [], "validation": [], "test": []}}
    metrics = {"release_id": "r-f1"}
    calibration = {"temperature": 1.5, "threshold": 0.6}

    bundle = build_release_bundle(
        "r-f1",
        checkpoint_path,
        release_dir,
        inputs={
            "dataset_lock": lock,
            "split_manifest": split,
            "metrics": metrics,
            "calibration": calibration,
        },
    )
    output_dir = tmp_path / "out"
    write_release_bundle(bundle, output_dir)

    loaded = load_release_bundle(output_dir)
    assert loaded.calibration is not None
    assert loaded.calibration["temperature"] == 1.5
    assert loaded.calibration["threshold"] == 0.6
