"""Tests for leakage-safe cross-domain UR Fall fine-tuning."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from risk.urfall_crossdomain_training import train_urfall_crossdomain_experiment
from tests.risk.test_urfall_temporal_training import _dataset


def _adl(root: Path) -> Path:
    rows = []
    (root / "windows").mkdir(parents=True)
    for index in range(5):
        relative = Path("windows") / f"adl-{index}.npz"
        np.savez_compressed(root / relative, pose=np.full((3, 33, 3), -index, dtype=np.float32), label=np.int8(0))
        rows.append({"sample_id": f"adl-{index}", "sequence_id": f"urfall-adl-{index}", "label": 0, "pose_path": relative.as_posix()})
    manifest = root / "pose_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest


def test_crossdomain_folds_keep_fall_and_adl_sequences_out_of_training(tmp_path: Path) -> None:
    fall = _dataset(tmp_path / "fall")
    adl = _adl(tmp_path / "adl")
    report = train_urfall_crossdomain_experiment(fall, fall.parent, adl, adl.parent, tmp_path / "output", epochs=1, device="cpu", adl_folds=5)
    assert report["external_experiment"] is True
    assert report["promoted"] is False
    assert report["adl_holdout_sequence_count"] == 5
    assert len(report["folds"]) == 4
    assert all(fold["held_out_fall_sequence"] not in fold["train_fall_sequences"] for fold in report["folds"])
    assert all(fold["adl_holdout_sequences"] for fold in report["folds"])
    assert (tmp_path / "output" / "fall_predictions.jsonl").is_file()
    assert (tmp_path / "output" / "adl_predictions.jsonl").is_file()
