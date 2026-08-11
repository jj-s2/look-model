"""Tests for held-out UR Fall ADL false-alert evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from risk.urfall_adl_evaluation import evaluate_urfall_adl_false_alerts
from risk.urfall_temporal_training import train_urfall_external_experiment
from tests.risk.test_urfall_temporal_training import _dataset


def test_scores_adl_windows_with_all_saved_fall_checkpoints(tmp_path: Path) -> None:
    fall_manifest = _dataset(tmp_path / "fall")
    training = tmp_path / "training"
    train_urfall_external_experiment(fall_manifest, fall_manifest.parent, training, epochs=1, device="cpu")
    adl_root = tmp_path / "adl"
    (adl_root / "windows").mkdir(parents=True)
    rows = []
    for index in range(2):
        relative = Path("windows") / f"adl-{index}.npz"
        np.savez_compressed(adl_root / relative, pose=np.zeros((3, 33, 3), dtype=np.float32), label=np.int8(0))
        rows.append({"sample_id": f"adl-{index}", "sequence_id": "urfall-adl-01", "label": 0, "pose_path": relative.as_posix()})
    manifest = adl_root / "pose_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = evaluate_urfall_adl_false_alerts(manifest, adl_root, training / "checkpoints", tmp_path / "adl-eval", device="cpu")
    assert report["external_adl_evaluation"] is True
    assert report["sample_count"] == 2
    assert 0.0 <= report["false_positive_rate"] <= 1.0
    assert len(report["folds"]) == 4
