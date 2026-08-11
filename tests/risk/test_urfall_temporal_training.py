"""Tests for UR Fall sequence-grouped external temporal experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from risk.urfall_temporal_training import train_urfall_external_experiment


def _dataset(root: Path) -> Path:
    rows = []
    windows = root / "windows"
    windows.mkdir(parents=True)
    for sequence in range(1, 5):
        for label, kind in ((1, "preimpact"), (0, "early-safe")):
            sample_id = f"urfall-fall-{sequence:02d}-{kind}"
            relative = Path("windows") / f"{sample_id}.npz"
            pose = np.full((3, 33, 3), fill_value=sequence * (1 if label else -1), dtype=np.float32)
            np.savez_compressed(root / relative, pose=pose, label=np.int8(label))
            rows.append({"sample_id": sample_id, "sequence_id": f"urfall-fall-{sequence:02d}", "label": label, "pose_path": relative.as_posix()})
    manifest = root / "pose_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest


def test_external_training_uses_sequence_grouped_folds_and_never_promotes(tmp_path: Path) -> None:
    manifest = _dataset(tmp_path / "data")
    report = train_urfall_external_experiment(manifest, manifest.parent, tmp_path / "output", epochs=1, device="cpu")
    assert report["external_experiment"] is True
    assert report["promoted"] is False
    assert report["split_unit"] == "sequence_id"
    assert report["sequence_count"] == 4
    assert all(fold["held_out_sequence"] not in fold["train_sequences"] for fold in report["folds"])
    assert (tmp_path / "output" / "checkpoints" / "urfall-fall-01.pt").is_file()
