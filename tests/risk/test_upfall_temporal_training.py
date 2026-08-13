import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from risk.upfall_temporal_training import (
    UpFallTemporalTCN,
    load_upfall_samples,
    normalize_upfall_pose,
    train_upfall_loso,
)


def _write_manifest(root: Path) -> Path:
    windows = root / "windows"
    windows.mkdir(parents=True)
    records = []
    for sample_id, label, subject, value in (
        ("sample-positive", 1, "upfall-s1", 1.0),
        ("sample-negative", 0, "upfall-s2", 0.0),
    ):
        relative = Path("windows") / f"{sample_id}.npz"
        pose = np.full((4, 33, 3), value, dtype=np.float32)
        pose[:, :, 0] += np.arange(4, dtype=np.float32)[:, None]
        np.savez_compressed(root / relative, pose=pose, label=np.int8(label))
        records.append({"sample_id": sample_id, "window_path": relative.as_posix(), "label": label, "subject_id": subject})
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in reversed(records)), encoding="utf-8")
    return manifest


def test_loader_preserves_33_joint_windows_and_subjects(tmp_path: Path):
    manifest = _write_manifest(tmp_path)

    poses, labels, subjects = load_upfall_samples(manifest, tmp_path)

    assert poses.shape == (2, 4, 33, 3)
    assert labels.tolist() == [0, 1]
    assert subjects == ["upfall-s2", "upfall-s1"]


def test_tcn_normalizes_pose_and_returns_one_logit_per_sample():
    torch = pytest.importorskip("torch")
    pose = torch.arange(2 * 4 * 33 * 3, dtype=torch.float32).reshape(2, 4, 33, 3)

    normalized = normalize_upfall_pose(pose)
    logits = UpFallTemporalTCN(joints=33, hidden_dim=8, dropout=0.0)(pose)

    assert normalized.shape == (2, 4, 33, 9)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_loso_training_never_promotes_the_experimental_upfall_model(tmp_path: Path):
    root = tmp_path / "samples"
    windows = root / "windows"
    windows.mkdir(parents=True)
    rows = []
    for subject in ("upfall-s1", "upfall-s2"):
        for label in (0, 1):
            sample_id = f"{subject}-{label}"
            relative = Path("windows") / f"{sample_id}.npz"
            pose = np.full((4, 33, 3), float(label), dtype=np.float32)
            pose[:, :, 0] += np.arange(4, dtype=np.float32)[:, None]
            np.savez_compressed(root / relative, pose=pose, label=np.int8(label))
            rows.append({"sample_id": sample_id, "window_path": relative.as_posix(), "label": label, "subject_id": subject})
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = train_upfall_loso(manifest, root, tmp_path / "output", epochs=1, hidden_dim=8, device="cpu")

    assert report["promoted"] is False
    assert report["experimental_only"] is True
    assert report["not_for_clinical_performance"] is True
    assert report["window_unit"] == "frames"
    assert len(report["folds"]) == 2
    assert (tmp_path / "output" / "metrics.json").is_file()
    assert (tmp_path / "output" / "model_card.json").is_file()


def test_cli_writes_an_experimental_model_card(tmp_path: Path):
    root = tmp_path / "samples"
    windows = root / "windows"
    windows.mkdir(parents=True)
    rows = []
    for subject in ("upfall-s1", "upfall-s2"):
        for label in (0, 1):
            sample_id = f"{subject}-{label}"
            relative = Path("windows") / f"{sample_id}.npz"
            np.savez_compressed(root / relative, pose=np.ones((4, 33, 3), dtype=np.float32) * label, label=np.int8(label))
            rows.append({"sample_id": sample_id, "window_path": relative.as_posix(), "label": label, "subject_id": subject})
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "output"
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_upfall_prefall_tcn.py"

    subprocess.run(
        [sys.executable, str(script), "--manifest", str(manifest), "--data-root", str(root),
         "--output-dir", str(output), "--epochs", "1", "--hidden-dim", "8", "--device", "cpu"],
        check=True, capture_output=True, text=True,
    )

    card = json.loads((output / "model_card.json").read_text(encoding="utf-8"))
    assert card["promoted"] is False
    assert card["window_unit"] == "frames"
