"""Tests for extracting auditable UR Fall pose windows."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from risk.urfall_pose_extraction import extract_urfall_pose_windows


class _Detector:
    def __init__(self, values: list[np.ndarray | None]) -> None:
        self._values = iter(values)

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        del image
        return next(self._values)


def _manifest(path: Path, source_sha256: str) -> Path:
    row = {
        "sample_id": "urfall-fall-01-preimpact",
        "sequence_id": "urfall-fall-01",
        "source_zip": "fall-01-cam0-rgb.zip",
        "source_sha256": source_sha256,
        "label": 1,
        "window_start_frame": 0,
        "window_end_frame": 3,
        "impact_frame": 8,
        "guard_frames": 2,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def _archive(directory: Path) -> Path:
    directory.mkdir(parents=True)
    path = directory / "fall-01-cam0-rgb.zip"
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for frame in range(1, 4):
            archive.writestr(f"fall-01-cam0-rgb/fall-01-cam0-rgb-{frame:03d}.png", b"image")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extracts_fixed_shape_pose_and_preserves_window_lineage(tmp_path: Path) -> None:
    rgb, output = tmp_path / "rgb", tmp_path / "output"
    archive = _archive(rgb)
    manifest = _manifest(tmp_path / "manifest.jsonl", _sha256(archive))
    positive = json.loads(manifest.read_text(encoding="utf-8"))
    negative = {**positive, "sample_id": "urfall-fall-01-early-safe", "label": 0}
    manifest.write_text(json.dumps(positive) + "\n" + json.dumps(negative) + "\n", encoding="utf-8")
    pose = np.arange(99, dtype=np.float32).reshape(33, 3)
    summary = extract_urfall_pose_windows(
        manifest, rgb, output, detector=_Detector([pose, pose + 1, pose + 2, pose, pose + 1, pose + 2]),
        image_loader=lambda _: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    with np.load(output / "windows" / "urfall-fall-01-preimpact.npz") as artifact:
        assert artifact["pose"].shape == (3, 33, 3)
        assert artifact["label"].item() == 1
    row = next(
        json.loads(line)
        for line in (output / "pose_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line and json.loads(line)["sample_id"] == "urfall-fall-01-preimpact"
    )
    assert summary["extracted_windows"] == 2
    assert row["pose_path"] == "windows/urfall-fall-01-preimpact.npz"
    assert row["source_sha256"] == _sha256(archive)


def test_excludes_window_when_any_frame_has_no_pose(tmp_path: Path) -> None:
    rgb, output = tmp_path / "rgb", tmp_path / "output"
    manifest = _manifest(tmp_path / "manifest.jsonl", _sha256(_archive(rgb)))
    pose = np.zeros((33, 3), dtype=np.float32)
    summary = extract_urfall_pose_windows(
        manifest, rgb, output, detector=_Detector([pose, None, pose]),
        image_loader=lambda _: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    assert summary["extracted_windows"] == 0
    assert summary["excluded_missing_pose"] == 1
    assert not (output / "windows" / "urfall-fall-01-preimpact.npz").exists()


def test_keeps_only_complete_positive_negative_sequence_pairs(tmp_path: Path) -> None:
    rgb, output = tmp_path / "rgb", tmp_path / "output"
    manifest = _manifest(tmp_path / "manifest.jsonl", _sha256(_archive(rgb)))
    positive = json.loads(manifest.read_text(encoding="utf-8"))
    negative = {**positive, "sample_id": "urfall-fall-01-early-safe", "label": 0}
    manifest.write_text(json.dumps(positive) + "\n" + json.dumps(negative) + "\n", encoding="utf-8")
    pose = np.zeros((33, 3), dtype=np.float32)
    summary = extract_urfall_pose_windows(
        manifest, rgb, output, detector=_Detector([pose, pose, pose, pose, None]),
        image_loader=lambda _: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    assert summary["extracted_windows"] == 0
    assert summary["excluded_incomplete_sequence"] == 2
    assert json.loads((output / "pose_manifest.jsonl").read_text(encoding="utf-8") or "[]") == []
