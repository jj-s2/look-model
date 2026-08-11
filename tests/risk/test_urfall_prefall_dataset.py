"""Tests for the UR Fall guarded-window dataset adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from risk.urfall_prefall_dataset import build_urfall_prefall_manifest


def _write_sequence(metadata: Path, rgb: Path, sequence_id: str, sv_values: list[float], *, missing_frame: int | None = None) -> None:
    metadata.mkdir(parents=True, exist_ok=True)
    rgb.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{index},{index * 33},{value}" for index, value in enumerate(sv_values, start=1)) + "\n"
    (metadata / f"{sequence_id}-data.csv").write_text(rows, encoding="utf-8")
    (metadata / f"{sequence_id}-acc.csv").write_text("0,0,0,0,0\n", encoding="utf-8")
    with ZipFile(rgb / f"{sequence_id}-cam0-rgb.zip", "w", ZIP_DEFLATED) as archive:
        for frame in range(1, len(sv_values) + 1):
            if frame != missing_frame:
                archive.writestr(f"{sequence_id}-cam0-rgb/{sequence_id}-cam0-rgb-{frame:03d}.png", b"png")


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_builds_guarded_pair_from_acceleration_peak_and_records_hashes(tmp_path: Path) -> None:
    metadata, rgb, output = tmp_path / "metadata", tmp_path / "rgb", tmp_path / "output"
    _write_sequence(metadata, rgb, "fall-01", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 7.0, 0.4, 0.2, 0.1])
    summary = build_urfall_prefall_manifest(metadata, rgb, output, pre_frames=2, guard_frames=1)
    records = _records(output / "manifest.jsonl")
    assert summary["positive_windows"] == 1
    assert summary["negative_windows"] == 1
    assert [record["label"] for record in records] == [1, 0]
    assert records[0]["window_start_frame"] == 3
    assert records[0]["window_end_frame"] == 5
    assert records[0]["window_end_frame"] <= records[0]["impact_frame"] - 1
    assert records[1]["window_start_frame"] == 1
    assert records[1]["window_end_frame"] == 3
    expected = hashlib.sha256((rgb / "fall-01-cam0-rgb.zip").read_bytes()).hexdigest()
    assert {record["source_sha256"] for record in records} == {expected}
    assert {record["sequence_id"] for record in records} == {"urfall-fall-01"}


def test_excludes_sequences_without_an_early_safe_window_or_complete_rgb(tmp_path: Path) -> None:
    metadata, rgb, output = tmp_path / "metadata", tmp_path / "rgb", tmp_path / "output"
    _write_sequence(metadata, rgb, "fall-02", [0.1, 9.0, 0.1, 0.1], missing_frame=None)
    _write_sequence(metadata, rgb, "fall-03", [0.1, 0.2, 0.3, 0.4, 9.0, 0.1, 0.1, 0.1], missing_frame=3)
    summary = build_urfall_prefall_manifest(metadata, rgb, output, pre_frames=2, guard_frames=1)
    assert summary["positive_windows"] == 0
    assert summary["negative_windows"] == 0
    assert summary["excluded_no_safe_window"] == 1
    assert summary["excluded_missing_rgb_frame"] == 1
    assert _records(output / "manifest.jsonl") == []
