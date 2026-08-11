"""Tests for audited UR Fall ADL negative-window construction."""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from risk.urfall_adl_dataset import build_urfall_adl_manifest


def _sequence(metadata: Path, rgb: Path, sequence: str, *, sv: float | None = 0.0) -> None:
    metadata.mkdir(parents=True, exist_ok=True)
    rgb.mkdir(parents=True, exist_ok=True)
    rows = (
        f"{frame},{frame * 33}\n" if sv is None else f"{frame},{frame * 33},{sv}\n"
        for frame in range(1, 41)
    )
    (metadata / f"{sequence}-data.csv").write_text("".join(rows), encoding="utf-8")
    (metadata / f"{sequence}-acc.csv").write_text("0,0,0,0,0\n", encoding="utf-8")
    with ZipFile(rgb / f"{sequence}-cam0-rgb.zip", "w", ZIP_DEFLATED) as archive:
        for frame in range(1, 41):
            archive.writestr(f"{sequence}-cam0-rgb/{sequence}-cam0-rgb-{frame:03d}.png", b"png")


def test_builds_two_separated_zero_impact_adl_windows(tmp_path: Path) -> None:
    metadata, rgb, output = tmp_path / "metadata", tmp_path / "rgb", tmp_path / "output"
    _sequence(metadata, rgb, "adl-01")
    summary = build_urfall_adl_manifest(metadata, rgb, output, window_frames=10)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["negative_windows"] == 2
    assert [row["label"] for row in rows] == [0, 0]
    assert rows[0]["window_end_frame"] <= rows[1]["window_start_frame"]
    assert all(row["sequence_id"] == "urfall-adl-01" for row in rows)
    assert all(len(row["source_sha256"]) == 64 for row in rows)


def test_excludes_nonzero_sync_signal(tmp_path: Path) -> None:
    metadata, rgb, output = tmp_path / "metadata", tmp_path / "rgb", tmp_path / "output"
    _sequence(metadata, rgb, "adl-02", sv=1.0)
    summary = build_urfall_adl_manifest(metadata, rgb, output, window_frames=10)
    assert summary["negative_windows"] == 0
    assert summary["excluded_nonzero_signal"] == 1


def test_accepts_official_two_column_adl_sync_format(tmp_path: Path) -> None:
    metadata, rgb, output = tmp_path / "metadata", tmp_path / "rgb", tmp_path / "output"
    _sequence(metadata, rgb, "adl-03", sv=None)
    summary = build_urfall_adl_manifest(metadata, rgb, output, window_frames=10)
    assert summary["negative_windows"] == 2
