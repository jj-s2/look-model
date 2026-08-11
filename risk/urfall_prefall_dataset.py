"""Leakage-safe UR Fall RGB pre-impact window manifest builder."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from math import isfinite
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

_SEQUENCE = re.compile(r"^(fall-\d{2})-data$", re.IGNORECASE)

def build_urfall_prefall_manifest(source_dir: Path, rgb_dir: Path, output_dir: Path, *, pre_frames: int = 10, guard_frames: int = 5) -> dict[str, int | str]:
    """Build guarded positive and earlier-safe windows from UR Fall sync CSV files."""
    _validate_parameters(pre_frames, guard_frames)
    source_dir, rgb_dir, output_dir = Path(source_dir), Path(rgb_dir), Path(output_dir)
    if not source_dir.is_dir():
        raise ValueError(f"source_dir is not a directory: {source_dir}")
    if not rgb_dir.is_dir():
        raise ValueError(f"rgb_dir is not a directory: {rgb_dir}")
    summary: dict[str, int | str] = {
        "dataset": "UR Fall Detection Dataset", "window_unit": "frames", "frame_index_base": 0,
        "pre_frames": pre_frames, "guard_frames": guard_frames, "input_files": 0,
        "positive_windows": 0, "negative_windows": 0, "excluded_invalid_sync": 0,
        "excluded_missing_acc": 0, "excluded_missing_rgb_archive": 0,
        "excluded_missing_rgb_frame": 0, "excluded_no_safe_window": 0,
    }
    records: list[dict[str, Any]] = []
    for data_path in sorted(source_dir.glob("fall-*-data.csv")):
        summary["input_files"] = int(summary["input_files"]) + 1
        match = _SEQUENCE.match(data_path.stem)
        if match is None:
            summary["excluded_invalid_sync"] = int(summary["excluded_invalid_sync"]) + 1
            continue
        sequence = match.group(1).lower()
        acc_path = source_dir / f"{sequence}-acc.csv"
        archive_path = rgb_dir / f"{sequence}-cam0-rgb.zip"
        if not acc_path.is_file():
            summary["excluded_missing_acc"] = int(summary["excluded_missing_acc"]) + 1
            continue
        loaded = _load_sync(data_path)
        if loaded is None:
            summary["excluded_invalid_sync"] = int(summary["excluded_invalid_sync"]) + 1
            continue
        frame_numbers, sv_values = loaded
        if not archive_path.is_file():
            summary["excluded_missing_rgb_archive"] = int(summary["excluded_missing_rgb_archive"]) + 1
            continue
        if not _has_all_frames(archive_path, sequence, frame_numbers):
            summary["excluded_missing_rgb_frame"] = int(summary["excluded_missing_rgb_frame"]) + 1
            continue
        max_sv = max(sv_values)
        impact_row = next(index for index, value in enumerate(sv_values) if value == max_sv)
        impact_frame = frame_numbers[impact_row]
        positive_end = impact_row - guard_frames
        positive_start = positive_end - pre_frames
        negative_end = positive_start
        negative_start = negative_end - pre_frames
        if negative_start < 0:
            summary["excluded_no_safe_window"] = int(summary["excluded_no_safe_window"]) + 1
            continue
        source_hash = _sha256(archive_path)
        for window_type, label, start, end in (("preimpact", 1, positive_start, positive_end), ("early_safe", 0, negative_start, negative_end)):
            records.append({
                "sample_id": f"urfall-{sequence}-{window_type}", "sequence_id": f"urfall-{sequence}",
                "source_data_csv": data_path.name, "source_acc_csv": acc_path.name, "source_zip": archive_path.name,
                "source_sha256": source_hash, "camera_id": "cam0", "label": label, "window_type": window_type,
                "impact_frame": impact_frame, "impact_sv_total": float(max_sv),
                "window_start_frame": start, "window_end_frame": end, "pre_frames": pre_frames,
                "guard_frames": guard_frames, "window_unit": "frames", "frame_index_base": 0,
            })
        summary["positive_windows"] = int(summary["positive_windows"]) + 1
        summary["negative_windows"] = int(summary["negative_windows"]) + 1
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary

def _validate_parameters(pre_frames: int, guard_frames: int) -> None:
    for name, value in (("pre_frames", pre_frames), ("guard_frames", guard_frames)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if pre_frames == 0:
        raise ValueError("pre_frames must be greater than zero")

def _load_sync(path: Path) -> tuple[list[int], list[float]] | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
    except (OSError, UnicodeError, csv.Error):
        return None
    if not rows:
        return None
    frames: list[int] = []
    values: list[float] = []
    for expected_index, row in enumerate(rows, start=1):
        if len(row) != 3:
            return None
        try:
            frame, timestamp, sv_total = int(row[0]), float(row[1]), float(row[2])
        except ValueError:
            return None
        if frame != expected_index or not isfinite(timestamp) or not isfinite(sv_total):
            return None
        frames.append(frame)
        values.append(sv_total)
    return frames, values

def _has_all_frames(archive_path: Path, sequence: str, frame_numbers: list[int]) -> bool:
    expected = {f"{sequence}-cam0-rgb/{sequence}-cam0-rgb-{frame:03d}.png" for frame in frame_numbers}
    try:
        with ZipFile(archive_path) as archive:
            return expected.issubset(set(archive.namelist()))
    except (BadZipFile, OSError):
        return False

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

