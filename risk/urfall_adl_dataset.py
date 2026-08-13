"""Audited UR Fall ADL negative-window manifest builder."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from math import isfinite
from pathlib import Path
from zipfile import BadZipFile, ZipFile

_SEQUENCE = re.compile(r"^(adl-\d{2})-data$", re.IGNORECASE)

def build_urfall_adl_manifest(source_dir: Path, rgb_dir: Path, output_dir: Path, *, window_frames: int = 10) -> dict[str, int]:
    """Build two separated fixed-length zero-impact ADL windows per valid sequence."""
    if type(window_frames) is not int or window_frames <= 0:
        raise ValueError("window_frames must be a positive integer")
    source_dir, rgb_dir, output_dir = Path(source_dir), Path(rgb_dir), Path(output_dir)
    if not source_dir.is_dir() or not rgb_dir.is_dir():
        raise ValueError("source_dir and rgb_dir must be directories")
    summary = {
        "input_files": 0, "negative_windows": 0, "excluded_invalid_sync": 0,
        "excluded_nonzero_signal": 0, "excluded_missing_rgb_archive": 0,
        "excluded_missing_rgb_frame": 0, "excluded_short_sequence": 0,
    }
    rows = []
    for data_path in sorted(source_dir.glob("adl-*-data.csv")):
        summary["input_files"] += 1
        match = _SEQUENCE.match(data_path.stem)
        if match is None:
            summary["excluded_invalid_sync"] += 1
            continue
        sequence = match.group(1).lower()
        values = _load_data(data_path)
        if values is None:
            summary["excluded_invalid_sync"] += 1
            continue
        frames, sv = values
        if sv is not None and any(value != 0.0 for value in sv):
            summary["excluded_nonzero_signal"] += 1
            continue
        archive_path = rgb_dir / f"{sequence}-cam0-rgb.zip"
        if not archive_path.is_file():
            summary["excluded_missing_rgb_archive"] += 1
            continue
        if not _has_frames(archive_path, sequence, frames):
            summary["excluded_missing_rgb_frame"] += 1
            continue
        first_start = len(frames) // 4 - window_frames // 2
        second_start = (3 * len(frames)) // 4 - window_frames // 2
        starts = (max(0, first_start), second_start)
        if starts[1] + window_frames > len(frames) or starts[0] + window_frames > starts[1]:
            summary["excluded_short_sequence"] += 1
            continue
        digest = _sha256(archive_path)
        for ordinal, start in enumerate(starts, start=1):
            rows.append({
                "sample_id": f"urfall-{sequence}-adl-{ordinal}",
                "sequence_id": f"urfall-{sequence}",
                "source_data_csv": data_path.name,
                "source_zip": archive_path.name,
                "source_sha256": digest,
                "camera_id": "cam0",
                "label": 0,
                "window_type": "adl_normal",
                "window_start_frame": start,
                "window_end_frame": start + window_frames,
                "window_frames": window_frames,
                "window_unit": "frames",
                "frame_index_base": 0,
            })
            summary["negative_windows"] += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary

def _load_data(path: Path) -> tuple[list[int], list[float] | None] | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
    except (OSError, UnicodeError, csv.Error):
        return None
    if not rows:
        return None
    frames: list[int] = []
    values: list[float] | None = [] if len(rows[0]) == 3 else None
    for expected, row in enumerate(rows, start=1):
        if len(row) not in {2, 3} or (values is None) != (len(row) == 2):
            return None
        try:
            frame, timestamp = int(row[0]), float(row[1])
            value = float(row[2]) if len(row) == 3 else None
        except ValueError:
            return None
        if frame != expected or not isfinite(timestamp) or (value is not None and not isfinite(value)):
            return None
        frames.append(frame)
        if values is not None and value is not None:
            values.append(value)
    return frames, values

def _has_frames(path: Path, sequence: str, frames: list[int]) -> bool:
    expected = {f"{sequence}-cam0-rgb/{sequence}-cam0-rgb-{frame:03d}.png" for frame in frames}
    try:
        with ZipFile(path) as archive:
            return expected.issubset(set(archive.namelist()))
    except (OSError, BadZipFile):
        return False

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
