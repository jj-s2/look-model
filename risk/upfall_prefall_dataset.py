"""Leakage-safe frame-window adapter for the UP-Fall 3D skeleton CSV files."""
from __future__ import annotations

import csv
import json
import re
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np


_IDENTITY = re.compile(
    r"(?:C(?P<camera>\d+))?S(?P<subject>\d+)A(?P<activity>\d+)T(?P<trial>\d+)",
    flags=re.IGNORECASE,
)
_JOINT_COLUMNS = tuple(f"Joint{joint}_{axis}" for joint in range(1, 34) for axis in ("X", "Y", "Z"))


def parse_upfall_identity(path: Path) -> dict[str, str | int]:
    """Return subject and source identifiers encoded by a UP-Fall CSV filename."""
    match = _IDENTITY.search(path.stem)
    if match is None:
        raise ValueError(f"cannot parse UP-Fall identity from {path.name}")
    groups = match.groupdict()
    camera = groups["camera"]
    return {
        "subject_id": f"upfall-s{int(groups['subject'])}",
        "camera_id": f"camera-{int(camera)}" if camera is not None else "camera-unknown",
        "activity_id": int(groups["activity"]),
        "trial_id": int(groups["trial"]),
    }


def build_upfall_prefall_dataset(
    source_dir: Path, output_dir: Path, *, pre_frames: int = 10, guard_frames: int = 5,
) -> dict[str, int | str]:
    """Write guarded pre-impact and earlier-safe 33-joint windows from UP-Fall CSVs.

    Window units are source frames because this skeleton release does not contain a
    reliable per-file frame-rate field. A source file is accepted only when it can
    produce both a pre-impact sample and an earlier safe sample of equal length.
    """
    _validate_window_parameters(pre_frames, guard_frames)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    summary: dict[str, int | str] = {
        "window_unit": "frames",
        "pre_frames": pre_frames,
        "guard_frames": guard_frames,
        "input_files": 0,
        "positive_windows": 0,
        "negative_windows": 0,
        "excluded_invalid_identity": 0,
        "excluded_invalid_coordinates": 0,
        "excluded_invalid_labels": 0,
        "excluded_no_impact": 0,
        "excluded_no_safe_window": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = output_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source_path in sorted(source_dir.rglob("*.csv")):
        summary["input_files"] = int(summary["input_files"]) + 1
        try:
            identity = parse_upfall_identity(source_path)
        except ValueError:
            summary["excluded_invalid_identity"] = int(summary["excluded_invalid_identity"]) + 1
            continue
        loaded = _load_upfall_csv(source_path)
        if loaded is None:
            summary["excluded_invalid_coordinates"] = int(summary["excluded_invalid_coordinates"]) + 1
            continue
        pose, labels = loaded
        if labels is None:
            summary["excluded_invalid_labels"] = int(summary["excluded_invalid_labels"]) + 1
            continue
        onset_frame = next((index for index, label in enumerate(labels) if label == 1), None)
        if onset_frame is None:
            summary["excluded_no_impact"] = int(summary["excluded_no_impact"]) + 1
            continue
        positive_end = onset_frame - guard_frames
        positive_start = positive_end - pre_frames
        negative_end = positive_start
        negative_start = negative_end - pre_frames
        if negative_start < 0:
            summary["excluded_no_safe_window"] = int(summary["excluded_no_safe_window"]) + 1
            continue
        records.extend(_write_pair(
            source_path, output_dir, pose, identity, onset_frame, pre_frames, guard_frames,
            positive_start, positive_end, negative_start, negative_end,
        ))
        summary["positive_windows"] = int(summary["positive_windows"]) + 1
        summary["negative_windows"] = int(summary["negative_windows"]) + 1
    manifest = output_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    return summary


def _validate_window_parameters(pre_frames: int, guard_frames: int) -> None:
    for name, value in (("pre_frames", pre_frames), ("guard_frames", guard_frames)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if pre_frames == 0:
        raise ValueError("pre_frames must be greater than zero")


def _load_upfall_csv(path: Path) -> tuple[np.ndarray, list[int] | None] | None:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or any(column not in reader.fieldnames for column in (*_JOINT_COLUMNS, "LABEL")):
                return None
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error):
        return None
    if not rows:
        return None
    try:
        coordinates = np.asarray(
            [[float(row[column]) for column in _JOINT_COLUMNS] for row in rows], dtype=np.float32,
        ).reshape(len(rows), 33, 3)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(coordinates).all():
        return None
    labels: list[int] = []
    for row in rows:
        try:
            value = float(row["LABEL"])
        except (KeyError, TypeError, ValueError):
            return coordinates, None
        if not isfinite(value) or value not in {0.0, 1.0}:
            return coordinates, None
        labels.append(int(value))
    return coordinates, labels


def _write_pair(
    source_path: Path, output_dir: Path, pose: np.ndarray, identity: dict[str, str | int], onset_frame: int,
    pre_frames: int, guard_frames: int, positive_start: int, positive_end: int,
    negative_start: int, negative_end: int,
) -> list[dict[str, Any]]:
    records = []
    for kind, label, start, end in (
        ("preimpact", 1, positive_start, positive_end),
        ("early_safe", 0, negative_start, negative_end),
    ):
        sample_id = f"{source_path.stem}-{kind}"
        relative_path = Path("windows") / f"{sample_id}.npz"
        np.savez_compressed(output_dir / relative_path, pose=pose[start:end].astype(np.float32), label=np.int8(label))
        records.append({
            "sample_id": sample_id,
            "source_file": source_path.name,
            **identity,
            "label": label,
            "onset_frame": onset_frame,
            "window_start_frame": start,
            "window_end_frame": end,
            "pre_frames": pre_frames,
            "guard_frames": guard_frames,
            "window_path": relative_path.as_posix(),
        })
    return records
