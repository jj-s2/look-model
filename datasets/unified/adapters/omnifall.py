"""OmniFall annotation-table adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .base import ensure_root, make_clip


class OmniFallAdapter:
    name = "OmniFall"
    version = "huggingface-card-2026-08-04"

    def scan(self, root: Path):
        root = ensure_root(root)
        annotations = next(
            (path for path in (root / "annotations.jsonl", root / "annotations.csv") if path.exists()),
            None,
        )
        if annotations is None:
            raise ValueError(f"OmniFall annotations file not found: {root}")
        rows = _read_rows(annotations)
        clips = []
        for index, row in enumerate(rows, start=1):
            source_dataset = str(row.get("dataset") or "unknown")
            media_path = str(row.get("path") or row.get("media_path") or "")
            if not media_path:
                raise ValueError(f"OmniFall row {index} missing path")
            start = float(row.get("start", row.get("start_sec", 0.0)))
            end = float(row.get("end", row.get("end_sec", 1.0)))
            subject = str(row.get("subject", row.get("subject_id", "unknown")))
            camera = str(row.get("cam", row.get("camera_id", "unknown")))
            label = str(row.get("label", row.get("phase", "unknown")))
            clips.append(
                make_clip(
                    dataset=source_dataset,
                    version=self.version,
                    root=root,
                    media_path=media_path,
                    subject_id=subject,
                    camera_id=camera,
                    event_group_id=f"{media_path}:{start:.6f}",
                    start_sec=start,
                    end_sec=end,
                    source_label=label,
                    source_file=annotations.name,
                )
            )
        return tuple(clips)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
