"""GMDCSA24 v2.1 metadata adapter."""

from __future__ import annotations

import csv
from pathlib import Path

from .base import DatasetAdapter, ensure_root, make_clip


class GMDCSA24Adapter:
    name = "GMDCSA24"
    version = "v2.1"

    def scan(self, root: Path):
        root = ensure_root(root)
        metadata_path = root / "metadata.csv"
        if not metadata_path.exists():
            raise ValueError(f"GMDCSA24 metadata.csv not found: {metadata_path}")
        clips = []
        with metadata_path.open(encoding="utf-8", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                clips.append(
                    make_clip(
                        dataset=self.name,
                        version=self.version,
                        root=root,
                        media_path=_required(row, "media_path", row_number),
                        subject_id=_required(row, "subject_id", row_number),
                        camera_id=row.get("camera_id") or "unknown",
                        event_group_id=row.get("event_group_id") or f"row-{row_number}",
                        start_sec=float(row.get("start_sec") or 0.0),
                        end_sec=float(row.get("end_sec") or 1.0),
                        source_label=_required(row, "label", row_number),
                        source_file="metadata.csv",
                    )
                )
        return tuple(clips)


def _required(row: dict[str, str], field: str, row_number: int) -> str:
    value = row.get(field)
    if not value:
        raise ValueError(f"GMDCSA24 metadata row {row_number} missing {field}")
    return value
