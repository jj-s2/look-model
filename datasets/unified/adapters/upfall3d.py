"""Metadata-only adapter for the license-gated 3D UP-Fall skeleton files."""

from __future__ import annotations

import csv
from pathlib import Path
import re

from .base import ensure_root, make_clip


class UPFall3DAdapter:
    name = "UP-Fall-3D-Skeletons"
    version = "zenodo-12773013-v1"

    def scan(self, root: Path):
        root = ensure_root(root)
        clips = []
        for csv_path in sorted(root.rglob("*.csv")):
            rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
            match = re.search(r"C(\d+)S(\d+)A(\d+)T(\d+)", csv_path.stem, re.IGNORECASE)
            camera = f"C{match.group(1)}" if match else "unknown"
            subject = f"S{match.group(2)}" if match else csv_path.parent.name
            labels = {str(row.get("LABEL", "")).strip() for row in rows}
            source_label = "impact" if "1" in labels else "non-impact"
            relative = csv_path.relative_to(root).as_posix()
            clips.append(
                make_clip(
                    dataset=self.name,
                    version=self.version,
                    root=root,
                    media_path=relative,
                    subject_id=subject,
                    camera_id=camera,
                    event_group_id=csv_path.stem,
                    start_sec=0.0,
                    end_sec=max(1.0, len(rows) / 30.0),
                    source_label=source_label,
                    source_file=relative,
                    metadata_only=True,
                )
            )
        return tuple(clips)
