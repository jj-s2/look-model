"""CAUCAFall v4 home-environment adapter."""

from __future__ import annotations

from pathlib import Path

from .base import ensure_root, make_clip


class CAUCAFallAdapter:
    name = "CAUCAFall"
    version = "v4"

    def scan(self, root: Path):
        root = ensure_root(root)
        clips = []
        media_files = [
            item for item in sorted(root.rglob("*"))
            if item.suffix.lower() in {".avi", ".mp4", ".mov"}
        ]
        for media in media_files:
            relative = media.relative_to(root).as_posix()
            parts = media.relative_to(root).parts
            subject = next(
                (part for part in parts if part.lower().startswith("subject")),
                "unknown",
            )
            label = media.parent.name
            camera = next(
                (part for part in parts if "cam" in part.lower()),
                "unknown",
            )
            clips.append(
                make_clip(
                    dataset=self.name,
                    version=self.version,
                    root=root,
                    media_path=relative,
                    subject_id=subject,
                    camera_id=camera,
                    event_group_id=media.stem,
                    start_sec=0.0,
                    end_sec=1.0,
                    source_label=label,
                    source_file=relative,
                )
            )
        return tuple(clips)
