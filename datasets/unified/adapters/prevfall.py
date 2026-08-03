"""Pre-VFall image-folder adapter."""

from __future__ import annotations

from pathlib import Path

from .base import ensure_root, make_clip


class PreVFallAdapter:
    name = "Pre-VFall"
    version = "figshare-26488216-2025-04-28"

    def scan(self, root: Path):
        root = ensure_root(root)
        clips = []
        for media in sorted(root.rglob("*")):
            if media.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            relative = media.relative_to(root).as_posix()
            parts = media.relative_to(root).parts
            subject = next(
                (part for part in parts if part.lower().startswith(("participant", "subject"))),
                parts[0] if parts else "unknown",
            )
            camera = next(
                (part for part in parts if "45" in part or "90" in part),
                "unknown",
            )
            label = next(
                (part for part in reversed(parts[:-1]) if part.lower() in {"normal", "abnormal", "fall"}),
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
