"""Retention enforcement for locally stored, opted-in event clips."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path


class RetentionPolicy:
    """Remove only event clips whose local retention window has elapsed."""

    def __init__(self, directory: str | Path, *, keep_days: int = 7) -> None:
        if keep_days < 1:
            raise ValueError("keep_days must be at least one day")
        self.directory = Path(directory)
        self.keep_days = keep_days

    def prune(self, now: datetime) -> list[Path]:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        if not self.directory.exists():
            return []
        cutoff = now - timedelta(days=self.keep_days)
        removed: list[Path] = []
        for clip in sorted(self.directory.glob("event_*.npz")):
            modified_at = datetime.fromtimestamp(clip.stat().st_mtime, timezone.utc)
            if modified_at < cutoff.astimezone(timezone.utc):
                clip.unlink()
                removed.append(clip)
        return removed
