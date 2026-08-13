"""Memory-only pre-event frame buffering with explicit recording consent."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


class CircularClipBuffer:
    """Keep frames only in memory until an opted-in event capture completes."""

    def __init__(self, directory: str | Path, *, recording_opt_in: bool, pre_seconds: int = 10) -> None:
        if pre_seconds < 0:
            raise ValueError("pre_seconds must be non-negative")
        self.directory = Path(directory)
        self.recording_opt_in = recording_opt_in
        self.pre_seconds = pre_seconds
        self._frames: deque[tuple[datetime, Any]] = deque()
        self._pending: dict[str, tuple[datetime, datetime, list[tuple[datetime, Any]]]] = {}
        self._completed: dict[str, Path] = {}
        self._latest_timestamp: datetime | None = None

    @property
    def completed_clips(self) -> tuple[Path, ...]:
        return tuple(self._completed.values())

    def on_frame(self, frame: Any, timestamp: datetime) -> None:
        """Accept a frame without writing it to disk unless consent was granted."""
        self._validate_timestamp(timestamp)
        if not self.recording_opt_in:
            return
        self._latest_timestamp = timestamp
        self._frames.append((timestamp, frame))
        self._discard_before(timestamp - timedelta(seconds=self.pre_seconds))
        for event_id, (started_at, ends_at, frames) in tuple(self._pending.items()):
            if timestamp >= started_at and timestamp <= ends_at:
                frames.append((timestamp, frame))
            if timestamp >= ends_at:
                self._completed[event_id] = self._write_clip(event_id, started_at, frames)
                del self._pending[event_id]

    def confirm_event(self, event_id: str, post_seconds: int = 20) -> Path | None:
        """Start a 10-second-before/``post_seconds``-after opted-in event clip."""
        if not event_id:
            raise ValueError("event_id is required")
        if post_seconds < 0:
            raise ValueError("post_seconds must be non-negative")
        if not self.recording_opt_in:
            return None
        if event_id in self._completed:
            return self._completed[event_id]
        if event_id in self._pending:
            return None
        event_at = self._latest_timestamp or datetime.now(timezone.utc)
        pre_start = event_at - timedelta(seconds=self.pre_seconds)
        frames = [(timestamp, frame) for timestamp, frame in self._frames if pre_start <= timestamp <= event_at]
        end_at = event_at + timedelta(seconds=post_seconds)
        self._pending[event_id] = (event_at, end_at, frames)
        if post_seconds == 0:
            self._completed[event_id] = self._write_clip(event_id, event_at, frames)
            del self._pending[event_id]
            return self._completed[event_id]
        return None

    def _discard_before(self, earliest: datetime) -> None:
        while self._frames and self._frames[0][0] < earliest:
            self._frames.popleft()

    def _write_clip(self, event_id: str, event_at: datetime, frames: list[tuple[datetime, Any]]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in event_id)
        destination = self.directory / f"event_{safe_id}_{event_at.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}.npz"
        timestamps = np.asarray([timestamp.isoformat() for timestamp, _ in frames])
        values = np.asarray([frame for _, frame in frames])
        np.savez_compressed(destination, timestamps=timestamps, frames=values)
        return destination

    @staticmethod
    def _validate_timestamp(timestamp: datetime) -> None:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
