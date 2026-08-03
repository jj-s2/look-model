"""Timestamp-aware short and long pose windows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from .schema import PoseObservation


@dataclass(frozen=True)
class DualWindow:
    short: tuple[PoseObservation, ...]
    long: tuple[PoseObservation, ...]

    @property
    def timestamp(self) -> datetime:
        return self.short[-1].timestamp if self.short else self.long[-1].timestamp


class DualTimescaleBuffer:
    """Keep a bounded history and sample each branch against wall-clock time."""

    def __init__(
        self,
        *,
        short_frames: int = 48,
        short_fps: float = 10.0,
        long_frames: int = 64,
        long_fps: float = 2.0,
        min_coverage: float = 0.8,
        history_seconds: float = 40.0,
    ) -> None:
        if min(short_frames, long_frames) <= 0 or min(short_fps, long_fps) <= 0:
            raise ValueError("frame counts and fps must be positive")
        if not 0.0 < min_coverage <= 1.0:
            raise ValueError("min_coverage must be in (0, 1]")
        self.short_frames = int(short_frames)
        self.short_fps = float(short_fps)
        self.long_frames = int(long_frames)
        self.long_fps = float(long_fps)
        self.min_coverage = float(min_coverage)
        self.history_seconds = float(history_seconds)
        self._history: deque[PoseObservation] = deque()
        self._last_timestamp: datetime | None = None

    def append(self, observation: PoseObservation) -> DualWindow | None:
        if self._last_timestamp is not None and observation.timestamp < self._last_timestamp:
            raise ValueError("observation timestamps must be monotonic")
        self._last_timestamp = observation.timestamp
        self._history.append(observation)
        cutoff = observation.timestamp - timedelta(seconds=self.history_seconds)
        while self._history and self._history[0].timestamp < cutoff:
            self._history.popleft()
        short = self._sample(observation.timestamp, self.short_frames, self.short_fps)
        long = self._sample(observation.timestamp, self.long_frames, self.long_fps)
        if short is None or long is None:
            return None
        return DualWindow(short=short, long=long)

    def _sample(self, end: datetime, frames: int, fps: float) -> tuple[PoseObservation, ...] | None:
        interval = timedelta(seconds=1.0 / fps)
        tolerance = interval * 0.48
        targets = [end - interval * offset for offset in reversed(range(frames))]
        candidates = list(self._history)
        selected: list[PoseObservation] = []
        used: set[int] = set()
        for target in targets:
            ranked = sorted(
                ((abs((item.timestamp - target).total_seconds()), index, item) for index, item in enumerate(candidates) if index not in used),
                key=lambda value: value[0],
            )
            if not ranked or ranked[0][0] > tolerance.total_seconds():
                continue
            _, index, item = ranked[0]
            used.add(index)
            selected.append(item)
        if len(selected) < frames * self.min_coverage:
            return None
        selected.sort(key=lambda item: item.timestamp)
        # A model window must have the requested length; sparse gaps are
        # rejected above instead of repeating stale observations.
        if len(selected) != frames:
            return None
        return tuple(selected)


def last_non_none(items: Iterable[DualWindow | None]) -> DualWindow | None:
    """Return the last emitted window, useful for streaming callers and tests."""
    result = None
    for item in items:
        if item is not None:
            result = item
    return result
