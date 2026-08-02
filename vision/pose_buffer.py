"""Fixed-size, overlapping pose-frame windows for temporal classifiers."""
from __future__ import annotations

from collections import deque

from .pose_pipeline import PoseFrameResult


class PoseSequenceBuffer:
    def __init__(self, window_size: int, stride: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stride <= 0 or stride > window_size:
            raise ValueError("stride must be between 1 and window_size")
        self._window_size = window_size
        self._stride = stride
        self._frames: deque[PoseFrameResult] = deque()

    def append(self, result: PoseFrameResult) -> list[PoseFrameResult] | None:
        self._frames.append(result)
        if len(self._frames) < self._window_size:
            return None
        window = list(self._frames)
        for _ in range(self._stride):
            self._frames.popleft()
        return window


__all__ = ["PoseSequenceBuffer"]
