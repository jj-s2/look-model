"""Redaction-safe health information for a live video stream."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


StreamState = Literal["connecting", "healthy", "degraded", "offline", "closed"]


@dataclass(frozen=True)
class StreamHealth:
    """The observable stream state without a URL, token, or device identifier."""

    state: StreamState = "connecting"
    consecutive_failures: int = 0
    last_success_at: datetime | None = None
    reason: str | None = None
