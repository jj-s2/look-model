"""Auditable, dependency-light SDNL1 physiology source contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class PhysiologyRecord:
    """One timestamped measurement; raw source values remain available for audit."""

    timestamp: datetime
    heart_rate: float | None = None
    respiratory_rate: float | None = None
    sleep_stage: str | None = None
    source_fields: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_fields", dict(self.source_fields or {}))


@dataclass(frozen=True)
class PhysiologyQuality:
    available: bool
    demo: bool
    reason: str | None = None

    @property
    def display_message(self) -> str | None:
        if self.reason == "sdnl1_api_not_configured":
            return "设备在线但健康数据接口未接通"
        return self.reason


@dataclass(frozen=True)
class PhysiologyBatch:
    records: tuple[PhysiologyRecord, ...]
    quality: PhysiologyQuality


class PhysiologySource(Protocol):
    def poll(self, start: datetime, end: datetime) -> PhysiologyBatch:
        """Return measurements in ``[start, end)`` and their provenance."""
