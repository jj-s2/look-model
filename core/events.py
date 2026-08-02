"""Typed, serializable sensor-event contracts."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from numbers import Real
from typing import Mapping


class Source(str, Enum):
    VISION = "vision"
    RADAR = "radar"
    SCREENING = "screening"
    SYSTEM = "system"


class EventType(str, Enum):
    POSE = "pose"
    FALL_EVENT = "fall_event"
    FALL_FORECAST = "fall_forecast"
    PHYSIOLOGY = "physiology"
    WELLBEING_CHANGE = "wellbeing_change"
    AVAILABILITY = "availability"


@dataclass(frozen=True)
class DataQuality:
    available: bool
    confidence: float
    demo: bool
    reason: str | None = None


@dataclass(frozen=True)
class SensorEvent:
    timestamp: datetime
    source: Source
    event_type: EventType
    payload: Mapping[str, object]
    quality: DataQuality

    def __post_init__(self) -> None:
        if not isinstance(self.source, Source):
            raise ValueError("source must be a Source")
        if not isinstance(self.event_type, EventType):
            raise ValueError("event_type must be an EventType")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if not isinstance(self.quality.available, bool) or not isinstance(self.quality.demo, bool):
            raise ValueError("quality availability and demo flags must be boolean")
        if (
            isinstance(self.quality.confidence, bool)
            or not isinstance(self.quality.confidence, Real)
            or not 0.0 <= self.quality.confidence <= 1.0
        ):
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "source": self.source.value,
            "event_type": self.event_type.value,
            "payload": dict(self.payload),
            "quality": {
                "available": self.quality.available,
                "confidence": self.quality.confidence,
                "demo": self.quality.demo,
                "reason": self.quality.reason,
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SensorEvent":
        try:
            quality_data = data["quality"]
            payload = data["payload"]
            if not isinstance(quality_data, Mapping) or not isinstance(payload, Mapping):
                raise TypeError("event fields must be mappings")
            return cls(
                timestamp=datetime.fromisoformat(str(data["timestamp"])),
                source=Source(str(data["source"])),
                event_type=EventType(str(data["event_type"])),
                payload=dict(payload),
                quality=DataQuality(
                    available=quality_data["available"],  # type: ignore[arg-type]
                    confidence=quality_data["confidence"],  # type: ignore[arg-type]
                    demo=quality_data["demo"],  # type: ignore[arg-type]
                    reason=quality_data.get("reason"),  # type: ignore[arg-type]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid sensor event") from error
