from datetime import datetime, timezone

import pytest

from core.events import DataQuality, EventType, SensorEvent, Source


def test_event_round_trip_preserves_quality_and_payload():
    event = SensorEvent(
        timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
        source=Source.VISION,
        event_type=EventType.FALL_EVENT,
        payload={"probability": 0.91},
        quality=DataQuality(available=True, confidence=0.88, demo=False),
    )

    assert SensorEvent.from_dict(event.to_dict()) == event


def test_event_rejects_naive_timestamp():
    with pytest.raises(ValueError, match="timezone-aware"):
        SensorEvent(
            timestamp=datetime(2026, 8, 2),
            source=Source.VISION,
            event_type=EventType.FALL_EVENT,
            payload={},
            quality=DataQuality(True, 1.0, False),
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_event_rejects_confidence_outside_unit_interval(confidence):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SensorEvent(
            timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
            source=Source.VISION,
            event_type=EventType.FALL_EVENT,
            payload={},
            quality=DataQuality(True, confidence, False),
        )


@pytest.mark.parametrize("field", ["available", "demo"])
def test_from_dict_rejects_non_boolean_quality_flags(field):
    data = SensorEvent(
        timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
        source=Source.VISION,
        event_type=EventType.FALL_EVENT,
        payload={},
        quality=DataQuality(True, 1.0, False),
    ).to_dict()
    quality = dict(data["quality"])
    quality[field] = "false"
    data["quality"] = quality

    with pytest.raises(ValueError):
        SensorEvent.from_dict(data)


@pytest.mark.parametrize(
    ("source", "event_type"),
    [("vision", EventType.FALL_EVENT), (Source.VISION, "fall_event")],
)
def test_event_rejects_non_enum_source_and_event_type(source, event_type):
    with pytest.raises(ValueError, match="Source|EventType"):
        SensorEvent(
            timestamp=datetime(2026, 8, 2, tzinfo=timezone.utc),
            source=source,
            event_type=event_type,
            payload={},
            quality=DataQuality(True, 1.0, False),
        )
