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
