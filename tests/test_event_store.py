from datetime import datetime, timedelta, timezone

import pytest

from core.event_store import JsonlEventStore
from core.events import DataQuality, EventType, SensorEvent, Source


def test_query_filters_time_and_type(tmp_path):
    store = JsonlEventStore(tmp_path / "events.jsonl")
    now = datetime.now(timezone.utc)
    store.append(SensorEvent(now, Source.VISION, EventType.FALL_EVENT, {}, DataQuality(True, 1.0, False)))
    store.append(SensorEvent(now, Source.RADAR, EventType.PHYSIOLOGY, {}, DataQuality(False, 0.0, False)))

    result = store.query(now - timedelta(seconds=1), now + timedelta(seconds=1), {EventType.FALL_EVENT})

    assert [item.event_type for item in result] == [EventType.FALL_EVENT]


def test_query_rejects_corrupt_jsonl_without_exposing_payload(tmp_path):
    event_file = tmp_path / "events.jsonl"
    secret = "patient-secret-123"
    event_file.write_text(
        "{"
        '\"timestamp\": \"2026-08-02T00:00:00+00:00\", '
        '\"source\": \"vision\", '
        '\"event_type\": \"fall_event\", '
        f'\"payload\": {{\"token\": \"{secret}\"}}, '
        '\"quality\": {\"available\": true, \"confidence\": 1.0, \"demo\": false}'
        "}\nnot-json\n",
        encoding="utf-8",
    )

    store = JsonlEventStore(event_file)
    with pytest.raises(ValueError, match="line 2") as error:
        store.query(
            datetime(2026, 8, 2, tzinfo=timezone.utc),
            datetime(2026, 8, 3, tzinfo=timezone.utc),
        )

    assert secret not in str(error.value)
