from datetime import datetime, timezone

from core.events import DataQuality, EventType, SensorEvent, Source
from pipeline.live_service import LiveMonitoringService


NOW = datetime(2026, 8, 2, 9, tzinfo=timezone.utc)


class FakeSource:
    name = "camera"

    def poll(self, now: datetime):
        return [
            SensorEvent(
                timestamp=NOW,
                source=Source.VISION,
                event_type=EventType.FALL_EVENT,
                payload={"subject_id": "demo-person", "confirmed": True},
                quality=DataQuality(True, 0.95, True, "demo_fixture"),
            )
        ]


def test_live_service_emits_decision_from_fake_sources() -> None:
    service = LiveMonitoringService(event_sources=[FakeSource()], clock=lambda: NOW)

    snapshot = service.step()

    assert snapshot.camera_health == "healthy"
    assert snapshot.demo is True
    assert any(item.kind == "fall_event" for item in snapshot.decisions)


def test_source_failure_is_degraded_without_stopping_other_sources() -> None:
    class BrokenSource:
        name = "radar"

        def poll(self, now: datetime):
            raise RuntimeError("fixture unavailable")

    service = LiveMonitoringService(event_sources=[BrokenSource(), FakeSource()], clock=lambda: NOW)

    snapshot = service.step()

    assert snapshot.radar_health == "degraded"
    assert any(item.kind == "fall_event" for item in snapshot.decisions)
