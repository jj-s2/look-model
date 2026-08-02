from datetime import datetime, timezone

from core.events import DataQuality, EventType, SensorEvent, Source
from pipeline.live_service import LiveMonitoringService
from ui.dashboard import dashboard_view_model


NOW = datetime(2026, 8, 2, 9, tzinfo=timezone.utc)


class DemoSource:
    name = "camera"

    def poll(self, now: datetime):
        return [
            SensorEvent(
                timestamp=NOW,
                source=Source.VISION,
                event_type=EventType.FALL_EVENT,
                payload={"confirmed": True},
                quality=DataQuality(True, 0.9, True, "demo_fixture"),
            )
        ]


def test_dashboard_view_model_marks_demo_and_screening_as_non_diagnostic() -> None:
    model = dashboard_view_model(LiveMonitoringService([DemoSource()], clock=lambda: NOW).step())

    assert model["watermark"] == "演示数据 / demo=true"
    assert "筛查不构成诊断" in model["screening_notice"]
    assert model["fall_events"]
