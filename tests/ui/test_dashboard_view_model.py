from datetime import datetime, timezone

from core.events import DataQuality, EventType, SensorEvent, Source
from pipeline.live_service import LiveMonitoringService
from ui.dashboard import dashboard_view_model, start_gds15_screening, submit_gds15_answers


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


def test_user_initiated_gds_flow_loads_fifteen_questions_then_scores_answers() -> None:
    start = start_gds15_screening()

    assert start["active"] is True
    assert len(start["questions"]) == 15
    answers = {question["id"]: question["risk_answer"] for question in start["questions"]}

    result = submit_gds15_answers(answers)

    assert result["score"] == 15
    assert result["is_diagnosis"] is False
    assert "筛查不构成诊断" in result["notice"]
