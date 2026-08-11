from datetime import datetime, timezone

from core.events import DataQuality, EventType, SensorEvent, Source
from pipeline.live_service import LiveMonitoringService
from ui.dashboard import dashboard_view_model, start_gds15_screening, submit_gds15_answers
from fusion.decision_engine import RiskDecision
from pipeline.live_service import ServiceSnapshot


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
    answers = {question["id"]: False for question in start["questions"]}

    result = submit_gds15_answers(answers)

    assert result["score"] == 5
    assert result["is_diagnosis"] is False
    assert "筛查不构成诊断" in result["notice"]


def test_dashboard_separates_emergency_prefall_forecast_and_wellbeing():
    decisions = (
        RiskDecision("fall_event", "critical", 1.0, (), "vision_only", "check", "a", NOW),
        RiskDecision("prefall_warning", "warning", .8, (), "vision_only", "check", "a", NOW),
        RiskDecision("fall_forecast", "warning", .7, (), "vision_only", "check", "a", NOW),
        RiskDecision("wellbeing_change", "watch", .4, (), "screening_only", "check", "a", NOW),
    )
    snapshot = ServiceSnapshot(NOW, "healthy", "offline", decisions, (), False)
    model = dashboard_view_model(snapshot)
    assert len(model["emergency_events"]) == 1
    assert len(model["prefall_warnings"]) == 1
    assert len(model["fall_forecasts"]) == 1
    assert len(model["wellbeing_prompts"]) == 1
