from datetime import datetime, timezone

from core.events import DataQuality, EventType, SensorEvent, Source
from pipeline.live_service import LiveMonitoringService, SourceBatch
from ui.dashboard import dashboard_view_model, start_gds15_screening


NOW = datetime(2026, 8, 12, 10, tzinfo=timezone.utc)


def test_gds_payload_never_exposes_risk_answer() -> None:
    payload = start_gds15_screening()
    assert all("risk_answer" not in question for question in payload["questions"])
    assert all(len(question["options"]) == 2 for question in payload["questions"])


def test_wellbeing_view_model_uses_safe_chinese_labels_and_uncertainty() -> None:
    event = SensorEvent(
        timestamp=NOW, source=Source.SCREENING, event_type=EventType.WELLBEING_CHANGE,
        payload={"subject_id": "senior-1", "state": "invite_candidate", "delivery_scope": "external_forbidden", "uncertainty": 0.2},
        quality=DataQuality(True, 0.8, False),
    )
    snapshot = LiveMonitoringService(clock=lambda: NOW).step(SourceBatch((event,)))
    model = dashboard_view_model(snapshot)
    item = model["wellbeing_changes"][0]
    assert item["state_zh"] == "建议自愿简短问候"
    assert item["delivery_scope"] == "external_forbidden"
    assert item["uncertainty"] == 0.2
