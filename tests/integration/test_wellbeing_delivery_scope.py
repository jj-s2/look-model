from datetime import datetime, timezone

from alerts.dispatcher import AlertDispatcher
from core.events import DataQuality, EventType, SensorEvent, Source
from fusion.decision_engine import DecisionEngine
from mental.contracts import WellbeingAssessmentEvent
from pipeline.live_service import LiveMonitoringService, SourceBatch


NOW = datetime(2026, 8, 12, 10, tzinfo=timezone.utc)


def wellbeing_event(*, state: str = "invite_candidate", quality: float = 0.9, available: bool = True) -> SensorEvent:
    return WellbeingAssessmentEvent(
        event_id="wb-1", subject_alias="senior-1", timestamp=NOW, state=state,
        action="local_invite_short_checkin" if state == "invite_candidate" else "local_record_only",
        evidence_domains=("activity",), evidence_codes=("activity_change",), coverage=0.9,
        quality=quality, uncertainty=1.0 - quality, baseline_state="operational",
        model_version="pace_behavior_v1",
    ).to_sensor_event() if available else SensorEvent(
        timestamp=NOW, source=Source.SCREENING, event_type=EventType.WELLBEING_CHANGE,
        payload={"subject_id": "senior-1", "state": "abstained", "delivery_scope": "external_forbidden", "sustained_change": False},
        quality=DataQuality(False, quality, False, "insufficient_coverage"),
    )


def test_low_quality_wellbeing_abstains_in_fusion() -> None:
    decision = DecisionEngine().evaluate([wellbeing_event(available=False, quality=0.2)], NOW)[0]
    assert decision.kind == "wellbeing_change"
    assert decision.quality == "degraded"
    assert decision.score == 0.0
    assert "unavailable" in " ".join(decision.reasons)
    assert decision.delivery_scope == "external_forbidden"


def test_local_only_wellbeing_never_reaches_dispatcher_but_fall_does(tmp_path) -> None:
    class RecordingDispatcher:
        def __init__(self):
            self.calls = []

        def dispatch(self, decision):
            self.calls.append(decision.kind)

        def recent_alerts(self, limit=None):
            return []

    dispatcher = RecordingDispatcher()
    service = LiveMonitoringService(dispatcher=dispatcher, clock=lambda: NOW)
    service.step(SourceBatch((wellbeing_event(),)))
    assert dispatcher.calls == []
    fall = SensorEvent(
        timestamp=NOW, source=Source.VISION, event_type=EventType.FALL_EVENT,
        payload={"subject_id": "senior-1", "confirmed": True}, quality=DataQuality(True, 0.95, False),
    )
    service.step(SourceBatch((fall,)))
    assert dispatcher.calls == ["fall_event"]


def test_dispatcher_has_a_second_scope_firewall_for_wellbeing(tmp_path) -> None:
    dispatcher = AlertDispatcher(tmp_path / "alerts.jsonl")
    decision = DecisionEngine().evaluate([wellbeing_event()], NOW)[0]
    result = dispatcher.dispatch(decision)
    assert result.sent is False
    assert "forbidden" in result.reason
    assert dispatcher.recent_alerts()[-1]["sent"] is False


def test_human_review_wellbeing_is_internal_only() -> None:
    event = WellbeingAssessmentEvent(
        event_id="wb-review", subject_alias="senior-1", timestamp=NOW, state="human_review_required",
        action="enqueue_human_review", evidence_domains=("voluntary_episode",),
        evidence_codes=("self_harm_candidate",), coverage=1.0, quality=0.9, uncertainty=0.1,
        baseline_state="operational", model_version="research_shadow_v1",
    ).to_sensor_event()
    decision = DecisionEngine().evaluate([event], NOW)[0]
    assert decision.delivery_scope == "external_forbidden"
    assert decision.kind == "wellbeing_change"
