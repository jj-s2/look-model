from datetime import datetime, timezone
import json

from alerts.dispatcher import AlertDispatcher
from fusion.decision_engine import RiskDecision


NOW = datetime(2026, 8, 2, 9, tzinfo=timezone.utc)
FALL_DECISION = RiskDecision(
    kind="fall_event",
    level="critical",
    score=1.0,
    reasons=("confirmed fall from vision", "radar evidence unavailable"),
    quality="vision_only",
    recommended_action="contact emergency responder",
    subject_id="elder-1",
    timestamp=NOW,
)


def test_same_fall_is_dispatched_only_once_within_cooldown(tmp_path) -> None:
    dispatcher = AlertDispatcher(tmp_path / "alerts.jsonl")

    assert dispatcher.dispatch(FALL_DECISION).sent is True
    assert dispatcher.dispatch(FALL_DECISION).sent is False


def test_dispatcher_writes_sent_decision_to_local_jsonl(tmp_path) -> None:
    destination = tmp_path / "alerts.jsonl"
    result = AlertDispatcher(destination).dispatch(FALL_DECISION)

    assert result.sent is True
    stored = json.loads(destination.read_text(encoding="utf-8").strip())
    assert stored["kind"] == "fall_event"
    assert stored["subject_id"] == "elder-1"


def test_confirmed_recovery_rearms_next_fall(tmp_path) -> None:
    dispatcher = AlertDispatcher(tmp_path / "alerts.jsonl")
    recovery = RiskDecision(
        kind="fall_event",
        level="info",
        score=0.0,
        reasons=("confirmed recovery", "vision stream available"),
        quality="vision_only",
        recommended_action="resume monitoring",
        subject_id="elder-1",
        timestamp=NOW,
    )

    assert dispatcher.dispatch(FALL_DECISION).sent is True
    assert dispatcher.dispatch(recovery).sent is False
    assert dispatcher.dispatch(FALL_DECISION).sent is True
