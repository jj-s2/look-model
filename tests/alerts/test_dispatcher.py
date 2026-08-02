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
        recovery_confirmed=True,
    )

    assert dispatcher.dispatch(FALL_DECISION).sent is True
    assert dispatcher.dispatch(recovery).sent is False
    assert dispatcher.dispatch(FALL_DECISION).sent is True


def test_cooldown_uses_rolling_timestamp_not_epoch_bucket(tmp_path) -> None:
    dispatcher = AlertDispatcher(tmp_path / "alerts.jsonl")
    first = RiskDecision(
        kind="fall_forecast", level="warning", score=.7,
        reasons=("vision score high", "radar unavailable"), quality="vision_only",
        recommended_action="check mobility", subject_id="elder-1",
        timestamp=datetime(2026, 8, 2, 9, 14, 59, tzinfo=timezone.utc),
    )
    second = RiskDecision(
        kind="fall_forecast", level="warning", score=.7,
        reasons=("vision score high", "radar unavailable"), quality="vision_only",
        recommended_action="check mobility", subject_id="elder-1",
        timestamp=datetime(2026, 8, 2, 9, 15, 1, tzinfo=timezone.utc),
    )

    assert dispatcher.dispatch(first).sent is True
    assert dispatcher.dispatch(second).sent is False


def test_restarted_dispatcher_rebuilds_active_fall_state(tmp_path) -> None:
    destination = tmp_path / "alerts.jsonl"
    assert AlertDispatcher(destination).dispatch(FALL_DECISION).sent is True

    restarted = AlertDispatcher(destination)

    assert restarted.dispatch(FALL_DECISION).sent is False


def test_restarted_dispatcher_keeps_confirmed_recovery_rearm(tmp_path) -> None:
    destination = tmp_path / "alerts.jsonl"
    dispatcher = AlertDispatcher(destination)
    recovery = RiskDecision(
        kind="fall_event", level="info", score=0.0,
        reasons=("person is standing", "vision stream available"), quality="vision_only",
        recommended_action="resume monitoring", subject_id="elder-1", timestamp=NOW,
        recovery_confirmed=True,
    )

    assert dispatcher.dispatch(FALL_DECISION).sent is True
    assert dispatcher.dispatch(recovery).sent is False

    assert AlertDispatcher(destination).dispatch(FALL_DECISION).sent is True


def test_recovery_is_structured_not_inferred_from_reason_text(tmp_path) -> None:
    dispatcher = AlertDispatcher(tmp_path / "alerts.jsonl")
    misleading_text = RiskDecision(
        kind="fall_event", level="info", score=0.0,
        reasons=("confirmed recovery", "vision stream available"), quality="vision_only",
        recommended_action="resume monitoring", subject_id="elder-1", timestamp=NOW,
    )
    recovery = RiskDecision(
        kind="fall_event", level="info", score=0.0,
        reasons=("person is standing", "vision stream available"), quality="vision_only",
        recommended_action="resume monitoring", subject_id="elder-1", timestamp=NOW,
        recovery_confirmed=True,
    )

    assert dispatcher.dispatch(FALL_DECISION).sent is True
    assert dispatcher.dispatch(misleading_text).sent is False
    assert dispatcher.dispatch(FALL_DECISION).sent is False
    assert dispatcher.dispatch(recovery).sent is False
    assert dispatcher.dispatch(FALL_DECISION).sent is True


def test_recent_alerts_reads_jsonl_history_after_restart(tmp_path) -> None:
    destination = tmp_path / "alerts.jsonl"
    AlertDispatcher(destination).dispatch(FALL_DECISION)

    history = AlertDispatcher(destination).recent_alerts()

    assert len(history) == 1
    assert history[0]["kind"] == "fall_event"
