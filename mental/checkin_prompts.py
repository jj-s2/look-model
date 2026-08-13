"""Low-burden, consent-gated local prompts for wellbeing check-ins."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from mental.contracts import WellbeingAssessmentEvent


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


@dataclass(frozen=True)
class CheckinPromptState:
    invited_at: datetime | None = None
    declined: bool = False
    stopped: bool = False

    def __post_init__(self) -> None:
        if self.invited_at is not None:
            _aware(self.invited_at, "invited_at")
        if not isinstance(self.declined, bool) or not isinstance(self.stopped, bool):
            raise ValueError("declined and stopped must be boolean")


def apply_prompt_action(state: CheckinPromptState, action: str, at: datetime) -> CheckinPromptState:
    if not isinstance(state, CheckinPromptState):
        raise ValueError("state must be CheckinPromptState")
    timestamp = _aware(at, "at")
    if action not in {"accept", "decline", "stop"}:
        raise ValueError("action must be accept, decline, or stop")
    if state.stopped:
        return state
    if action == "accept":
        return CheckinPromptState(invited_at=timestamp, declined=False, stopped=False)
    if action == "decline":
        return CheckinPromptState(invited_at=state.invited_at, declined=True, stopped=False)
    return CheckinPromptState(invited_at=state.invited_at, declined=state.declined, stopped=True)


def build_short_checkin_prompt(
    event: WellbeingAssessmentEvent,
    *,
    now: datetime,
    state: CheckinPromptState | None = None,
) -> dict[str, object]:
    if not isinstance(event, WellbeingAssessmentEvent):
        raise ValueError("event must be WellbeingAssessmentEvent")
    timestamp = _aware(now, "now")
    prompt_state = state or CheckinPromptState()
    base = {
        "subject_alias": event.subject_alias,
        "event_id": event.event_id,
        "delivery_scope": "external_forbidden",
        "consent_required": True,
        "is_diagnosis": False,
        "prompts": (),
    }
    if prompt_state.stopped:
        return {**base, "active": False, "reason": "stopped"}
    if prompt_state.declined:
        return {**base, "active": False, "reason": "declined"}
    if event.state != "invite_candidate" or event.action != "local_invite_short_checkin":
        return {**base, "active": False, "reason": "no_candidate"}
    if prompt_state.invited_at is not None:
        elapsed = timestamp - prompt_state.invited_at
        if elapsed < timedelta(days=7):
            return {**base, "active": False, "reason": "cooldown"}
    return {
        **base,
        "active": True,
        "reason": "sustained_change",
        "prompts": (
            "这几天睡得怎么样？",
            "最近一天里，做事情的精力怎么样？",
            "最近有没有什么事情让你觉得不舒服或担心？",
        ),
    }


def build_prompt_for_decision(decision: object, *, now: datetime) -> dict[str, object] | None:
    """Create a redaction-safe prompt envelope from a fusion decision."""
    timestamp = _aware(now, "now")
    if getattr(decision, "kind", None) != "wellbeing_change":
        return None
    if getattr(decision, "recommended_action", "") != "invite a voluntary short wellbeing check-in":
        return None
    if getattr(decision, "delivery_scope", "external_forbidden") != "external_forbidden":
        return None
    return {
        "active": True,
        "reason": "sustained_change",
        "event_id": f"decision-{getattr(decision, 'subject_id', 'unknown')}-{timestamp.isoformat()}",
        "subject_alias": str(getattr(decision, "subject_id", "unknown")),
        "delivery_scope": "external_forbidden",
        "consent_required": True,
        "is_diagnosis": False,
        "prompts": (
            "这几天睡得怎么样？",
            "最近一天里，做事情的精力怎么样？",
            "最近有没有什么事情让你觉得不舒服或担心？",
        ),
    }
