from datetime import datetime, timedelta, timezone

from mental.checkin_prompts import CheckinPromptState, apply_prompt_action, build_short_checkin_prompt
from mental.contracts import WellbeingAssessmentEvent


NOW = datetime(2026, 8, 12, 10, tzinfo=timezone.utc)


def candidate() -> WellbeingAssessmentEvent:
    return WellbeingAssessmentEvent(
        event_id="wb-1", subject_alias="senior-1", timestamp=NOW, state="invite_candidate",
        action="local_invite_short_checkin", evidence_domains=("activity",),
        evidence_codes=("activity_change",), coverage=0.9, quality=0.8, uncertainty=0.2,
        baseline_state="operational", model_version="pace_behavior_v1",
    )


def test_prompt_is_short_plain_language_and_local_only() -> None:
    payload = build_short_checkin_prompt(candidate(), now=NOW)
    assert payload["active"] is True
    assert 2 <= len(payload["prompts"]) <= 3
    assert payload["delivery_scope"] == "external_forbidden"
    assert payload["consent_required"] is True
    assert all(len(prompt) < 80 for prompt in payload["prompts"])


def test_decline_and_stop_persist_without_reprompting() -> None:
    declined = apply_prompt_action(CheckinPromptState(), "decline", NOW)
    stopped = apply_prompt_action(declined, "stop", NOW + timedelta(minutes=1))
    assert build_short_checkin_prompt(candidate(), now=NOW + timedelta(days=1), state=declined)["reason"] == "declined"
    assert build_short_checkin_prompt(candidate(), now=NOW + timedelta(days=1), state=stopped)["reason"] == "stopped"


def test_prompt_budget_has_seven_day_cooldown() -> None:
    invited = apply_prompt_action(CheckinPromptState(), "accept", NOW)
    assert build_short_checkin_prompt(candidate(), now=NOW + timedelta(days=3), state=invited)["reason"] == "cooldown"
    assert build_short_checkin_prompt(candidate(), now=NOW + timedelta(days=8), state=invited)["active"] is True
