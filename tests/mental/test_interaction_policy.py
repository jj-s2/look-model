from datetime import datetime, timedelta, timezone

from mental.interaction_policy import InteractionContext, InteractionPolicy


TZ = timezone(timedelta(hours=8))


def context_at(value, **changes):
    data = {"now": datetime.fromisoformat(value), "sustained_change": False}
    data.update(changes)
    return InteractionContext(**data)


def test_full_screening_is_not_invited_twice_within_28_days():
    policy = InteractionPolicy()
    context = context_at("2026-08-02T10:00:00+08:00", last_full_screening="2026-07-10T10:00:00+08:00")
    assert policy.evaluate(context).invite_full_gds is False


def test_short_checkin_is_not_invited_twice_within_seven_days():
    policy = InteractionPolicy()
    context = context_at("2026-08-02T10:00:00+08:00", sustained_change=True, last_short_checkin="2026-07-30T10:00:00+08:00")
    assert policy.evaluate(context).invite_short_checkin is False


def test_quiet_hours_suppress_non_emergency_prompt():
    policy = InteractionPolicy()
    context = context_at("2026-08-02T22:00:00+08:00", sustained_change=True)
    decision = policy.evaluate(context)
    assert decision.prompt is None
    assert decision.reason == "quiet_hours"


def test_confirmed_fall_allows_one_immediate_check():
    policy = InteractionPolicy()
    first = policy.evaluate(context_at("2026-08-02T22:00:00+08:00", confirmed_fall=True))
    second = policy.evaluate(context_at("2026-08-02T22:01:00+08:00", confirmed_fall=True, fall_check_already_sent=True))
    assert first.prompt == "fall_confirmation"
    assert second.prompt is None


def test_user_initiated_full_screening_bypasses_invitation_cooldown():
    decision = InteractionPolicy().evaluate(context_at("2026-08-02T10:00:00+08:00", user_initiated_full_gds=True, last_full_screening="2026-08-02T09:00:00+08:00"))
    assert decision.prompt == "full_gds"
    assert decision.invite_full_gds is False
