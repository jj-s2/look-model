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


def test_full_screening_invitation_has_its_own_28_day_cooldown():
    decision = InteractionPolicy().evaluate(context_at("2026-08-02T10:00:00+08:00", last_full_invitation="2026-07-10T10:00:00+08:00"))
    assert decision.invite_full_gds is False


def test_short_checkin_is_not_invited_twice_within_seven_days():
    policy = InteractionPolicy()
    context = context_at("2026-08-02T10:00:00+08:00", sustained_change=True, last_short_checkin="2026-07-30T10:00:00+08:00")
    assert policy.evaluate(context).invite_short_checkin is False


def test_short_checkin_invitation_has_its_own_seven_day_cooldown():
    decision = InteractionPolicy().evaluate(context_at("2026-08-02T10:00:00+08:00", sustained_change=True, last_short_invitation="2026-07-30T10:00:00+08:00"))
    assert decision.invite_short_checkin is False


def test_quiet_hours_suppress_non_emergency_prompt():
    policy = InteractionPolicy()
    context = context_at("2026-08-02T22:00:00+08:00", sustained_change=True)
    decision = policy.evaluate(context)
    assert decision.prompt is None
    assert decision.reason == "quiet_hours"


def test_quiet_hour_boundaries_allow_at_0800_and_suppress_at_2100():
    policy = InteractionPolicy()
    allowed = policy.evaluate(context_at("2026-08-02T08:00:00+08:00", sustained_change=True))
    quiet = policy.evaluate(context_at("2026-08-02T21:00:00+08:00", sustained_change=True))
    assert allowed.prompt == "short_checkin"
    assert quiet.reason == "quiet_hours"


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


def test_user_initiated_short_checkin_bypasses_invitation_cooldown():
    decision = InteractionPolicy().evaluate(context_at("2026-08-02T10:00:00+08:00", user_initiated_short_checkin=True, last_short_invitation="2026-08-02T09:00:00+08:00"))
    assert decision.prompt == "short_checkin"


def test_naive_timestamps_are_rejected_and_timezone_offsets_are_compared_correctly():
    policy = InteractionPolicy()
    import pytest
    with pytest.raises(ValueError, match="timezone-aware"):
        policy.evaluate(InteractionContext(now=datetime(2026, 8, 2, 10)))
    decision = policy.evaluate(context_at("2026-08-02T10:00:00+08:00", last_full_invitation="2026-07-05T02:00:00+00:00"))
    assert decision.invite_full_gds is True
