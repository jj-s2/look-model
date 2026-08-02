from datetime import date, timedelta

from mental.trend import ActivitySummary, CheckinResult, DailyPhysiologySummary, WellbeingTrendAnalyzer


def test_single_day_deviation_is_observation_not_short_checkin_invitation():
    analyzer = WellbeingTrendAnalyzer()
    start = date(2026, 7, 1)
    for offset in range(7):
        analyzer.update(start + timedelta(days=offset), DailyPhysiologySummary(sleep_hours=7.0), ActivitySummary(steps=3000), CheckinResult(wellbeing=4))

    result = analyzer.update(start + timedelta(days=7), DailyPhysiologySummary(sleep_hours=4.0), ActivitySummary(steps=900), CheckinResult(wellbeing=1))

    assert result.baseline_ready is True
    assert result.observation_evidence
    assert result.invite_short_checkin is False


def test_sustained_deviation_after_seven_day_baseline_invites_short_checkin():
    analyzer = WellbeingTrendAnalyzer()
    start = date(2026, 7, 1)
    for offset in range(7):
        analyzer.update(start + timedelta(days=offset), DailyPhysiologySummary(sleep_hours=7.0), ActivitySummary(steps=3000), CheckinResult(wellbeing=4))

    analyzer.update(start + timedelta(days=7), DailyPhysiologySummary(sleep_hours=4.0), ActivitySummary(steps=900), CheckinResult(wellbeing=1))
    result = analyzer.update(start + timedelta(days=8), DailyPhysiologySummary(sleep_hours=4.0), ActivitySummary(steps=900), CheckinResult(wellbeing=1))

    assert result.sustained_change is True
    assert result.invite_short_checkin is True


def test_self_harm_expression_creates_highest_priority_human_review_event():
    result = WellbeingTrendAnalyzer().update(date(2026, 8, 2), None, None, CheckinResult(self_harm_expression=True))

    assert result.human_review_event is not None
    assert result.human_review_event.priority == "highest"
    assert result.human_review_event.reason == "self_harm_expression"

