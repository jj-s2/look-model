from datetime import date, timedelta

from mental.trend import ActivitySummary, CheckinResult, DailyPhysiologySummary, WellbeingTrendAnalyzer


def test_single_day_deviation_is_observation_not_short_checkin_invitation():
    analyzer = WellbeingTrendAnalyzer()
    start = date(2026, 7, 1)
    for offset in range(14):
        analyzer.update(start + timedelta(days=offset), DailyPhysiologySummary(sleep_hours=7.0), ActivitySummary(steps=3000), CheckinResult(wellbeing=4))

    result = analyzer.update(start + timedelta(days=14), DailyPhysiologySummary(sleep_hours=4.0), ActivitySummary(steps=900), CheckinResult(wellbeing=1))

    assert result.baseline_ready is True
    assert result.observation_evidence
    assert result.invite_short_checkin is False


def test_sustained_deviation_after_seven_day_baseline_invites_short_checkin():
    analyzer = WellbeingTrendAnalyzer()
    start = date(2026, 7, 1)
    for offset in range(14):
        analyzer.update(start + timedelta(days=offset), DailyPhysiologySummary(sleep_hours=7.0), ActivitySummary(steps=3000), CheckinResult(wellbeing=4))

    analyzer.update(start + timedelta(days=14), DailyPhysiologySummary(sleep_hours=4.0), ActivitySummary(steps=900), CheckinResult(wellbeing=1))
    result = analyzer.update(start + timedelta(days=15), DailyPhysiologySummary(sleep_hours=4.0), ActivitySummary(steps=900), CheckinResult(wellbeing=1))

    assert result.sustained_change is True
    assert result.invite_short_checkin is True


def test_non_adjacent_days_do_not_count_as_sustained_change():
    analyzer = WellbeingTrendAnalyzer()
    start = date(2026, 7, 1)
    for offset in range(14):
        analyzer.update(start + timedelta(days=offset), DailyPhysiologySummary(sleep_hours=7.0), ActivitySummary(steps=3000), CheckinResult(wellbeing=4))
    analyzer.update(date(2026, 8, 1), DailyPhysiologySummary(sleep_hours=4.0), ActivitySummary(steps=900), CheckinResult(wellbeing=1))
    result = analyzer.update(date(2026, 8, 10), DailyPhysiologySummary(sleep_hours=4.0), ActivitySummary(steps=900), CheckinResult(wellbeing=1))
    assert result.sustained_change is False


def test_same_day_update_does_not_increment_sustained_change_streak():
    analyzer = WellbeingTrendAnalyzer()
    start = date(2026, 7, 1)
    for offset in range(14):
        analyzer.update(start + timedelta(days=offset), DailyPhysiologySummary(sleep_hours=7.0), ActivitySummary(steps=3000), CheckinResult(wellbeing=4))
    analyzer.update(start + timedelta(days=14), DailyPhysiologySummary(sleep_hours=4.0), ActivitySummary(steps=900), CheckinResult(wellbeing=1))
    result = analyzer.update(start + timedelta(days=14), DailyPhysiologySummary(sleep_hours=4.0), ActivitySummary(steps=900), CheckinResult(wellbeing=1))
    assert result.sustained_change is False


def test_out_of_order_day_is_rejected_to_preserve_natural_day_sequence():
    analyzer = WellbeingTrendAnalyzer()
    analyzer.update(date(2026, 8, 2), None, None, None)
    import pytest
    with pytest.raises(ValueError, match="chronological"):
        analyzer.update(date(2026, 8, 1), None, None, None)


def test_self_harm_expression_creates_highest_priority_human_review_event():
    result = WellbeingTrendAnalyzer().update(date(2026, 8, 2), None, None, CheckinResult(self_harm_expression=True))

    assert result.human_review_event is not None
    assert result.human_review_event.priority == "highest"
    assert result.human_review_event.reason == "self_harm_expression"


def test_empty_days_do_not_report_baseline_ready_or_operational():
    analyzer = WellbeingTrendAnalyzer()
    start = date(2026, 8, 1)
    result = None
    for offset in range(14):
        result = analyzer.update(start + timedelta(days=offset), None, None, None)
    assert result is not None
    assert result.baseline_ready is False
    assert result.baseline_state == "cold_start"


def test_invalid_numeric_values_are_rejected():
    import pytest

    with pytest.raises(ValueError, match="sleep_hours"):
        WellbeingTrendAnalyzer().update(
            date(2026, 8, 1), DailyPhysiologySummary(sleep_hours=float("nan")), None, None
        )
    with pytest.raises(ValueError, match="steps"):
        WellbeingTrendAnalyzer().update(
            date(2026, 8, 1), None, ActivitySummary(steps=-1), None
        )
    with pytest.raises(ValueError, match="wellbeing"):
        WellbeingTrendAnalyzer().update(
            date(2026, 8, 1), None, None, CheckinResult(wellbeing=11)
        )


def test_different_modalities_on_adjacent_days_do_not_form_one_sustained_streak():
    analyzer = WellbeingTrendAnalyzer()
    start = date(2026, 8, 1)
    for offset in range(14):
        analyzer.update(start + timedelta(days=offset), DailyPhysiologySummary(7.0, .8), ActivitySummary(3000), CheckinResult(4))
    analyzer.update(start + timedelta(days=14), DailyPhysiologySummary(7.0, .8), ActivitySummary(900), CheckinResult(4))
    result = analyzer.update(start + timedelta(days=15), DailyPhysiologySummary(4.0, .8), ActivitySummary(3000), CheckinResult(4))
    assert result.sustained_change is False


def test_same_feature_two_of_three_effective_days_forms_sustained_change():
    analyzer = WellbeingTrendAnalyzer()
    start = date(2026, 8, 1)
    for offset in range(14):
        analyzer.update(start + timedelta(days=offset), DailyPhysiologySummary(7.0, .8), ActivitySummary(3000), CheckinResult(4))
    analyzer.update(start + timedelta(days=14), DailyPhysiologySummary(7.0, .8), ActivitySummary(900), CheckinResult(4))
    result = analyzer.update(start + timedelta(days=15), DailyPhysiologySummary(7.0, .8), ActivitySummary(900), CheckinResult(4))
    assert result.sustained_change is True
    assert result.invite_short_checkin is True
