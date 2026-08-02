from datetime import date, timedelta

import pytest

from risk.personal_baseline import RobustPersonalBaseline


DAYS = [date(2026, 1, 1) + timedelta(days=index) for index in range(9)]


def _ready_baseline():
    baseline = RobustPersonalBaseline(min_valid_days=7)
    for index, day in enumerate(DAYS[:7]):
        baseline.add_day(day, {"sway": 0.20 + index * 0.01}, valid=True)
    return baseline


def test_baseline_is_not_ready_before_seven_valid_days():
    baseline = RobustPersonalBaseline(min_valid_days=7)
    for day in DAYS[:6]:
        baseline.add_day(day, {"sway": 0.2}, valid=True)

    assert baseline.ready is False
    assert baseline.score({"sway": 0.2}).baseline_ready is False


def test_anomalous_day_does_not_shift_reference_distribution():
    baseline = _ready_baseline()
    before = baseline.reference["sway"]

    baseline.add_day(DAYS[7], {"sway": 99.0}, valid=False)

    assert baseline.reference["sway"] == before
    assert baseline.valid_day_count == 7


def test_score_uses_iqr_floor_for_constant_personal_history():
    baseline = RobustPersonalBaseline(min_valid_days=7, iqr_floor=0.01)
    for day in DAYS[:7]:
        baseline.add_day(day, {"sway": 0.2}, valid=True)

    score = baseline.score({"sway": 0.21})

    assert score.baseline_ready is True
    assert score.deviations["sway"] == pytest.approx(1.0)
    assert 0.0 <= score.score <= 1.0


def test_score_marks_a_feature_unready_until_it_has_seven_valid_days():
    baseline = RobustPersonalBaseline(min_valid_days=7)
    for day in DAYS[:7]:
        baseline.add_day(day, {"sway": 0.2}, valid=True)
    baseline.add_day(DAYS[7], {"step_width": 0.4}, valid=True)

    score = baseline.score({"sway": 0.3, "step_width": 0.5})

    assert baseline.ready is False
    assert score.baseline_ready is False
    assert score.unready_features == ("step_width",)
    assert "step_width" not in score.deviations
    assert score.score == 0.0
