import pytest

from risk.pmcc.survival import SurvivalLabel, cumulative_risk_for_horizons, hazards_to_cumulative


def test_cumulative_risk_is_monotonic_and_uses_conditional_hazards():
    hazards = (0.1, 0.2, 0.05, 0.0, 0.1, 0.0, 0.0)
    values = hazards_to_cumulative(hazards)

    assert len(values) == 7
    assert list(values) == sorted(values)
    assert values[2] == pytest.approx(1 - (1 - 0.1) * (1 - 0.2) * (1 - 0.05))


def test_horizon_mapping_uses_1_3_and_7_day_cumulative_risk():
    risk = cumulative_risk_for_horizons((0.1,) * 7)

    assert risk == {
        "24h": pytest.approx(0.1),
        "72h": pytest.approx(1 - 0.9**3),
        "7d": pytest.approx(1 - 0.9**7),
    }


@pytest.mark.parametrize("hazards", [(0.1,) * 6, (0.1,) * 8, (0.1,) * 6 + (1.2,), (0.1,) * 6 + (float("nan"),)])
def test_hazards_must_be_seven_finite_probabilities(hazards):
    with pytest.raises(ValueError, match="seven|\[0, 1\]|finite"):
        hazards_to_cumulative(hazards)


def test_survival_label_validates_event_against_censoring_window():
    assert SurvivalLabel(event_day=None, censor_day=7).censor_day == 7
    with pytest.raises(ValueError, match="censor_day"):
        SurvivalLabel(event_day=None, censor_day=0)
    with pytest.raises(ValueError, match="event_day"):
        SurvivalLabel(event_day=4, censor_day=3)
