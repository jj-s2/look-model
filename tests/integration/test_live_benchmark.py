import pytest

from scripts.benchmark_live_pipeline import false_alarms_per_hour


def test_false_alarm_rate_uses_observed_duration():
    assert false_alarms_per_hour(false_alerts=2, duration_seconds=1800) == 4.0


@pytest.mark.parametrize("false_alerts,duration", [(-1, 10), (1, 0)])
def test_false_alarm_rate_rejects_invalid_observation(false_alerts, duration):
    with pytest.raises(ValueError):
        false_alarms_per_hour(false_alerts, duration)
