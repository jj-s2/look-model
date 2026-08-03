import pytest

from risk.phase_model.calibration import TemperatureCalibrator


def test_calibrator_refuses_test_partition():
    calibrator = TemperatureCalibrator()
    with pytest.raises(ValueError, match="validation"):
        calibrator.fit([0.0, 1.0], [0, 1], partition="test")


def test_calibrator_fits_positive_temperature_and_reports_metrics():
    calibrator = TemperatureCalibrator()
    result = calibrator.fit([-2.0, 2.0, -1.0, 1.0], [0, 1, 0, 1], partition="validation")
    assert result.temperature > 0
    assert result.sample_count == 4
    assert "brier_before" in result.metrics and "ece_after" in result.metrics
