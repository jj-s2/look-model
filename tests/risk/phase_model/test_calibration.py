import warnings

import numpy as np
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


def test_temperature_calibrator_accepts_numpy_arrays():
    result = TemperatureCalibrator().fit(
        np.array([-2.0, 2.0, -1.0, 1.0], dtype=np.float64),
        np.array([0, 1, 0, 1], dtype=np.int64),
        partition="validation",
    )

    assert result.sample_count == 4
    assert result.temperature > 0.0


def test_temperature_calibrator_rejects_empty_numpy_arrays_without_truth_testing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="non-empty"):
            TemperatureCalibrator().fit(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
                partition="validation",
            )
