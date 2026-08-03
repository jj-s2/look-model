import importlib
import sys

import pytest

from risk.pmcc.features import FeatureWindow
from risk.pmcc.survival import SurvivalLabel


def _window(*, feature_names=("sleep:raw", "chain_strength"), values=None, missing=None, quality=None):
    width = len(feature_names)
    values = values or tuple((1.0, 0.2) for _ in range(14))
    missing = missing or tuple((False,) * width for _ in range(14))
    quality = quality or tuple((1.0,) * width for _ in range(14))
    return FeatureWindow(values=values, missing_mask=missing, quality=quality, feature_names=feature_names)


def _training_windows():
    quiet = _window()
    elevated = _window(values=tuple((4.0, 0.9) for _ in range(14)))
    return (quiet, elevated, quiet, elevated)


def test_cpu_calibrator_module_imports_without_eager_sklearn_import():
    # Regression target: importing edge inference contracts must not require ML extras.
    sys.modules.pop("risk.pmcc.calibrator", None)
    sys.modules.pop("sklearn", None)

    module = importlib.import_module("risk.pmcc.calibrator")

    assert "sklearn" not in sys.modules
    assert module.RuleSurvivalCalibrator is not None


def test_calibrator_refuses_training_labels_with_only_one_class():
    from risk.pmcc.calibrator import RuleSurvivalCalibrator

    with pytest.raises(ValueError, match="both event and non-event"):
        RuleSurvivalCalibrator().fit(_training_windows(), (SurvivalLabel(None, 7),) * 4)


def test_fitted_calibrator_returns_seven_bounded_hazards_and_schema_metadata():
    from risk.pmcc.calibrator import RuleSurvivalCalibrator

    model = RuleSurvivalCalibrator(random_seed=11).fit(
        _training_windows(),
        (SurvivalLabel(None, 7), SurvivalLabel(2, 7), SurvivalLabel(None, 7), SurvivalLabel(1, 7)),
    )

    hazards = model.predict_hazards(_training_windows()[1])
    metadata = model.metadata()

    assert len(hazards) == 7
    assert all(0.0 <= hazard <= 1.0 for hazard in hazards)
    assert metadata["feature_schema"] == ["sleep:raw", "chain_strength"]
    assert metadata["class_weight"] == "balanced"
    assert metadata["subject_weighting"] == "one window per subject-equivalent training row"


def test_calibrator_rejects_feature_schema_drift_at_prediction_time():
    from risk.pmcc.calibrator import RuleSurvivalCalibrator

    model = RuleSurvivalCalibrator().fit(
        _training_windows(),
        (SurvivalLabel(None, 7), SurvivalLabel(2, 7), SurvivalLabel(None, 7), SurvivalLabel(1, 7)),
    )

    with pytest.raises(ValueError, match="feature schema mismatch"):
        model.predict_hazards(_window(feature_names=("chain_strength", "sleep:raw")))


def test_synthetic_artifacts_cannot_be_promoted_in_metadata():
    from risk.pmcc.calibrator import RuleSurvivalCalibrator

    model = RuleSurvivalCalibrator(evidence_tier="synthetic_research").fit(
        _training_windows(),
        (SurvivalLabel(None, 7), SurvivalLabel(2, 7), SurvivalLabel(None, 7), SurvivalLabel(1, 7)),
    )

    assert model.metadata()["promoted"] is False


def test_optional_tcn_availability_is_boolean_and_dependency_is_deferred():
    from risk.pmcc.calibrator import OptionalTCNSurvivalModel

    assert isinstance(OptionalTCNSurvivalModel.available(), bool)
    if not OptionalTCNSurvivalModel.available():
        with pytest.raises(RuntimeError, match="PyTorch"):
            OptionalTCNSurvivalModel().predict_hazards(_window())
