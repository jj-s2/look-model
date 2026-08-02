import pytest

from risk.prefall_model import PrefallModel


class Frame:
    def __init__(self, columns):
        self.columns = columns

    def drop(self, *, columns):
        return Frame([column for column in self.columns if column not in columns])

    def __len__(self):
        return 2


@pytest.fixture
def trained_model():
    model = PrefallModel()
    model.feature_names = ("sway", "step_width")
    model._estimator = object()
    return model


def test_model_rejects_missing_or_reordered_features(trained_model):
    with pytest.raises(ValueError, match="feature schema"):
        trained_model.predict_proba(Frame(["step_width", "sway"]))

    with pytest.raises(ValueError, match="feature schema"):
        trained_model.predict_proba(Frame(["sway"]))


def test_model_reports_missing_scikit_learn_only_when_training_requested(monkeypatch):
    model = PrefallModel()

    monkeypatch.setattr(model, "_load_training_dependencies", lambda: (_ for _ in ()).throw(
        RuntimeError("scikit-learn is required for PrefallModel.fit")
    ))

    with pytest.raises(RuntimeError, match="scikit-learn is required"):
        model.fit(Frame(["sway"]), [0, 1])
