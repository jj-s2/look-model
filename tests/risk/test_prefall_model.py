import pytest

from risk.prefall_model import PrefallModel


class SchemaArray:
    """Small named matrix that sklearn can consume without pandas."""

    def __init__(self, rows, columns=("sway", "step_width")):
        self.rows = [list(row) for row in rows]
        self.columns = tuple(columns)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)

    def __array__(self, dtype=None):
        import numpy as np
        return np.asarray(self.rows, dtype=dtype)

    def take_rows(self, indices):
        return SchemaArray([self.rows[index] for index in indices], self.columns)


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


def test_model_fits_real_sklearn_pipeline_and_returns_positive_probabilities():
    pytest.importorskip("sklearn")
    features = SchemaArray([[0.1, 0.2], [0.2, 0.3], [0.8, 0.7], [0.9, 0.8]])

    model = PrefallModel(random_seed=7).fit(features, [0, 0, 1, 1])

    probabilities = model.predict_proba(features)
    assert probabilities.shape == (4,)
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert model._estimator.named_steps["classifier"].class_weight == "balanced"


def test_model_can_fit_extra_trees_for_non_linear_prefall_features():
    pytest.importorskip("sklearn")
    features = SchemaArray([[0.1, 0.2], [0.2, 0.3], [0.8, 0.7], [0.9, 0.8]])

    model = PrefallModel(random_seed=7, estimator_name="extra_trees").fit(features, [0, 0, 1, 1])

    assert model.predict_proba(features).shape == (4,)
    assert model.metadata()["estimator_name"] == "extra_trees"
