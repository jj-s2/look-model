"""Reproducible, explicitly dependency-gated pre-fall classifier."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class PrefallModel:
    """A scaler plus balanced logistic-regression classifier.

    The module intentionally imports neither NumPy nor scikit-learn at import
    time, so edge/runtime code can still use schema checks without ML extras.
    """

    random_seed: int = 42
    threshold: float = 0.5
    estimator_name: str = "logistic_regression"
    feature_names: tuple[str, ...] = field(default_factory=tuple, init=False)
    training_summary: dict[str, Any] = field(default_factory=dict, init=False)
    _estimator: Any = field(default=None, init=False, repr=False)

    def fit(self, features: Any, labels: Any) -> "PrefallModel":
        columns = self._columns(features)
        if not columns:
            raise ValueError("feature schema must contain at least one named column")
        if len(labels) != len(features):
            raise ValueError("features and labels must have the same number of rows")
        if len(set(labels)) < 2:
            raise ValueError("labels must contain both pre-fall and non-pre-fall examples")
        Pipeline, StandardScaler, LogisticRegression, ExtraTreesClassifier = self._load_training_dependencies()
        self.feature_names = columns
        if self.estimator_name == "logistic_regression":
            classifier = LogisticRegression(class_weight="balanced", random_state=self.random_seed)
        elif self.estimator_name == "extra_trees":
            classifier = ExtraTreesClassifier(
                n_estimators=400, class_weight="balanced", random_state=self.random_seed,
                max_features="sqrt", min_samples_leaf=2, n_jobs=-1,
            )
        else:
            raise ValueError("estimator_name must be 'logistic_regression' or 'extra_trees'")
        self._estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])
        self._estimator.fit(features, labels)
        self.training_summary = {
            "rows": len(features),
            "positive_rows": sum(1 for label in labels if int(label) == 1),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        return self

    def predict_proba(self, features: Any) -> Any:
        self._validate_schema(features)
        if self._estimator is None:
            raise RuntimeError("PrefallModel must be fitted before prediction")
        probabilities = self._estimator.predict_proba(features)
        return probabilities[:, 1]

    def predict(self, features: Any) -> Any:
        return self.predict_proba(features) >= self.threshold

    def metadata(self, *, metrics: dict[str, float] | None = None, promoted: bool | None = None) -> dict[str, Any]:
        return {
            "model_type": f"StandardScaler+{type(self._estimator.named_steps['classifier']).__name__}"
            if self._estimator is not None else "unfitted",
            "estimator_name": self.estimator_name,
            "feature_schema": list(self.feature_names),
            "random_seed": self.random_seed,
            "threshold": self.threshold,
            "training_summary": self.training_summary,
            "metrics": metrics or {},
            "promoted": promoted,
        }

    @staticmethod
    def _columns(features: Any) -> tuple[str, ...]:
        columns = getattr(features, "columns", None)
        if columns is None:
            raise ValueError("feature schema requires a DataFrame with named columns")
        return tuple(str(column) for column in columns)

    def _validate_schema(self, features: Any) -> None:
        received = self._columns(features)
        if not self.feature_names or received != self.feature_names:
            raise ValueError(
                "feature schema mismatch: expected "
                f"{list(self.feature_names)!r}, received {list(received)!r}"
            )

    @staticmethod
    def _load_training_dependencies() -> tuple[Any, Any, Any, Any]:
        try:
            from sklearn.ensemble import ExtraTreesClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as error:
            raise RuntimeError(
                "scikit-learn is required for PrefallModel.fit. "
                "Install the project's ML training dependencies before training."
            ) from error
        return Pipeline, StandardScaler, LogisticRegression, ExtraTreesClassifier
