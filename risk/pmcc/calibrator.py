"""CPU survival calibration with an explicitly optional TCN extension.

The default model is a deterministic, horizon-specific logistic calibrator.
It only imports scikit-learn while fitting, so schema validation and offline
edge code remain usable in installations without ML training extras.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from .features import FeatureWindow
from .survival import SurvivalLabel


class SurvivalRiskModel(Protocol):
    """A model which emits conditional hazards for the next seven days."""

    def predict_hazards(self, features: FeatureWindow) -> Sequence[float]: ...


@dataclass
class RuleSurvivalCalibrator:
    """A reproducible CPU calibrator for seven discrete survival horizons.

    Each ``FeatureWindow`` is reduced to a quality-weighted value, observed
    coverage, and quality summary per original feature.  The latter two
    summaries make missing data explicit instead of interpreting a missing
    value as a normal zero.  One logistic model is fitted for each horizon;
    rows censored before that horizon are excluded to avoid using future or
    unobserved outcome information.  ``class_weight='balanced'`` corrects
    class imbalance, while every supplied window carries unit subject-row
    weight.  Callers must enforce subject-disjoint evaluation splits.
    """

    random_seed: int = 42
    evidence_tier: str = "real_device_longitudinal"
    feature_names: tuple[str, ...] = field(default_factory=tuple, init=False)
    _estimators: tuple[Any, ...] = field(default_factory=tuple, init=False, repr=False)

    def fit(
        self,
        windows: Sequence[FeatureWindow],
        labels: Sequence[SurvivalLabel],
    ) -> "RuleSurvivalCalibrator":
        records = tuple(windows)
        outcomes = tuple(labels)
        if not records:
            raise ValueError("windows must not be empty")
        if len(records) != len(outcomes):
            raise ValueError("windows and labels must have the same length")
        if not all(isinstance(window, FeatureWindow) for window in records):
            raise ValueError("windows must contain FeatureWindow records")
        if not all(isinstance(label, SurvivalLabel) for label in outcomes):
            raise ValueError("labels must contain SurvivalLabel records")
        schema = records[0].feature_names
        if any(window.feature_names != schema for window in records):
            raise ValueError("all training windows must use the same feature schema")
        if not any(label.event_day is not None for label in outcomes):
            raise ValueError("labels must contain both event and non-event examples")
        if not any(label.event_day is None for label in outcomes):
            raise ValueError("labels must contain both event and non-event examples")

        matrix = np.vstack([_summarize_window(window) for window in records])
        estimators: list[Any] = []
        for horizon in range(1, 8):
            indices = [index for index, label in enumerate(outcomes) if _observed_at_horizon(label, horizon)]
            targets = np.asarray([_event_by_horizon(outcomes[index], horizon) for index in indices], dtype=int)
            if len(np.unique(targets)) != 2:
                raise ValueError(f"labels must contain both event and non-event examples at horizon {horizon}")
            estimator = _make_logistic_estimator(self.random_seed)
            estimator.fit(matrix[indices], targets)
            estimators.append(estimator)
        self.feature_names = schema
        self._estimators = tuple(estimators)
        return self

    def predict_hazards(self, features: FeatureWindow) -> tuple[float, ...]:
        if not self._estimators:
            raise RuntimeError("RuleSurvivalCalibrator must be fitted before prediction")
        if features.feature_names != self.feature_names:
            raise ValueError(
                "feature schema mismatch: expected "
                f"{list(self.feature_names)!r}, received {list(features.feature_names)!r}"
            )
        vector = _summarize_window(features).reshape(1, -1)
        cumulative = np.asarray([_positive_probability(model, vector) for model in self._estimators])
        cumulative = np.maximum.accumulate(np.clip(cumulative, 0.0, 1.0))
        hazards = np.empty(7, dtype=float)
        previous_survival = 1.0
        for index, risk in enumerate(cumulative):
            hazards[index] = 0.0 if previous_survival <= 0.0 else (risk - (1.0 - previous_survival)) / previous_survival
            hazards[index] = float(np.clip(hazards[index], 0.0, 1.0))
            previous_survival *= 1.0 - hazards[index]
        return tuple(float(value) for value in hazards)

    def metadata(self) -> dict[str, object]:
        return {
            "model_type": "quality_weighted_one_vs_horizon_logistic",
            "feature_schema": list(self.feature_names),
            "random_seed": self.random_seed,
            "class_weight": "balanced",
            "subject_weighting": "one window per subject-equivalent training row",
            "evidence_tier": self.evidence_tier,
            "promoted": self.evidence_tier not in {"synthetic_research", "offline_fixture"},
            "tcn_evaluation_claim": None,
        }


class OptionalTCNSurvivalModel:
    """Optional placeholder that never imports PyTorch during normal import."""

    @classmethod
    def available(cls) -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("torch") is not None
        except (ImportError, ValueError):
            return False

    def fit(self, windows: Sequence[FeatureWindow], labels: Sequence[SurvivalLabel]) -> "OptionalTCNSurvivalModel":
        self._require_torch()
        raise NotImplementedError("Optional TCN training requires a separately evaluated longitudinal implementation")

    def predict_hazards(self, features: FeatureWindow) -> tuple[float, ...]:
        self._require_torch()
        raise NotImplementedError("Optional TCN inference requires a separately evaluated longitudinal implementation")

    @classmethod
    def _require_torch(cls) -> None:
        if not cls.available():
            raise RuntimeError("PyTorch is optional and is required only for OptionalTCNSurvivalModel")


def _make_logistic_estimator(random_seed: int) -> Any:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        # A small NumPy fallback keeps the local/offline demo usable when the
        # optional training package is not installed.  It has the same
        # balanced logistic objective, but is deliberately not a TCN or a
        # performance claim.
        return _NumpyBalancedLogistic(random_seed=random_seed)
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            random_state=random_seed,
            solver="liblinear",
            max_iter=1000,
        ),
    )


def _positive_probability(model: Any, vector: np.ndarray) -> float:
    probabilities = model.predict_proba(vector)
    return float(probabilities[0, 1])


@dataclass
class _NumpyBalancedLogistic:
    """Minimal deterministic balanced logistic regression fallback."""

    random_seed: int
    _mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _scale: np.ndarray | None = field(default=None, init=False, repr=False)
    _weights: np.ndarray | None = field(default=None, init=False, repr=False)
    _intercept: float = field(default=0.0, init=False, repr=False)

    def fit(self, values: np.ndarray, labels: np.ndarray) -> "_NumpyBalancedLogistic":
        self._mean = values.mean(axis=0)
        self._scale = values.std(axis=0)
        self._scale = np.where(self._scale > 1e-12, self._scale, 1.0)
        matrix = (values - self._mean) / self._scale
        self._weights = np.zeros(matrix.shape[1], dtype=float)
        self._intercept = 0.0
        positives = max(int(labels.sum()), 1)
        negatives = max(len(labels) - positives, 1)
        sample_weight = np.where(labels == 1, len(labels) / (2.0 * positives), len(labels) / (2.0 * negatives))
        for _ in range(1_000):
            logits = np.clip(matrix @ self._weights + self._intercept, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            residual = (probabilities - labels) * sample_weight
            self._weights -= 0.05 * ((matrix.T @ residual) / len(labels) + 1e-4 * self._weights)
            self._intercept -= 0.05 * float(residual.mean())
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self._mean is None or self._scale is None or self._weights is None:
            raise RuntimeError("logistic estimator must be fitted before prediction")
        matrix = (values - self._mean) / self._scale
        logits = np.clip(matrix @ self._weights + self._intercept, -30.0, 30.0)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack((1.0 - positive, positive))


def _summarize_window(window: FeatureWindow) -> np.ndarray:
    values = np.asarray(window.values, dtype=float)
    missing = np.asarray(window.missing_mask, dtype=bool)
    quality = np.asarray(window.quality, dtype=float)
    observed = ~missing
    weights = np.where(observed, quality, 0.0)
    denominators = weights.sum(axis=0)
    weighted_sum = np.where(observed, values * weights, 0.0).sum(axis=0)
    weighted_mean = np.divide(weighted_sum, denominators, out=np.zeros_like(denominators), where=denominators > 0)
    coverage = observed.mean(axis=0)
    mean_quality = np.where(observed, quality, 0.0).mean(axis=0)
    return np.concatenate((weighted_mean, coverage, mean_quality))


def _observed_at_horizon(label: SurvivalLabel, horizon: int) -> bool:
    return label.event_day is not None or label.censor_day >= horizon


def _event_by_horizon(label: SurvivalLabel, horizon: int) -> int:
    return int(label.event_day is not None and label.event_day <= horizon)
