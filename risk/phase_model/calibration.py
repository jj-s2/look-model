"""Validation-only calibration: temperature scaling and threshold selection.

This module exposes both the original ``TemperatureCalibrator`` interface
and the canonical ``Calibrator`` bundle used by releases and inference.
Low-level threshold helpers are kept in ``risk.phase_model.selection``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .selection import apply_temperature as _apply_temperature
from .selection import fit_temperature as _fit_temperature


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _brier(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    return sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(probabilities)


def _ece(probabilities: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [i for i, probability in enumerate(probabilities) if lower <= probability < upper or (index == bins - 1 and probability == upper)]
        if not selected:
            continue
        confidence = sum(probabilities[i] for i in selected) / len(selected)
        accuracy = sum(labels[i] for i in selected) / len(selected)
        error += len(selected) / len(probabilities) * abs(confidence - accuracy)
    return error


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    sample_count: int
    metrics: dict[str, float]


class TemperatureCalibrator:
    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, logits: Sequence[float], labels: Sequence[int], *, partition: str = "validation", split_hash: str | None = None) -> CalibrationResult:
        if partition != "validation":
            raise ValueError("temperature calibration must use the validation partition")
        if len(logits) != len(labels) or not logits:
            raise ValueError("logits and labels must be non-empty and have equal length")
        normalized_labels = [int(value) for value in labels]
        normalized_logits = [float(value) for value in logits]
        best_temperature, best_loss = 1.0, float("inf")
        for step in range(1, 201):
            temperature = 0.05 * (200 ** (step / 200))
            loss = 0.0
            for logit, label in zip(normalized_logits, normalized_labels):
                probability = min(1 - 1e-7, max(1e-7, _sigmoid(logit / temperature)))
                loss -= label * math.log(probability) + (1 - label) * math.log(1 - probability)
            if loss < best_loss:
                best_temperature, best_loss = temperature, loss
        self.temperature = best_temperature
        before = [_sigmoid(value) for value in normalized_logits]
        after = [_sigmoid(value / best_temperature) for value in normalized_logits]
        metrics: dict[str, float] = {
            "brier_before": _brier(before, normalized_labels),
            "brier_after": _brier(after, normalized_labels),
            "ece_before": _ece(before, normalized_labels),
            "ece_after": _ece(after, normalized_labels),
        }
        if split_hash is not None:
            metrics["split_hash"] = split_hash
        return CalibrationResult(best_temperature, len(normalized_labels), metrics)


def apply_temperature(logits: Sequence[float], temperature: float) -> list[float]:
    """Convert logits into calibrated probabilities with a positive temperature."""
    return _apply_temperature(logits, temperature)


def fit_temperature(logits: Sequence[float], labels: Sequence[int]) -> float:
    """Fit one validation-only temperature with LBFGS over log-temperature."""
    return _fit_temperature(logits, labels)


@dataclass(frozen=True)
class Calibrator:
    """Temperature-scaling + threshold calibration bundle for a release."""

    temperature: float = 1.0
    threshold: float = 0.5

    def __post_init__(self) -> None:
        temperature = float(self.temperature)
        threshold = float(self.threshold)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive and finite")
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be finite and in [0, 1]")

    def calibrate(self, logits: Sequence[float]) -> list[float]:
        """Return calibrated probabilities for raw logits."""
        return apply_temperature(logits, self.temperature)

    def decide(self, probabilities: Sequence[float]) -> list[int]:
        """Return binary decisions from calibrated probabilities."""
        return [1 if float(probability) >= self.threshold else 0 for probability in probabilities]

    def calibrate_and_decide(self, logits: Sequence[float]) -> tuple[list[float], list[int]]:
        """Return both calibrated probabilities and binary decisions."""
        probabilities = self.calibrate(logits)
        return probabilities, self.decide(probabilities)
