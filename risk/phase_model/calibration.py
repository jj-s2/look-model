"""Validation-fold temperature scaling and calibration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _brier(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    return sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(labels)


def _ece(probabilities: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [i for i, probability in enumerate(probabilities) if lower <= probability < upper or (index == bins - 1 and probability == upper)]
        if not selected:
            continue
        confidence = sum(probabilities[i] for i in selected) / len(selected)
        accuracy = sum(labels[i] for i in selected) / len(selected)
        error += len(selected) / len(labels) * abs(confidence - accuracy)
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
        labels = [int(value) for value in labels]
        logits = [float(value) for value in logits]
        best_temperature, best_loss = 1.0, float("inf")
        for step in range(1, 201):
            temperature = 0.05 * (200 ** (step / 200))
            loss = 0.0
            for logit, label in zip(logits, labels):
                probability = min(1 - 1e-7, max(1e-7, _sigmoid(logit / temperature)))
                loss -= label * math.log(probability) + (1 - label) * math.log(1 - probability)
            if loss < best_loss:
                best_temperature, best_loss = temperature, loss
        self.temperature = best_temperature
        before = [_sigmoid(value) for value in logits]
        after = [_sigmoid(value / best_temperature) for value in logits]
        metrics = {
            "brier_before": _brier(before, labels), "brier_after": _brier(after, labels),
            "ece_before": _ece(before, labels), "ece_after": _ece(after, labels),
        }
        if split_hash is not None:
            metrics["split_hash"] = split_hash  # type: ignore[assignment]
        return CalibrationResult(best_temperature, len(labels), metrics)
