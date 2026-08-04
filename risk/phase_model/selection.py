"""Subject-safe binary model selection and validation calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch.nn import functional as functional


@dataclass(frozen=True)
class BinaryMetrics:
    """Binary classification metrics at one fixed decision threshold."""

    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


@dataclass(frozen=True)
class ThresholdSelection:
    """The validation threshold selected without mixing subject identities."""

    threshold: float
    macro_f1: float
    precision: float
    recall: float
    subject_macro_f1: float
    worst_subject_f1: float
    false_positive_rate: float
    f1: float


def _validated_binary_inputs(labels: Sequence[int], scores: Sequence[float]) -> tuple[list[int], list[float]]:
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and have equal length")
    normalized_labels = [int(value) for value in labels]
    if any(value not in (0, 1) for value in normalized_labels):
        raise ValueError("labels must contain only 0 or 1")
    normalized_scores = [float(value) for value in scores]
    if any(not math.isfinite(value) for value in normalized_scores):
        raise ValueError("scores must be finite")
    return normalized_labels, normalized_scores


def _validate_threshold(threshold: float) -> float:
    normalized = float(threshold)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    return normalized


def metrics_at_threshold(labels: Sequence[int], scores: Sequence[float], threshold: float) -> BinaryMetrics:
    """Compute hand-auditable binary metrics using ``score >= threshold``."""

    normalized_labels, normalized_scores = _validated_binary_inputs(labels, scores)
    threshold = _validate_threshold(threshold)
    predictions = [int(score >= threshold) for score in normalized_scores]
    true_positive = sum(label == prediction == 1 for label, prediction in zip(normalized_labels, predictions))
    false_positive = sum(label == 0 and prediction == 1 for label, prediction in zip(normalized_labels, predictions))
    false_negative = sum(label == 1 and prediction == 0 for label, prediction in zip(normalized_labels, predictions))
    true_negative = sum(label == prediction == 0 for label, prediction in zip(normalized_labels, predictions))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0
    return BinaryMetrics(precision, recall, f1, false_positive_rate, true_positive, false_positive, false_negative, true_negative)


def _validated_records(records: Sequence[Mapping[str, object]]) -> tuple[list[str], list[int], list[float]]:
    if not records:
        raise ValueError("records must not be empty")
    subjects: list[str] = []
    labels: list[int] = []
    scores: list[float] = []
    for record in records:
        if "subject_id" not in record or "label" not in record or "score" not in record:
            raise ValueError("each record requires subject_id, label, and score")
        subject = str(record["subject_id"]).strip()
        if not subject:
            raise ValueError("subject_id must not be empty")
        subjects.append(subject)
        labels.append(int(record["label"]))
        scores.append(float(record["score"]))
    _validated_binary_inputs(labels, scores)
    return subjects, labels, scores


def choose_threshold(
    records: Sequence[Mapping[str, object]],
    recall_floor: float = 0.75,
    *,
    lower: float = 0.20,
    upper: float = 0.80,
    step: float = 0.01,
) -> ThresholdSelection:
    """Select an inner-validation threshold while preserving the recall floor.

    Candidate thresholds are ranked by subject-macro F1, worst subject F1, then
    lower false-positive rate.  This intentionally avoids selecting a threshold
    whose aggregate score hides a poor-performing held-out subject.
    """

    subjects, labels, scores = _validated_records(records)
    recall_floor = float(recall_floor)
    lower, upper, step = float(lower), float(upper), float(step)
    if not math.isfinite(recall_floor) or not 0.0 <= recall_floor <= 1.0:
        raise ValueError("recall_floor must be finite and in [0, 1]")
    _validate_threshold(lower)
    _validate_threshold(upper)
    if lower > upper:
        raise ValueError("lower must not exceed upper")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("step must be positive and finite")

    grouped: dict[str, tuple[list[int], list[float]]] = {}
    for subject, label, score in zip(subjects, labels, scores):
        group_labels, group_scores = grouped.setdefault(subject, ([], []))
        group_labels.append(label)
        group_scores.append(score)

    candidates: list[ThresholdSelection] = []
    candidate_count = int(math.floor((upper - lower) / step + 1e-10)) + 1
    for index in range(candidate_count):
        threshold = round(lower + index * step, 12)
        if threshold > upper + 1e-12:
            continue
        aggregate = metrics_at_threshold(labels, scores, threshold)
        if aggregate.recall + 1e-12 < recall_floor:
            continue
        subject_metrics = [metrics_at_threshold(group_labels, group_scores, threshold) for group_labels, group_scores in grouped.values()]
        subject_macro_f1 = sum(metric.f1 for metric in subject_metrics) / len(subject_metrics)
        candidates.append(
            ThresholdSelection(
                threshold=threshold,
                macro_f1=subject_macro_f1,
                precision=aggregate.precision,
                recall=aggregate.recall,
                subject_macro_f1=subject_macro_f1,
                worst_subject_f1=min(metric.f1 for metric in subject_metrics),
                false_positive_rate=aggregate.false_positive_rate,
                f1=aggregate.f1,
            )
        )
    if not candidates:
        raise ValueError("no threshold satisfies recall_floor")
    return max(candidates, key=lambda candidate: (candidate.subject_macro_f1, candidate.worst_subject_f1, -candidate.false_positive_rate))


def apply_temperature(logits: Sequence[float], temperature: float) -> list[float]:
    """Convert logits into calibrated probabilities with a positive temperature."""

    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    normalized_logits = [float(value) for value in logits]
    if any(not math.isfinite(value) for value in normalized_logits):
        raise ValueError("logits must be finite")
    probabilities: list[float] = []
    for value in normalized_logits:
        scaled = value / temperature
        if scaled >= 0.0:
            probabilities.append(1.0 / (1.0 + math.exp(-scaled)))
        else:
            exponent = math.exp(scaled)
            probabilities.append(exponent / (1.0 + exponent))
    return probabilities


def fit_temperature(logits: Sequence[float], labels: Sequence[int]) -> float:
    """Fit one validation-only temperature with LBFGS over log-temperature."""

    normalized_labels, normalized_logits = _validated_binary_inputs(labels, logits)
    logits_tensor = torch.tensor(normalized_logits, dtype=torch.float64)
    labels_tensor = torch.tensor(normalized_labels, dtype=torch.float64)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=1e-4, max=1e4)
        loss = functional.binary_cross_entropy_with_logits(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = math.exp(float(log_temperature.detach().item()))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature optimization produced a non-finite value")
    return temperature


def _finite_metric(values: Mapping[str, object], *names: str) -> float | None:
    for name in names:
        if name in values:
            try:
                value = float(values[name])
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None
    return None


def candidate_passes(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> bool:
    """Return whether a candidate meets the pre-registered F1 promotion gates."""

    baseline_f1 = _finite_metric(baseline, "inner_mean_f1", "mean_f1", "f1")
    candidate_f1 = _finite_metric(candidate, "inner_mean_f1", "mean_f1", "f1")
    inner_recall = _finite_metric(candidate, "inner_mean_recall", "mean_recall", "recall")
    worst_recall = _finite_metric(candidate, "worst_subject_recall")
    confirmation_f1 = _finite_metric(candidate, "confirmation_f1")
    confirmation_recall = _finite_metric(candidate, "confirmation_recall")
    ece = _finite_metric(candidate, "ece")
    brier = _finite_metric(candidate, "brier")
    baseline_brier = _finite_metric(baseline, "brier")
    if None in (baseline_f1, candidate_f1, inner_recall, worst_recall, confirmation_f1, confirmation_recall, ece, brier):
        return False
    brier_limit = baseline_brier if baseline_brier is not None else 0.215
    return bool(
        candidate_f1 + 1e-12 >= baseline_f1 + 0.03
        and inner_recall + 1e-12 >= 0.75
        and worst_recall + 1e-12 >= 0.60
        and confirmation_f1 + 1e-12 >= 0.80
        and confirmation_recall + 1e-12 >= 0.75
        and ece <= 0.15 + 1e-12
        and brier <= brier_limit + 1e-12
    )
