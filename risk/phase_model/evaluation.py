"""Dependency-light evaluation and promotion gates for phase-risk releases."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, Mapping


@dataclass(frozen=True)
class PhaseEvaluation:
    metrics: Mapping[str, object]
    per_subject: Mapping[str, Mapping[str, object]]
    per_dataset: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reasons: tuple[str, ...]


def _binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    tp = sum(label == pred == 1 for label, pred in zip(labels, predictions))
    fp = sum(label == 0 and pred == 1 for label, pred in zip(labels, predictions))
    fn = sum(label == 1 and pred == 0 for label, pred in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _roc_auc(labels: list[int], scores: list[float]) -> float | str:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return "unavailable"
    order = sorted(range(len(labels)), key=lambda index: scores[index])
    rank_sum = sum(rank for rank, index in enumerate(order, start=1) if labels[index] == 1)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _auprc(labels: list[int], scores: list[float]) -> float | str:
    if not any(labels) or all(labels):
        return "unavailable"
    order = sorted(range(len(labels)), key=lambda index: scores[index], reverse=True)
    total_positive = sum(labels)
    tp = fp = 0
    previous_recall = 0.0
    area = 0.0
    for index in order:
        if labels[index]:
            tp += 1
        else:
            fp += 1
        recall = tp / total_positive
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def _fpr_at_recall(labels: list[int], scores: list[float], target: float = .9) -> float | str:
    if not any(labels) or all(labels):
        return "unavailable"
    thresholds = sorted(set(scores), reverse=True)
    candidates = []
    for threshold in thresholds:
        metrics = _binary_metrics(labels, [int(score >= threshold) for score in scores])
        if metrics["recall"] >= target:
            negatives = len(labels) - sum(labels)
            fp = sum(label == 0 and score >= threshold for label, score in zip(labels, scores))
            candidates.append(fp / negatives if negatives else 0.0)
    return min(candidates) if candidates else "unavailable"


def _expected_calibration_error(labels: list[int], scores: list[float], bins: int = 10) -> float:
    """Compute expected calibration error with equal-width bins."""
    if len(labels) == 0:
        return 0.0
    edges = [index / bins for index in range(bins + 1)]
    total_error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = [lower <= score < upper or (upper == 1.0 and score == 1.0) for score in scores]
        bin_labels = [label for label, include in zip(labels, mask) if include]
        bin_scores = [score for score, include in zip(scores, mask) if include]
        if not bin_labels:
            continue
        mean_predicted = sum(bin_scores) / len(bin_scores)
        fraction_positive = sum(bin_labels) / len(bin_labels)
        total_error += len(bin_labels) * abs(mean_predicted - fraction_positive)
    return total_error / len(labels)


def _brier_score(labels: list[int], scores: list[float]) -> float:
    """Compute mean squared error between probabilities and binary labels."""
    if len(labels) == 0:
        return 0.0
    return sum((float(score) - float(label)) ** 2 for score, label in zip(scores, labels)) / len(labels)


def _evaluate_group(records: list[Mapping[str, object]], threshold: float) -> dict[str, object]:
    phase_labels = [str(record.get("phase_true", "unknown")) for record in records]
    phase_predictions = [str(record.get("phase_pred", "unknown")) for record in records]
    classes = sorted(set(phase_labels) | set(phase_predictions))
    f1_values = []
    for label in classes:
        binary_true = [int(value == label) for value in phase_labels]
        binary_pred = [int(value == label) for value in phase_predictions]
        f1_values.append(_binary_metrics(binary_true, binary_pred)["f1"])
    fall_true = [int(record.get("fall_true", 0)) for record in records]
    fall_scores = [float(record.get("fall_prob", 0.0)) for record in records]
    fall = _binary_metrics(fall_true, [int(score >= threshold) for score in fall_scores])
    prefall_true = [int(record.get("prefall_true", 0)) for record in records]
    prefall_scores = [float(record.get("prefall_prob", 0.0)) for record in records]
    duration = sum(float(record.get("duration_seconds", 0.0)) for record in records)
    false_alarms = sum(label == 0 and score >= threshold for label, score in zip(fall_true, fall_scores))
    leads = [float(record["lead_time_seconds"]) for record in records if record.get("lead_time_seconds") is not None and float(record["lead_time_seconds"]) >= 0]
    return {
        "phase_macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "fall_precision": fall["precision"], "fall_recall": fall["recall"], "fall_f1": fall["f1"],
        "prefall_auprc": _auprc(prefall_true, prefall_scores),
        "prefall_roc_auc": _roc_auc(prefall_true, prefall_scores),
        "fpr_at_recall_90": _fpr_at_recall(prefall_true, prefall_scores),
        "lead_time_seconds_median": median(leads) if leads else "unavailable",
        "false_alarms_per_hour": false_alarms / (duration / 3600.0) if duration > 0 else "unavailable",
        "abstention_coverage": sum(not bool(record.get("abstained", False)) for record in records) / len(records) if records else 0.0,
        "sample_count": len(records),
    }


def evaluate_predictions(records: Iterable[Mapping[str, object]], threshold: float = .5) -> PhaseEvaluation:
    materialized = list(records)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    metrics = _evaluate_group(materialized, threshold)
    subjects: dict[str, list[Mapping[str, object]]] = {}
    datasets: dict[str, list[Mapping[str, object]]] = {}
    for record in materialized:
        subjects.setdefault(str(record.get("subject_id", "unknown")), []).append(record)
        datasets.setdefault(str(record.get("dataset", "unknown")), []).append(record)
    return PhaseEvaluation(
        metrics=metrics,
        per_subject={key: _evaluate_group(value, threshold) for key, value in sorted(subjects.items())},
        per_dataset={key: _evaluate_group(value, threshold) for key, value in sorted(datasets.items())},
    )


def evaluate_fall_event(
    labels: list[int],
    probabilities: list[float],
    subjects: list[str],
    threshold: float,
) -> dict[str, object]:
    """Return fall-event metrics needed by the F1 promotion gate."""
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    overall = _binary_metrics(labels, predictions)

    per_subject: dict[str, dict[str, object]] = {}
    subject_ids = sorted(set(subjects))
    for subject in subject_ids:
        mask = [index for index, subject_id in enumerate(subjects) if subject_id == subject]
        subject_labels = [labels[index] for index in mask]
        subject_probs = [probabilities[index] for index in mask]
        subject_preds = [predictions[index] for index in mask]
        metrics = _binary_metrics(subject_labels, subject_preds)
        per_subject[subject] = {
            **metrics,
            "samples": len(mask),
            "ece": _expected_calibration_error(subject_labels, subject_probs),
            "brier": _brier_score(subject_labels, subject_probs),
        }

    subject_recalls = [float(per_subject[subject]["recall"]) for subject in subject_ids]
    subject_f1s = [float(per_subject[subject]["f1"]) for subject in subject_ids]
    return {
        "inner_mean_f1": overall["f1"],
        "inner_mean_recall": overall["recall"],
        "inner_mean_precision": overall["precision"],
        "worst_subject_recall": min(subject_recalls) if subject_recalls else 0.0,
        "subject_macro_f1": sum(subject_f1s) / len(subject_f1s) if subject_f1s else 0.0,
        "ece": _expected_calibration_error(labels, probabilities),
        "brier": _brier_score(labels, probabilities),
        "per_subject": per_subject,
    }


def should_promote_phase_model(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> PromotionDecision:
    reasons: list[str] = []
    if bool(candidate.get("demo")) or bool(candidate.get("simulated_young_subjects")):
        return PromotionDecision(False, ("demo or simulated results cannot be promoted",))
    recall = float(candidate.get("recall", candidate.get("fall_recall", 0.0)))
    baseline_recall = float(baseline.get("recall", baseline.get("fall_recall", 0.0)))
    if recall < baseline_recall:
        reasons.append("recall is lower than baseline")
    candidate_auprc = float(candidate.get("auprc", candidate.get("prefall_auprc", 0.0)))
    baseline_auprc = float(baseline.get("auprc", baseline.get("prefall_auprc", 0.0)))
    candidate_fpr = float(candidate.get("fpr_at_recall", candidate.get("fpr_at_recall_90", 1.0)))
    baseline_fpr = float(baseline.get("fpr_at_recall", baseline.get("fpr_at_recall_90", 1.0)))
    improved_auprc = baseline_auprc > 0 and candidate_auprc >= baseline_auprc * 1.05
    improved_fpr = baseline_fpr > 0 and candidate_fpr <= baseline_fpr * 0.80
    if not (improved_auprc or improved_fpr):
        reasons.append("neither AUPRC nor equal-recall FPR meets the improvement gate")
    folds = candidate.get("leave_one_dataset_out")
    if not isinstance(folds, (list, tuple)) or len(folds) < 3:
        reasons.append("at least three leave-one-dataset-out results are required")
    promoted = not reasons
    if promoted:
        reasons.append("candidate meets recall, improvement, and cross-dataset gates")
    return PromotionDecision(promoted, tuple(reasons))
