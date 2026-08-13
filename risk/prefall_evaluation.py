"""Leakage-safe subject-level evaluation for the pre-fall model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from risk.prefall_model import PrefallModel


ACCEPTANCE_F1 = 0.90
ACCEPTANCE_RECALL = 0.88


@dataclass(frozen=True)
class EvaluationReport:
    strategy: str
    n_subjects: int
    fold_metrics: tuple[dict[str, float], ...]
    metrics: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_subject_folds(
    subject_ids: Sequence[Any], n_splits: int = 5, *, leave_one_subject_out: bool = False
) -> list[tuple[list[int], list[int]]]:
    """Return group-isolated indices, using sklearn splitters when available."""
    groups = list(subject_ids)
    unique = list(dict.fromkeys(groups))
    if len(unique) < 2:
        raise ValueError("subject-wise evaluation requires at least two distinct subject IDs")
    if leave_one_subject_out:
        return [
            ([index for index, value in enumerate(groups) if value != subject],
             [index for index, value in enumerate(groups) if value == subject])
            for subject in unique
        ]
    effective_splits = min(max(int(n_splits), 2), len(unique))
    try:
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=effective_splits)
        return [(train.tolist(), valid.tolist()) for train, valid in splitter.split(groups, groups, groups)]
    except ImportError:
        # Equivalent deterministic group-only fallback for environments without sklearn.
        buckets = [[] for _ in range(effective_splits)]
        for index, subject in enumerate(unique):
            buckets[index % effective_splits].append(subject)
        return [
            ([index for index, value in enumerate(groups) if value not in held_out],
             [index for index, value in enumerate(groups) if value in held_out])
            for held_out in buckets
        ]


def evaluate_subject_wise(
    features: Any, labels: Any, subject_ids: Sequence[Any], *, n_splits: int = 5,
    threshold: float = 0.5, random_seed: int = 42, leave_one_subject_out: bool = False,
    estimator_name: str = "logistic_regression",
) -> EvaluationReport:
    """Fit a fresh model per group-isolated fold and aggregate validation metrics."""
    if len(features) != len(labels) or len(labels) != len(subject_ids):
        raise ValueError("features, labels, and subject_ids must have the same number of rows")
    folds = make_subject_folds(subject_ids, n_splits, leave_one_subject_out=leave_one_subject_out)
    metrics = []
    for train_idx, valid_idx in folds:
        model = PrefallModel(
            random_seed=random_seed, threshold=threshold, estimator_name=estimator_name,
        )
        model.fit(_take_rows(features, train_idx), _take_rows(labels, train_idx))
        predictions = model.predict(_take_rows(features, valid_idx))
        metrics.append(_binary_metrics(_take_rows(labels, valid_idx), predictions))
    aggregate = {
        name: sum(fold[name] for fold in metrics) / len(metrics)
        for name in ("precision", "recall", "f1")
    }
    return EvaluationReport(
        strategy="LeaveOneGroupOut" if leave_one_subject_out else "GroupKFold",
        n_subjects=len(set(subject_ids)), fold_metrics=tuple(metrics), metrics=aggregate,
    )


def should_promote(candidate: dict[str, float], baseline: dict[str, float]) -> bool:
    """Promote only validation candidates that beat baseline and acceptance gates."""
    return (
        candidate.get("f1", 0.0) >= baseline.get("f1", 0.0)
        and candidate.get("recall", 0.0) >= baseline.get("recall", 0.0)
        and candidate.get("f1", 0.0) >= ACCEPTANCE_F1
        and candidate.get("recall", 0.0) >= ACCEPTANCE_RECALL
    )


def _take_rows(values: Any, indices: Iterable[int]) -> Any:
    index_list = list(indices)
    if hasattr(values, "iloc"):
        return values.iloc[index_list]
    if hasattr(values, "take_rows"):
        return values.take_rows(index_list)
    return [values[index] for index in index_list]


def _binary_metrics(labels: Iterable[Any], predictions: Iterable[Any]) -> dict[str, float]:
    pairs = [(int(label) == 1, bool(prediction)) for label, prediction in zip(labels, predictions)]
    tp = sum(1 for label, prediction in pairs if label and prediction)
    fp = sum(1 for label, prediction in pairs if not label and prediction)
    fn = sum(1 for label, prediction in pairs if label and not prediction)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
