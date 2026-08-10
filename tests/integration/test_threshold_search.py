import numpy as np
import pytest

from risk.phase_model.selection import metrics_at_threshold
from scripts.optimize_phase_threshold import _choose_threshold, inner_threshold_search


def _synthetic_logits(
    *,
    n_subjects: int = 6,
    samples_per_subject: int = 20,
    fall_ratio: float = 0.3,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    logits = []
    labels = []
    subjects = []
    for subject_index in range(n_subjects):
        subject_id = f"subject-{subject_index}"
        n_falls = max(1, int(samples_per_subject * fall_ratio))
        n_adls = samples_per_subject - n_falls
        fall_logits = rng.normal(loc=1.5, scale=0.8, size=n_falls)
        adl_logits = rng.normal(loc=-1.5, scale=0.8, size=n_adls)
        logits.extend(fall_logits.tolist())
        logits.extend(adl_logits.tolist())
        labels.extend([1] * n_falls)
        labels.extend([0] * n_adls)
        subjects.extend([subject_id] * samples_per_subject)
    return np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.int64), np.asarray(subjects, dtype=object)


def test_inner_search_returns_threshold_between_zero_and_one():
    logits, labels, subjects = _synthetic_logits()
    result = inner_threshold_search(logits, labels, subjects, n_folds=3, seed=42)
    assert 0.0 <= result["threshold"] <= 1.0
    assert len(result["fold_thresholds"]) == 3
    assert all(0.0 <= threshold <= 1.0 for threshold in result["fold_thresholds"])


def test_inner_search_aggregates_median():
    logits, labels, subjects = _synthetic_logits()
    result = inner_threshold_search(
        logits, labels, subjects, n_folds=3, aggregation="median", seed=42
    )
    assert result["aggregation"] == "median"
    assert 0.0 <= result["threshold"] <= 1.0


def test_inner_search_rejects_mismatched_lengths():
    logits = np.zeros(10, dtype=np.float64)
    labels = np.zeros(10, dtype=np.int64)
    subjects = np.array(["s"] * 9, dtype=object)
    with pytest.raises(ValueError, match="same length"):
        inner_threshold_search(logits, labels, subjects)


def test_inner_search_rejects_single_fold():
    logits, labels, subjects = _synthetic_logits()
    with pytest.raises(ValueError, match="n_folds"):
        inner_threshold_search(logits, labels, subjects, n_folds=1)


def test_inner_search_rejects_empty_inputs():
    with pytest.raises(ValueError, match="empty"):
        inner_threshold_search([], [], [])


def test_selection_accepts_numpy_sequences_without_truth_value_errors():
    metrics = metrics_at_threshold(
        np.array([1, 0], dtype=np.int64),
        np.array([0.9, 0.1], dtype=np.float64),
        0.5,
    )

    assert metrics.true_positive == 1
    assert metrics.true_negative == 1


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({"recall_floor": float("nan")}, "recall_floor"),
        ({"recall_floor": float("inf")}, "recall_floor"),
        ({"recall_floor": -0.01}, "recall_floor"),
        ({"recall_floor": 1.01}, "recall_floor"),
        ({"recall_floor": "invalid"}, "recall_floor"),
        ({"min_recall_per_subject": float("nan")}, "min_recall_per_subject"),
        ({"min_recall_per_subject": float("-inf")}, "min_recall_per_subject"),
        ({"min_recall_per_subject": -0.01}, "min_recall_per_subject"),
        ({"min_recall_per_subject": 1.01}, "min_recall_per_subject"),
        ({"fpr_ceiling": float("nan")}, "fpr_ceiling"),
        ({"fpr_ceiling": float("inf")}, "fpr_ceiling"),
        ({"fpr_ceiling": -0.01}, "fpr_ceiling"),
        ({"fpr_ceiling": 1.01}, "fpr_ceiling"),
    ],
)
def test_inner_search_rejects_invalid_probability_constraints(kwargs, parameter):
    with pytest.raises(ValueError, match=parameter):
        inner_threshold_search(
            [2.0, -2.0, 1.0, -1.0],
            [1, 0, 1, 0],
            ["s1", "s1", "s2", "s2"],
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({"lower": float("nan")}, "lower"),
        ({"lower": float("-inf")}, "lower"),
        ({"lower": -0.01}, "lower"),
        ({"lower": 1.01}, "lower"),
        ({"upper": float("nan")}, "upper"),
        ({"upper": float("inf")}, "upper"),
        ({"upper": -0.01}, "upper"),
        ({"upper": 1.01}, "upper"),
    ],
)
def test_choose_threshold_rejects_invalid_probability_bounds(kwargs, parameter):
    with pytest.raises(ValueError, match=parameter):
        _choose_threshold(
            [0.9, 0.1],
            [1, 0],
            ["s1", "s2"],
            **kwargs,
        )
