import numpy as np
import pytest

from scripts.optimize_phase_threshold import inner_threshold_search


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
