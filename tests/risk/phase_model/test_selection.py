import math

import pytest

from risk.phase_model.selection import (
    apply_temperature,
    candidate_passes,
    choose_threshold,
    fit_temperature,
    metrics_at_threshold,
)


def test_metrics_at_threshold_reports_hand_checked_binary_rates():
    metrics = metrics_at_threshold([1, 1, 0, 0], [0.9, 0.4, 0.8, 0.1], 0.5)

    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.false_positive_rate == 0.5


def test_threshold_selection_enforces_recall_floor():
    records = [
        {"subject_id": "s2", "label": 1, "score": 0.70},
        {"subject_id": "s2", "label": 1, "score": 0.40},
        {"subject_id": "s3", "label": 0, "score": 0.60},
        {"subject_id": "s3", "label": 0, "score": 0.10},
    ]

    selected = choose_threshold(records, recall_floor=1.0, lower=0.20, upper=0.80, step=0.01)

    assert selected.recall == 1.0
    assert selected.threshold <= 0.40


def test_threshold_selection_prefers_subject_macro_then_worst_subject_then_lower_fpr():
    records = [
        {"subject_id": "s2", "label": 1, "score": 0.9},
        {"subject_id": "s2", "label": 0, "score": 0.7},
        {"subject_id": "s3", "label": 1, "score": 0.6},
        {"subject_id": "s3", "label": 0, "score": 0.5},
    ]

    selected = choose_threshold(records, recall_floor=0.5, lower=0.55, upper=0.75, step=0.1)

    assert selected.threshold == pytest.approx(0.55)
    assert selected.subject_macro_f1 == pytest.approx(5 / 6)
    assert selected.worst_subject_f1 == pytest.approx(2 / 3)


def test_threshold_selection_rejects_empty_or_impossible_candidate_range():
    with pytest.raises(ValueError, match="records must not be empty"):
        choose_threshold([])
    with pytest.raises(ValueError, match="no threshold satisfies recall_floor"):
        choose_threshold([{"subject_id": "s2", "label": 1, "score": 0.1}], recall_floor=1.0, lower=0.2, upper=0.8)


def test_temperature_is_positive_and_finite():
    result = fit_temperature([5.0, -5.0, 2.0, -2.0], [1, 0, 0, 1])

    assert result > 0.0
    assert math.isfinite(result)


def test_apply_temperature_scales_logits_and_rejects_invalid_temperature():
    probabilities = apply_temperature([2.0, -2.0], 2.0)

    assert probabilities == pytest.approx([0.73105858, 0.26894142])
    with pytest.raises(ValueError, match="positive and finite"):
        apply_temperature([0.0], float("nan"))


def test_candidate_passes_requires_f1_gain_recall_and_calibration_gates():
    baseline = {"inner_mean_f1": 0.70, "brier": 0.215}
    candidate = {
        "inner_mean_f1": 0.73,
        "inner_mean_recall": 0.80,
        "worst_subject_recall": 0.60,
        "confirmation_f1": 0.80,
        "confirmation_recall": 0.75,
        "ece": 0.15,
        "brier": 0.215,
    }

    assert candidate_passes(candidate, baseline)
    assert not candidate_passes({**candidate, "ece": 0.151}, baseline)
