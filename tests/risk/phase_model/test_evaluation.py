import warnings

import numpy as np
import pytest

from risk.phase_model.evaluation import evaluate_fall_event, evaluate_predictions, should_promote_phase_model


def test_evaluation_reports_phase_fall_prefall_and_abstention_metrics():
    records = [
        {"subject_id": "a", "dataset": "gmdcsa24", "phase_true": "normal_adl", "phase_pred": "normal_adl", "fall_true": 0, "fall_prob": .1, "prefall_true": 0, "prefall_prob": .1, "abstained": False, "duration_seconds": 3600},
        {"subject_id": "a", "dataset": "gmdcsa24", "phase_true": "descending", "phase_pred": "impact", "fall_true": 1, "fall_prob": .9, "prefall_true": 1, "prefall_prob": .8, "abstained": False, "duration_seconds": 3600, "lead_time_seconds": 2.0},
        {"subject_id": "b", "dataset": "prevfall", "phase_true": "fallen", "phase_pred": "fallen", "fall_true": 1, "fall_prob": .8, "prefall_true": 0, "prefall_prob": .2, "abstained": True, "duration_seconds": 1800},
    ]
    result = evaluate_predictions(records, threshold=.5)
    assert 0.0 <= result.metrics["phase_macro_f1"] <= 1.0
    assert result.metrics["fall_recall"] == 1.0
    assert "prefall_auprc" in result.metrics
    assert result.metrics["abstention_coverage"] == 2 / 3
    assert "a" in result.per_subject and "prevfall" in result.per_dataset


def test_single_class_auc_is_unavailable():
    result = evaluate_predictions([{"phase_true": "normal_adl", "phase_pred": "normal_adl", "fall_true": 0, "fall_prob": .1, "prefall_true": 0, "prefall_prob": .1, "subject_id": "a", "dataset": "x", "duration_seconds": 1}], threshold=.5)
    assert result.metrics["prefall_roc_auc"] == "unavailable"


def test_promotion_requires_recall_and_real_data():
    decision = should_promote_phase_model(
        {"auprc": .63, "recall": .90, "fpr_at_recall": .24, "leave_one_dataset_out": [1, 2, 3]},
        {"auprc": .60, "recall": .90, "fpr_at_recall": .33},
    )
    assert decision.promoted
    assert not should_promote_phase_model({"auprc": .9, "recall": .99, "fpr_at_recall": .1, "demo": True}, {"auprc": .1, "recall": .1, "fpr_at_recall": .9}).promoted


def test_evaluate_fall_event_accepts_numpy_arrays():
    result = evaluate_fall_event(
        np.array([1, 0], dtype=np.int64),
        np.array([0.9, 0.1], dtype=np.float64),
        np.array(["s1", "s1"], dtype=object),
        0.5,
    )

    assert result["inner_mean_f1"] == 1.0
    assert result["ece"] == pytest.approx(0.1)
    assert result["brier"] == pytest.approx(0.01)


def test_evaluate_fall_event_preserves_empty_numpy_array_metrics_without_truth_testing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = evaluate_fall_event(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=object),
            0.5,
        )

    assert result["inner_mean_f1"] == 0.0
    assert result["ece"] == 0.0
    assert result["brier"] == 0.0
    assert result["per_subject"] == {}
