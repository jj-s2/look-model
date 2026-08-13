import numpy as np
import pytest

from configs.skeleton import padtsf_v1_f1 as config
from risk.phase_model.calibration import Calibrator
from risk.phase_model.evaluation import evaluate_fall_event
from risk.phase_model.release import build_release_bundle, write_release_bundle
from scripts.run_phase_f1_experiment import evaluate_and_promote


def _synthetic_experiment_data(*, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_subjects = 6
    samples_per_subject = 30
    validation_logits = []
    validation_labels = []
    validation_subjects = []
    test_logits = []
    test_labels = []
    test_subjects = []

    for subject_index in range(n_subjects):
        subject_id = f"subject-{subject_index}"
        n_falls = 10
        n_adls = samples_per_subject - n_falls

        val_fall = rng.normal(loc=2.0, scale=0.6, size=n_falls)
        val_adl = rng.normal(loc=-2.0, scale=0.6, size=n_adls)
        validation_logits.extend(val_fall.tolist())
        validation_logits.extend(val_adl.tolist())
        validation_labels.extend([1] * n_falls)
        validation_labels.extend([0] * n_adls)
        validation_subjects.extend([subject_id] * samples_per_subject)

        test_fall = rng.normal(loc=2.0, scale=0.6, size=n_falls)
        test_adl = rng.normal(loc=-2.0, scale=0.6, size=n_adls)
        test_logits.extend(test_fall.tolist())
        test_logits.extend(test_adl.tolist())
        test_labels.extend([1] * n_falls)
        test_labels.extend([0] * n_adls)
        test_subjects.extend([subject_id] * samples_per_subject)

    return (
        np.asarray(validation_logits, dtype=np.float64),
        np.asarray(validation_labels, dtype=np.int64),
        validation_subjects,
        np.asarray(test_logits, dtype=np.float64),
        np.asarray(test_labels, dtype=np.int64),
        test_subjects,
    )


def test_evaluate_and_promote_returns_promoted_result():
    (
        val_logits,
        val_labels,
        val_subjects,
        test_logits,
        test_labels,
        test_subjects,
    ) = _synthetic_experiment_data()

    baseline = {
        "inner_mean_f1": 0.50,
        "inner_mean_recall": 0.70,
        "worst_subject_recall": 0.55,
        "confirmation_f1": 0.50,
        "confirmation_recall": 0.70,
        "ece": 0.20,
        "brier": 0.30,
    }

    experiment = evaluate_and_promote(
        validation_logits=val_logits,
        validation_labels=val_labels,
        validation_subjects=val_subjects,
        test_logits=test_logits,
        test_labels=test_labels,
        test_subjects=test_subjects,
        baseline=baseline,
        config=config,
    )

    assert experiment["promoted"] is True
    assert 0.0 <= experiment["threshold"] <= 1.0
    assert experiment["temperature"] > 0.0
    assert experiment["candidate"]["inner_mean_recall"] >= 0.75
    assert experiment["candidate"]["worst_subject_recall"] >= 0.60
    assert experiment["candidate"]["inner_mean_f1"] >= baseline["inner_mean_f1"] + 0.03
    assert experiment["candidate"]["confirmation_f1"] >= 0.80
    assert experiment["candidate"]["confirmation_recall"] >= 0.75
    assert experiment["candidate"]["ece"] <= 0.15


def test_evaluate_and_promote_can_reject_poor_candidate():
    rng = np.random.default_rng(99)
    logits = np.concatenate([rng.normal(-1.0, 0.5, 90), rng.normal(0.5, 0.5, 90)])
    labels = np.array([0] * 90 + [1] * 90)
    subjects = [f"subject-{i // 30}" for i in range(180)]

    baseline = {
        "inner_mean_f1": 0.90,
        "inner_mean_recall": 0.90,
        "worst_subject_recall": 0.90,
        "confirmation_f1": 0.90,
        "confirmation_recall": 0.90,
        "ece": 0.01,
        "brier": 0.01,
    }

    experiment = evaluate_and_promote(
        validation_logits=logits,
        validation_labels=labels,
        validation_subjects=subjects,
        test_logits=logits,
        test_labels=labels,
        test_subjects=subjects,
        baseline=baseline,
        config=config,
    )

    assert experiment["promoted"] is False
    assert experiment["reason"] == "no threshold satisfies recall_floor"
    assert experiment["candidate"] is None


def test_evaluate_and_promote_reports_feasible_promotion_gate_rejection():
    (
        val_logits,
        val_labels,
        val_subjects,
        test_logits,
        test_labels,
        test_subjects,
    ) = _synthetic_experiment_data()
    baseline = {
        "inner_mean_f1": 1.0,
        "inner_mean_recall": 1.0,
        "worst_subject_recall": 1.0,
        "confirmation_f1": 1.0,
        "confirmation_recall": 1.0,
        "ece": 0.0,
        "brier": 0.0,
    }

    result = evaluate_and_promote(
        validation_logits=val_logits,
        validation_labels=val_labels,
        validation_subjects=val_subjects,
        test_logits=test_logits,
        test_labels=test_labels,
        test_subjects=test_subjects,
        baseline=baseline,
        config=config,
    )

    assert result["promoted"] is False
    assert result["reason"] == "promotion gate rejected candidate"
    assert result["threshold"] is not None
    assert result["candidate"] is not None


class _InvalidRecallConfig:
    INNER_RECALL_FLOOR = float("nan")


def test_evaluate_and_promote_does_not_swallow_invalid_constraints_as_infeasible():
    with pytest.raises(ValueError, match="recall_floor"):
        evaluate_and_promote(
            validation_logits=[2.0, -2.0, 1.0, -1.0],
            validation_labels=[1, 0, 1, 0],
            validation_subjects=["s1", "s1", "s2", "s2"],
            test_logits=[1.0, -1.0],
            test_labels=[1, 0],
            test_subjects=["s3", "s3"],
            baseline={"inner_mean_f1": 0.5, "brier": 0.25},
            config=_InvalidRecallConfig,
        )


def test_evaluate_and_promote_rejects_single_class_confirmation_without_crashing():
    result = evaluate_and_promote(
        validation_logits=[2.0, -2.0, 1.0, -1.0],
        validation_labels=[1, 0, 1, 0],
        validation_subjects=["s1", "s1", "s2", "s2"],
        test_logits=[1.0, 2.0],
        test_labels=[1, 1],
        test_subjects=["s3", "s3"],
        baseline={"inner_mean_f1": 0.5, "brier": 0.25},
        config=config,
    )

    assert result == {
        "promoted": False,
        "reason": "confirmation fold requires both classes",
        "temperature": 1.0,
        "threshold": None,
        "inner_metrics": None,
        "confirmation_metrics": None,
        "candidate": None,
        "baseline": {"inner_mean_f1": 0.5, "brier": 0.25},
        "fold_thresholds": None,
        "fold_temperatures": None,
    }


def test_evaluate_and_promote_rejects_mismatched_confirmation_lengths():
    with pytest.raises(ValueError, match="same length"):
        evaluate_and_promote(
            validation_logits=[2.0, -2.0, 1.0, -1.0],
            validation_labels=[1, 0, 1, 0],
            validation_subjects=["s1", "s1", "s2", "s2"],
            test_logits=[1.0, -1.0],
            test_labels=[1],
            test_subjects=["s3", "s3"],
            baseline={"inner_mean_f1": 0.5, "brier": 0.25},
            config=config,
        )


def test_f1_support_dependency_closure_round_trips_release(tmp_path):
    calibrator = Calibrator(temperature=1.0, threshold=0.5)
    probabilities = calibrator.calibrate([2.0, -2.0])
    metrics = evaluate_fall_event([1, 0], probabilities, ["s1", "s1"], 0.5)
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    bundle = build_release_bundle(
        "r1",
        checkpoint,
        tmp_path,
        inputs={
            "dataset_lock": {"clips": []},
            "split_manifest": {"partitions": {}},
            "metrics": {"promoted": False, "inner_mean_f1": metrics["inner_mean_f1"]},
            "calibration": {"temperature": 1.0, "threshold": 0.5},
        },
    )

    release_dir = tmp_path / "release"
    write_release_bundle(bundle, release_dir)

    assert (release_dir / "checkpoint.pt").read_bytes() == b"checkpoint"
    assert (release_dir / "calibration.json").exists()
    assert bundle.release_id == "r1"
