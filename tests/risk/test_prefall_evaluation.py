import json
import subprocess
import sys
from pathlib import Path

import pytest

from risk.prefall_evaluation import evaluate_subject_wise, make_subject_folds, should_promote
from test_prefall_model import SchemaArray


SUBJECT_IDS = ["a", "a", "b", "b", "c", "c"]


def test_subjects_never_cross_train_and_validation_folds():
    folds = make_subject_folds(SUBJECT_IDS)

    for train_idx, valid_idx in folds:
        train_subjects = {SUBJECT_IDS[index] for index in train_idx}
        valid_subjects = {SUBJECT_IDS[index] for index in valid_idx}
        assert train_subjects.isdisjoint(valid_subjects)


def test_candidate_is_not_promoted_when_recall_regresses():
    assert should_promote(
        candidate={"f1": 0.92, "recall": 0.84},
        baseline={"f1": 0.90, "recall": 0.88},
    ) is False


def test_promoted_candidate_must_meet_baseline_and_global_gates():
    assert should_promote(
        candidate={"f1": 0.91, "recall": 0.89},
        baseline={"f1": 0.90, "recall": 0.88},
    ) is True
    assert should_promote(
        candidate={"f1": 0.89, "recall": 0.90},
        baseline={"f1": 0.80, "recall": 0.80},
    ) is False


def test_groupkfold_and_logo_keep_subjects_out_of_real_validation_metrics():
    pytest.importorskip("sklearn")
    features = SchemaArray([
        [0.1, 0.2], [0.2, 0.3], [0.8, 0.7], [0.9, 0.8],
        [0.15, 0.25], [0.85, 0.75], [0.18, 0.28], [0.88, 0.78],
    ])
    labels = [0, 0, 1, 1, 0, 1, 0, 1]
    subject_ids = ["a", "a", "b", "b", "c", "c", "d", "d"]

    group_report = evaluate_subject_wise(features, labels, subject_ids, n_splits=3)
    logo_report = evaluate_subject_wise(features, labels, subject_ids, leave_one_subject_out=True)

    assert group_report.strategy == "GroupKFold"
    assert logo_report.strategy == "LeaveOneGroupOut"
    assert len(logo_report.fold_metrics) == 4
    assert group_report.metrics["f1"] == pytest.approx(
        sum(fold["f1"] for fold in group_report.fold_metrics) / len(group_report.fold_metrics)
    )
    for train_idx, valid_idx in make_subject_folds(subject_ids, leave_one_subject_out=True):
        assert {subject_ids[index] for index in train_idx}.isdisjoint(
            {subject_ids[index] for index in valid_idx}
        )


def test_synthetic_smoke_artifacts_are_never_promoted_or_clinical(tmp_path):
    pytest.importorskip("sklearn")
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_prefall_model.py"
    result = subprocess.run(
        [sys.executable, str(script), "--synthetic-smoke-test", "--output-dir", str(tmp_path)],
        check=True, capture_output=True, text=True,
    )
    assert result.returncode == 0
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    card = json.loads((tmp_path / "model_card.json").read_text(encoding="utf-8"))
    for artifact in (metrics, card):
        assert artifact["synthetic"] is True
        assert artifact["not_for_clinical_performance"] is True
        assert artifact["source"] == "synthetic-smoke-test"
        assert artifact["fixture"]
        assert artifact["promoted"] is False


def test_real_training_model_card_records_dataset_window_audit(tmp_path):
    pytest.importorskip("sklearn")
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_prefall_model.py"
    csv_path = tmp_path / "prefall.csv"
    csv_path.write_text(
        "subject_id,label,sway,step_width\n"
        "a,0,0.1,0.2\n"
        "a,1,0.9,0.8\n"
        "b,0,0.2,0.1\n"
        "b,1,0.8,0.9\n",
        encoding="utf-8",
    )
    audit = {"horizon_sec": 3.0, "guard_sec": 0.5, "excluded_no_window": 2}
    audit_path = tmp_path / "dataset_summary.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    subprocess.run(
        [sys.executable, str(script), "--input-csv", str(csv_path), "--output-dir", str(tmp_path / "run"),
         "--dataset-summary", str(audit_path)],
        check=True, capture_output=True, text=True,
    )

    card = json.loads((tmp_path / "run" / "model_card.json").read_text(encoding="utf-8"))
    assert card["dataset_audit"] == audit


def test_real_training_can_use_leave_one_subject_out(tmp_path):
    pytest.importorskip("sklearn")
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_prefall_model.py"
    csv_path = tmp_path / "prefall.csv"
    csv_path.write_text(
        "subject_id,label,sway,step_width\n"
        "a,0,0.1,0.2\n"
        "a,1,0.9,0.8\n"
        "b,0,0.2,0.1\n"
        "b,1,0.8,0.9\n",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, str(script), "--input-csv", str(csv_path), "--output-dir", str(tmp_path / "run"),
         "--leave-one-subject-out"],
        check=True, capture_output=True, text=True,
    )

    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["validation"]["strategy"] == "LeaveOneGroupOut"


def test_subject_evaluation_supports_the_selected_estimator():
    pytest.importorskip("sklearn")
    features = SchemaArray([
        [0.1, 0.2], [0.2, 0.3], [0.8, 0.7], [0.9, 0.8],
        [0.15, 0.25], [0.85, 0.75], [0.18, 0.28], [0.88, 0.78],
    ])
    report = evaluate_subject_wise(
        features, [0, 0, 1, 1, 0, 1, 0, 1], ["a", "a", "b", "b", "c", "c", "d", "d"],
        leave_one_subject_out=True, estimator_name="extra_trees",
    )

    assert report.strategy == "LeaveOneGroupOut"


def test_real_training_with_zero_guard_cannot_be_promoted(tmp_path):
    pytest.importorskip("sklearn")
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_prefall_model.py"
    csv_path = tmp_path / "prefall.csv"
    csv_path.write_text(
        "subject_id,label,sway,step_width\n"
        "a,0,0.1,0.2\n"
        "a,1,0.9,0.8\n"
        "b,0,0.2,0.1\n"
        "b,1,0.8,0.9\n",
        encoding="utf-8",
    )
    audit_path = tmp_path / "dataset_summary.json"
    audit_path.write_text(json.dumps({"horizon_sec": 1.0, "guard_sec": 0.0}), encoding="utf-8")

    subprocess.run(
        [sys.executable, str(script), "--input-csv", str(csv_path), "--output-dir", str(tmp_path / "run"),
         "--dataset-summary", str(audit_path), "--leave-one-subject-out"],
        check=True, capture_output=True, text=True,
    )

    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["promoted"] is False
    assert "guard_sec below 0.5" in metrics["promotion_blockers"]
