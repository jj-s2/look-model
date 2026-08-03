from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *arguments], cwd=ROOT, text=True, capture_output=True, check=False)


def _write_dataset(path: Path, tier: str = "real_public") -> None:
    rows = []
    for subject, event_day, score in (("alice", 1, 0.90), ("bob", None, 0.10), ("carol", 3, 0.75), ("dave", None, 0.20)):
        rows.append({
            "schema_version": "pmcc.dataset.v1", "dataset_id": "dataset-1", "release_id": "release-1",
            "subject_split_id": f"subject-{subject}",
            "observation": {"subject_id": subject, "observed_at": "2026-01-01T00:00:00+00:00",
                            "features": {"activity_count": 100.0}, "quality": {"activity_count": 0.9},
                            "availability": {"activity_count": True},
                            "provenance": {"evidence_tier": tier, "promoted": tier == "real_public"}},
            "provenance": {"evidence_tier": tier, "promoted": tier == "real_public"},
            "label": {"event_day": event_day, "censor_day": 7},
            "predictions": {
                "baseline": {"24h": min(1.0, score * 0.6), "72h": min(1.0, score * 0.7), "7d": min(1.0, score * 0.8)},
                "full": {"24h": score * 0.8, "72h": score * 0.9, "7d": score},
            },
            "abstained": subject == "dave", "coverage": 0.6 if subject != "dave" else 0.4,
            "quality": 0.9 if subject != "dave" else 0.4, "subgroup": "female" if subject in {"alice", "carol"} else "male",
        })
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_model(path: Path, release_id: str = "release-1", tier: str = "real_public") -> None:
    path.mkdir()
    (path / "model-card.json").write_text(json.dumps({"artifact_id": "model-1", "release_id": release_id,
        "evidence_tier": tier, "promoted": tier == "real_public", "clinical_use": False}), encoding="utf-8")


def test_evaluation_uses_subject_disjoint_splits_and_required_audits(tmp_path: Path) -> None:
    dataset, model, output = tmp_path / "data.jsonl", tmp_path / "model", tmp_path / "evaluation.json"
    _write_dataset(dataset)
    _write_model(model)

    completed = _run("scripts/evaluate_pmcc.py", "--input", str(dataset), "--model", str(model), "--output", str(output))

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["split_strategy"] == "subject_grouped_outer_with_inner_calibration"
    assert result["leakage_check"]["passed"] is True
    assert result["model_rows"].keys() == {"baseline", "full"}
    assert set(result["ablations"]) >= {"baseline", "chains", "sleep_physiology", "gait", "mask", "quality_gate", "uncertainty", "tcn"}
    assert result["coverage"]["abstention_rate"] == 0.25
    assert result["stratified"]["subgroup"]["unavailable"] is None
    assert result["provenance"]["release_id"] == "release-1"
    assert result["release_metrics_eligible"] is True
    assert result["model_rows"]["full"]["horizons"]["24h"]["brier"] is not None
    assert result["model_rows"]["full"]["horizons"]["24h"]["lead_time_days"] is not None
    assert result["model_rows"]["full"]["horizons"]["24h"]["false_alerts_per_subject_day"] is not None


def test_evaluation_rejects_mismatched_release_or_nonmonotonic_predictions(tmp_path: Path) -> None:
    dataset, model, output = tmp_path / "data.jsonl", tmp_path / "model", tmp_path / "evaluation.json"
    _write_dataset(dataset)
    _write_model(model, release_id="other-release")
    mismatch = _run("scripts/evaluate_pmcc.py", "--input", str(dataset), "--model", str(model), "--output", str(output))
    assert mismatch.returncode != 0
    assert "release" in mismatch.stderr.lower()

    _write_model(model, release_id="release-1") if not model.exists() else (model / "model-card.json").write_text(json.dumps({"release_id": "release-1", "evidence_tier": "real_public", "promoted": True}), encoding="utf-8")
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    rows[0]["predictions"]["full"] = {"24h": 0.8, "72h": 0.7, "7d": 0.9}
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    nonmonotonic = _run("scripts/evaluate_pmcc.py", "--input", str(dataset), "--model", str(model), "--output", str(output))
    assert nonmonotonic.returncode != 0
    assert "monotonic" in nonmonotonic.stderr.lower()


def test_synthetic_metrics_are_research_only_and_report_never_promotes_them(tmp_path: Path) -> None:
    dataset, model, output, report = tmp_path / "data.jsonl", tmp_path / "model", tmp_path / "evaluation.json", tmp_path / "report.md"
    _write_dataset(dataset, tier="synthetic_research")
    _write_model(model, tier="synthetic_research")

    completed = _run("scripts/evaluate_pmcc.py", "--input", str(dataset), "--model", str(model), "--output", str(output))
    assert completed.returncode == 0, completed.stderr
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["claim_boundary"] == "research_only"
    assert metrics["release_metrics_eligible"] is False
    release = _run("scripts/evaluate_pmcc.py", "--input", str(dataset), "--model", str(model), "--output", str(output), "--release-mode")
    assert release.returncode != 0

    generated = _run("scripts/generate_pmcc_report.py", "--metrics", str(output), "--output", str(report))
    assert generated.returncode == 0, generated.stderr
    text = report.read_text(encoding="utf-8")
    assert "research_only" in text
    assert "absolute" not in text.lower()


def test_evaluation_uses_only_held_out_subjects_and_time_blocked_inner_data(tmp_path: Path) -> None:
    dataset, model, output = tmp_path / "data.jsonl", tmp_path / "model", tmp_path / "evaluation.json"
    _write_dataset(dataset)
    _write_model(model)

    completed = _run("scripts/evaluate_pmcc.py", "--input", str(dataset), "--model", str(model), "--output", str(output))

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["split_metadata"]["evaluation_population"] == "outer_held_out_subject_windows_only"
    for fold in result["split_metadata"]["folds"]:
        assert not set(fold["test_subjects"]) & set(fold["training_subjects"])
        assert fold["inner_calibration"]["time_blocked"] is True
        assert fold["held_out_records"] > 0


def test_censoring_excludes_unknown_horizon_outcomes_from_metrics(tmp_path: Path) -> None:
    dataset, model, output = tmp_path / "data.jsonl", tmp_path / "model", tmp_path / "evaluation.json"
    _write_dataset(dataset)
    _write_model(model)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    rows[1]["label"] = {"event_day": None, "censor_day": 1}
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    completed = _run("scripts/evaluate_pmcc.py", "--input", str(dataset), "--model", str(model), "--output", str(output))

    assert completed.returncode == 0, completed.stderr
    horizon = json.loads(output.read_text(encoding="utf-8"))["model_rows"]["full"]["horizons"]["72h"]
    assert horizon["censored_before_horizon"] == 1
    assert horizon["evaluated_records"] == 2  # dave abstains; bob is censored at day 1


def test_real_scoreless_records_are_rejected_and_synthetic_scoreless_metrics_are_unavailable(tmp_path: Path) -> None:
    dataset, model, output, report = tmp_path / "data.jsonl", tmp_path / "model", tmp_path / "evaluation.json", tmp_path / "report.md"
    _write_dataset(dataset)
    _write_model(model)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row.pop("predictions")
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    real = _run("scripts/evaluate_pmcc.py", "--input", str(dataset), "--model", str(model), "--output", str(output))
    assert real.returncode != 0
    assert "predictions" in real.stderr.lower()

    _write_dataset(dataset, tier="synthetic_research")
    _write_model(model, tier="synthetic_research") if not model.exists() else (model / "model-card.json").write_text(json.dumps({"evidence_tier": "synthetic_research", "promoted": False}), encoding="utf-8")
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row.pop("predictions")
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    synthetic = _run("scripts/evaluate_pmcc.py", "--input", str(dataset), "--model", str(model), "--output", str(output))
    assert synthetic.returncode == 0, synthetic.stderr
    metric = json.loads(output.read_text(encoding="utf-8"))["model_rows"]["full"]["horizons"]["24h"]
    assert metric["auroc"] is None
    assert metric["reason"] == "scored_predictions_unavailable"
    assert _run("scripts/generate_pmcc_report.py", "--metrics", str(output), "--output", str(report)).returncode == 0
    assert "| full | 24h | unavailable | unavailable | unavailable |" in report.read_text(encoding="utf-8")
