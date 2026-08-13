from __future__ import annotations

import hashlib
import json
from pathlib import Path

from risk.phase_model.experiment_matrix import build_experiment_matrix
from scripts.run_rg_pcnet_experiment_matrix import run_experiment_matrix


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_real_artifact(run, run_dir: Path, input_hashes: dict[str, str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run.run_id,
        "status": "success",
        "synthetic": False,
        "demo": False,
        "manual_metrics": False,
        "input_hashes": input_hashes,
        "config_sha256": run.config_sha256,
        "variant": run.variant,
        "seed": run.seed,
        "outer_subject": run.outer_subject,
        "train_datasets": list(run.train_datasets),
        "held_out_dataset": run.held_out_dataset,
        "config_overrides": dict(run.config_overrides),
        "metrics": {
            "event_f1": 0.75,
            "event_recall": 0.8,
            "false_alerts_per_hour": 0.1,
            "coverage": 0.9,
            "aurc": 0.2,
            "latency_median_seconds": 1.2,
        },
        "cross_domain": {"dataset": run.held_out_dataset, "event_f1": 0.7},
        "corruption": {"severity": "mild", "event_f1": 0.65},
        "latency": {"p50_ms": 12.0, "p95_ms": 20.0},
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_matrix_aggregates_real_artifacts_and_resumes_by_hash(tmp_path: Path):
    matrix = build_experiment_matrix(
        outer_subjects=("s1",), seeds=(42,), variants=("fall_only", "phase"),
        train_datasets=("GMDCSA24",), held_out_dataset="OmniFall",
    )
    input_hashes = {"dataset_lock": _sha("dataset-lock")}
    calls: list[str] = []

    def runner(run, run_dir):
        calls.append(run.run_id)
        _write_real_artifact(run, run_dir, input_hashes)

    first = run_experiment_matrix(
        matrix,
        output_root=tmp_path,
        runner=runner,
        input_hashes=input_hashes,
    )
    assert len(calls) == 2
    assert first["status"] == "success"
    assert first["run_count"] == 2
    assert first["summaries"]["macro"]["event_f1"] == 0.75
    assert (tmp_path / "experiment_matrix.json").is_file()

    second = run_experiment_matrix(
        matrix,
        output_root=tmp_path,
        runner=runner,
        input_hashes=input_hashes,
    )
    assert len(calls) == 2, "matching successful hashes must resume without rerunning"
    assert second["resumed_count"] == 2


def test_matrix_rejects_synthetic_artifact(tmp_path: Path):
    matrix = build_experiment_matrix(outer_subjects=("s1",), seeds=(42,), variants=("phase",))
    input_hashes = {"dataset_lock": _sha("dataset-lock")}

    def runner(run, run_dir):
        _write_real_artifact(run, run_dir, input_hashes)
        path = run_dir / "run_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["synthetic"] = True
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    try:
        run_experiment_matrix(matrix, output_root=tmp_path, runner=runner, input_hashes=input_hashes)
    except ValueError as exc:
        assert "synthetic" in str(exc)
    else:  # pragma: no cover - aggregation must never accept demo data
        raise AssertionError("synthetic artifacts must be rejected")


def test_matrix_aggregation_is_independent_of_input_run_order(tmp_path: Path):
    matrix = build_experiment_matrix(
        outer_subjects=("s1", "s2"), seeds=(41,), variants=("fall_only", "phase")
    )
    input_hashes = {"dataset_lock": _sha("dataset-lock")}

    def runner(run, run_dir):
        _write_real_artifact(run, run_dir, input_hashes)

    forward = run_experiment_matrix(
        matrix, output_root=tmp_path / "forward", runner=runner, input_hashes=input_hashes
    )
    reverse = run_experiment_matrix(
        tuple(reversed(matrix)), output_root=tmp_path / "reverse", runner=runner, input_hashes=input_hashes
    )

    assert reverse["matrix_sha256"] == forward["matrix_sha256"]
    assert reverse["runs"] == forward["runs"]
    assert reverse["summaries"] == forward["summaries"]


def test_matrix_rejects_manifest_with_mismatched_run_spec(tmp_path: Path):
    matrix = build_experiment_matrix(outer_subjects=("s1",), seeds=(42,), variants=("phase",))
    input_hashes = {"dataset_lock": _sha("dataset-lock")}

    def runner(run, run_dir):
        _write_real_artifact(run, run_dir, input_hashes)
        path = run_dir / "run_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["train_datasets"] = ["OmniFall"]
        payload["held_out_dataset"] = "OmniFall"
        payload["config_overrides"] = {"phase_weight": 999}
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    try:
        run_experiment_matrix(matrix, output_root=tmp_path, runner=runner, input_hashes=input_hashes)
    except ValueError as exc:
        assert "run specification mismatch" in str(exc)
    else:  # pragma: no cover - a mismatched runner spec must never be accepted
        raise AssertionError("mismatched run specification must be rejected")


def test_matrix_rejects_numeric_type_drift_in_run_spec(tmp_path: Path):
    matrix = build_experiment_matrix(outer_subjects=("s1",), seeds=(42,), variants=("phase",))
    input_hashes = {"dataset_lock": _sha("dataset-lock")}

    def runner(run, run_dir):
        _write_real_artifact(run, run_dir, input_hashes)
        path = run_dir / "run_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["seed"] = 42.0
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    try:
        run_experiment_matrix(matrix, output_root=tmp_path, runner=runner, input_hashes=input_hashes)
    except ValueError as exc:
        assert "run specification mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("numeric type drift must be rejected")
