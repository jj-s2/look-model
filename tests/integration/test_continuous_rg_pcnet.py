import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from risk.phase_model.release_config import RGPCReleaseConfig, write_release_config
from scripts.evaluate_continuous_rg_pcnet import evaluate_continuous


def _release() -> RGPCReleaseConfig:
    return RGPCReleaseConfig(
        schema_version="rgpc.release.v1",
        release_id="r1",
        model_sha256="a" * 64,
        dataset_sha256="b" * 64,
        split_sha256="c" * 64,
        temperature=1.0,
        fall_threshold=0.6,
        reliability_threshold=0.7,
        confirm_seconds=2.0,
        recovery_seconds=2.0,
        cooldown_seconds=3.0,
        minimum_coverage=0.8,
    )


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    predictions = tmp_path / "predictions.jsonl"
    truth = tmp_path / "truth-events.json"
    release = tmp_path / "release_config.json"
    output = tmp_path / "artifacts"
    rows = []
    for timestamp in range(10):
        abstain = timestamp == 4
        rows.append(
            {
                "release_id": "r1",
                "timestamp": float(timestamp),
                "fall_probability": 0.9 if timestamp <= 3 else 0.1,
                "phase": "descent_or_impact" if timestamp <= 3 else "normal",
                "reliable": not abstain,
            }
        )
    predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    truth.write_text(
        json.dumps(
            {
                "release_id": "r1",
                "duration_seconds": 60.0,
                "tolerance_seconds": 0.0,
                "events": [{"event_id": "truth-1", "start": 1.0, "end": 5.0}],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_release_config(_release(), release)
    return predictions, truth, release, output


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def test_continuous_evaluation_writes_verifiable_artifacts(tmp_path: Path):
    predictions, truth, release, output = _write_fixture(tmp_path)
    result = evaluate_continuous(
        predictions=predictions,
        truth_events=truth,
        release_config=release,
        output=output,
        minimum_event_recall=1.0,
        maximum_false_alerts_per_hour=0.0,
        minimum_coverage=0.8,
    )

    assert set(result) == {
        "continuous_metrics",
        "transitions",
        "alerts",
        "promotion_gate",
    }
    metrics = json.loads((output / "continuous_metrics.json").read_text(encoding="utf-8"))
    gate = json.loads((output / "promotion_gate.json").read_text(encoding="utf-8"))
    assert metrics["event_recall"] == 1.0
    assert metrics["false_alerts_per_hour"] == 0.0
    assert metrics["coverage"] == pytest.approx(0.9)
    assert gate["passed"] is True
    assert gate["release_id"] == "r1"
    assert len(gate["input_sha256"]) == 64
    assert gate["constraints"] == {
        "minimum_event_recall": 1.0,
        "maximum_false_alerts_per_hour": 0.0,
        "minimum_coverage": 0.8,
    }
    output_hashes = gate["output_sha256"]
    assert set(output_hashes) == {
        "continuous_metrics.json",
        "transitions.jsonl",
        "alerts.jsonl",
        "promotion_gate.json",
    }
    for name in ("continuous_metrics.json", "transitions.jsonl", "alerts.jsonl"):
        assert output_hashes[name] == _sha256((output / name).read_bytes())

    # The gate cannot contain a fixed-point hash of its own final bytes.  Its
    # self entry is therefore explicitly self-excluding and must hash the
    # final canonical gate with only that entry removed.
    self_excluding_gate = dict(gate)
    self_excluding_output_hashes = dict(output_hashes)
    self_excluding_output_hashes.pop("promotion_gate.json")
    self_excluding_gate["output_sha256"] = self_excluding_output_hashes
    assert output_hashes["promotion_gate.json"] == _sha256(
        _canonical_json(self_excluding_gate)
    )
    assert gate["output_sha256_scope"] == (
        "promotion_gate.json hash is over canonical gate JSON with its "
        "self entry removed"
    )
    alerts = [
        json.loads(line)
        for line in (output / "alerts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {record["emitted"] for record in alerts} >= {
        "fall_confirmed",
        "fall_recovered",
    }
    assert (output / "transitions.jsonl").read_text(encoding="utf-8").endswith("\n")
    assert (output / "alerts.jsonl").read_text(encoding="utf-8").endswith("\n")


def test_gate_is_conjunctive_and_prediction_order_is_strict(tmp_path: Path):
    predictions, truth, release, output = _write_fixture(tmp_path)
    rows = predictions.read_text(encoding="utf-8").splitlines()
    rows[1], rows[2] = rows[2], rows[1]
    predictions.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_continuous(
            predictions=predictions,
            truth_events=truth,
            release_config=release,
            output=output,
            minimum_event_recall=1.0,
            maximum_false_alerts_per_hour=0.0,
            minimum_coverage=0.8,
        )


def test_direct_cli_invocation_bootstraps_repository_imports():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "evaluate_continuous_rg_pcnet.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--predictions" in result.stdout
