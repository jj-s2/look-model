import json
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
