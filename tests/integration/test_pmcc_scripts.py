from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_fixture_generation_is_seeded_and_marks_research_only(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    first_run = _run("scripts/generate_pmcc_fixture.py", "--output", str(first), "--seed", "42")
    second_run = _run("scripts/generate_pmcc_fixture.py", "--output", str(second), "--seed", "42")

    assert first_run.returncode == 0, first_run.stderr
    assert second_run.returncode == 0, second_run.stderr
    assert first.read_bytes() == second.read_bytes()
    records = _read_jsonl(first)
    assert {record["scenario"] for record in records} == {
        "normal", "sleep_activity_gait_cascade", "missing_physiology", "timestamp_shift"
    }
    assert all(record["provenance"]["evidence_tier"] == "synthetic_research" for record in records)
    assert all(record["provenance"]["promoted"] is False for record in records)


def test_dataset_builder_validates_source_and_preserves_evidence_tier(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    assert _run("scripts/generate_pmcc_fixture.py", "--output", str(fixture)).returncode == 0

    completed = _run(
        "scripts/build_pmcc_dataset.py", "--input", str(fixture), "--output", str(dataset),
        "--evidence-tier", "synthetic_research",
    )

    assert completed.returncode == 0, completed.stderr
    records = _read_jsonl(dataset)
    assert records
    assert {record["provenance"]["evidence_tier"] for record in records} == {"synthetic_research"}
    assert all(record["provenance"]["promoted"] is False for record in records)
    assert all(record["release_id"] and record["dataset_id"] and record["subject_split_id"] for record in records)
    assert all(record["schema_version"] == "pmcc.dataset.v1" for record in records)


def test_dataset_builder_rejects_mixed_or_invalid_evidence(tmp_path: Path) -> None:
    source = tmp_path / "mixed.jsonl"
    source.write_text(
        "\n".join((
            json.dumps({"subject_id": "a", "observed_at": "2026-01-01T00:00:00+00:00", "features": {}, "quality": {}, "availability": {}, "provenance": {"evidence_tier": "real_public", "promoted": True}, "label": {"event_day": None, "censor_day": 7}}),
            json.dumps({"subject_id": "b", "observed_at": "2026-01-02T00:00:00+00:00", "features": {}, "quality": {}, "availability": {}, "provenance": {"evidence_tier": "synthetic_research", "promoted": False}, "label": {"event_day": 1, "censor_day": 7}}),
        )),
        encoding="utf-8",
    )

    completed = _run(
        "scripts/build_pmcc_dataset.py", "--input", str(source), "--output", str(tmp_path / "dataset.jsonl"),
        "--evidence-tier", "real_public",
    )

    assert completed.returncode != 0
    assert "evidence" in completed.stderr.lower()


def test_builder_and_trainer_reject_forged_top_level_provenance(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.jsonl"
    forged = tmp_path / "forged.jsonl"
    assert _run("scripts/generate_pmcc_fixture.py", "--output", str(fixture)).returncode == 0
    records = _read_jsonl(fixture)[:1]
    records[0]["provenance"] = {"evidence_tier": "real_public", "promoted": True}
    forged.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    built = _run(
        "scripts/build_pmcc_dataset.py", "--input", str(forged), "--output", str(tmp_path / "dataset.jsonl"),
        "--evidence-tier", "real_public",
    )
    trained = _run("scripts/train_pmcc.py", "--input", str(forged), "--output", str(tmp_path / "model"))

    assert built.returncode != 0
    assert trained.returncode != 0
    assert "provenance" in built.stderr.lower()
    assert "provenance" in trained.stderr.lower()


def test_train_script_writes_non_promoted_synthetic_model_card(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.jsonl"
    model_dir = tmp_path / "model"
    assert _run("scripts/generate_pmcc_fixture.py", "--output", str(fixture), "--seed", "42").returncode == 0

    completed = _run("scripts/train_pmcc.py", "--input", str(fixture), "--output", str(model_dir), "--seed", "42")

    assert completed.returncode == 0, completed.stderr
    card = json.loads((model_dir / "model-card.json").read_text(encoding="utf-8"))
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    assert (model_dir / "model.json").exists()
    assert card["promoted"] is False
    assert card["evidence_tier"] == "synthetic_research"
    assert card["clinical_use"] is False
    assert any(name.endswith(":directional_z") for name in card["feature_schema"])
    assert any(name.endswith(":raw") for name in card["feature_schema"])
    assert "chain_strength" in card["feature_schema"]
    assert metrics["release_metrics"] is None


def test_portable_model_artifact_round_trips_and_is_accepted_by_service(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.jsonl"
    model_dir = tmp_path / "model"
    assert _run("scripts/generate_pmcc_fixture.py", "--output", str(fixture), "--seed", "42").returncode == 0
    assert _run("scripts/train_pmcc.py", "--input", str(fixture), "--output", str(model_dir), "--seed", "42").returncode == 0

    from risk.pmcc.calibrator import RuleSurvivalCalibrator
    from risk.pmcc.features import FeatureWindow
    from risk.pmcc.schema import DailyObservation
    from risk.pmcc.service import PMCCService

    payload = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    loaded = RuleSurvivalCalibrator.from_artifact(payload)
    reloaded = RuleSurvivalCalibrator.load_artifact(model_dir / "model.json")
    assert loaded.metadata() == reloaded.metadata()
    schema = tuple(payload["feature_schema"])
    window = FeatureWindow(
        values=tuple(tuple(1.0 for _ in schema) for _ in range(14)),
        missing_mask=tuple(tuple(False for _ in schema) for _ in range(14)),
        quality=tuple(tuple(1.0 for _ in schema) for _ in range(14)),
        feature_names=schema,
    )
    assert loaded.predict_hazards(window) == reloaded.predict_hazards(window)
    malformed = json.loads(json.dumps(payload))
    malformed["estimators"][0]["weights"].append(0.0)
    malformed["estimators"][0]["mean"].append(0.0)
    malformed["estimators"][0]["scale"].append(1.0)
    with pytest.raises(ValueError, match="width"):
        RuleSurvivalCalibrator.from_artifact(malformed)

    observation = DailyObservation.from_dict(_read_jsonl(fixture)[0]["observation"])
    forecast = PMCCService(model=model_dir / "model.json", observations=(observation,)).forecast(
        observation.subject_id, observation.observed_at.date()
    )
    assert forecast.provenance["model"] == "configured_model"

    missing_data = observation.to_dict()
    missing_data["features"].pop("heart_rate", None)
    missing_data["quality"].pop("heart_rate", None)
    missing_data["availability"].pop("heart_rate", None)
    missing_observation = DailyObservation.from_dict(missing_data)
    missing_forecast = PMCCService(
        model=model_dir / "model.json", observations=(missing_observation,)
    ).forecast(missing_observation.subject_id, missing_observation.observed_at.date())
    assert missing_forecast.provenance["model"] == "configured_model"
    assert "heart_rate:raw" in missing_forecast.provenance["feature_names"]


def test_model_card_carries_dataset_release_id(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.jsonl"
    dataset = tmp_path / "dataset.jsonl"
    model_dir = tmp_path / "model"
    assert _run("scripts/generate_pmcc_fixture.py", "--output", str(fixture), "--seed", "42").returncode == 0
    built = _run(
        "scripts/build_pmcc_dataset.py", "--input", str(fixture), "--output", str(dataset),
        "--evidence-tier", "synthetic_research",
    )
    assert built.returncode == 0, built.stderr
    trained = _run("scripts/train_pmcc.py", "--input", str(dataset), "--output", str(model_dir), "--seed", "42")
    assert trained.returncode == 0, trained.stderr
    card = json.loads((model_dir / "model-card.json").read_text(encoding="utf-8"))
    source_records = _read_jsonl(dataset)
    assert card["release_id"] == source_records[0]["release_id"]


def test_train_script_refuses_records_without_labels(tmp_path: Path) -> None:
    source = tmp_path / "unlabeled.jsonl"
    source.write_text(
        json.dumps({"subject_id": "a", "observed_at": "2026-01-01T00:00:00+00:00", "features": {}, "quality": {}, "availability": {}, "provenance": {"evidence_tier": "synthetic_research", "promoted": False}}) + "\n",
        encoding="utf-8",
    )
    completed = _run("scripts/train_pmcc.py", "--input", str(source), "--output", str(tmp_path / "model"))
    assert completed.returncode != 0
    assert "label" in completed.stderr.lower()
