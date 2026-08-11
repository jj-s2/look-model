from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.evaluate_rg_pcnet_loso import (
    RGPCPromotionArtifacts,
    finalize_rgpc_release,
)


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _candidate(tmp_path: Path) -> RGPCPromotionArtifacts:
    release_id = "rgpc-r1"
    config = {
        "schema_version": "rgpc.release.v1",
        "release_id": release_id,
        "model_sha256": _sha("checkpoint"),
        "dataset_sha256": _sha("dataset"),
        "split_sha256": _sha("split"),
        "temperature": 1.0,
        "fall_threshold": 0.6,
        "reliability_threshold": 0.7,
        "confirm_seconds": 0.8,
        "recovery_seconds": 2.0,
        "cooldown_seconds": 10.0,
        "minimum_coverage": 0.8,
    }
    baseline = {
        "macro_event_f1": 0.70,
        "false_alerts_per_hour": 0.30,
        "aurc": 0.30,
    }
    nested = {
        "release_id": release_id,
        "selection": {"feasible": True, "subject_macro_f1": 0.84, "aurc": 0.18},
        "aggregate_metrics": {
            "macro_event_f1": 0.84,
            "event_recall": 0.92,
            "false_alerts_per_hour": 0.10,
            "coverage": 0.90,
            "aurc": 0.18,
        },
        "calibration": {"valid": True, "ece": 0.04},
        "seeds": [41, 42, 43],
        "release_config": config,
    }
    continuous = {
        "release_id": release_id,
        "passed": True,
        "checks": {
            "event_recall": True,
            "false_alerts_per_hour": True,
            "coverage": True,
        },
        "constraints": {
            "minimum_event_recall": 0.80,
            "maximum_false_alerts_per_hour": 0.20,
            "minimum_coverage": 0.80,
        },
        "input_hashes": {
            "predictions": _sha("predictions"),
            "truth_events": _sha("truth"),
            "release_config": _sha("release-config-bytes"),
        },
        "output_sha256": {
            "continuous_metrics.json": _sha("metrics"),
            "transitions.jsonl": _sha("transitions"),
            "alerts.jsonl": _sha("alerts"),
            "promotion_gate.json": _sha("gate"),
        },
    }
    return RGPCPromotionArtifacts(
        nested_summary=nested,
        baseline_summary=baseline,
        continuous_gate=continuous,
        release_config=config,
        checkpoint_sha256=config["model_sha256"],
        config_sha256=continuous["input_hashes"]["release_config"],
        output_dir=tmp_path,
        input_hashes=continuous["input_hashes"],
    )


def test_release_is_not_promoted_when_continuous_gate_fails(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate.continuous_gate["passed"] = False

    result = finalize_rgpc_release(candidate)

    assert result["promoted"] is False
    assert "continuous" in result["reasons"]
    assert not (tmp_path / "release").exists()


def test_release_requires_matching_release_and_input_hashes(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate.continuous_gate["release_id"] = "another-release"

    with pytest.raises(ValueError, match="release_id mismatch"):
        finalize_rgpc_release(candidate)


def test_release_requires_all_conjunctive_evidence(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate.nested_summary["aggregate_metrics"]["coverage"] = 0.1
    candidate.nested_summary["seeds"] = [41]

    result = finalize_rgpc_release(candidate)

    assert result["promoted"] is False
    assert {"coverage", "seeds"}.issubset(set(result["reasons"]))
    assert not (tmp_path / "release").exists()


def test_passing_candidate_writes_release_only_after_all_hashes_agree(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)

    result = finalize_rgpc_release(candidate)

    release = tmp_path / "release"
    assert result["promoted"] is True
    assert release.is_dir()
    manifest = json.loads((release / "promotion_result.json").read_text(encoding="utf-8"))
    assert manifest["release_id"] == "rgpc-r1"
    assert manifest["promoted"] is True
    assert manifest["release_dir"] == str(release)


def test_promotion_requires_explicit_hash_provenance(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate.checkpoint_sha256 = None
    candidate.config_sha256 = None
    candidate.continuous_gate["input_hashes"] = {"foo": _sha("arbitrary")}
    candidate.continuous_gate["output_sha256"] = {"foo": _sha("arbitrary-output")}

    result = finalize_rgpc_release(candidate)

    assert result["promoted"] is False
    assert "hashes" in result["reasons"]
    assert not (tmp_path / "release").exists()


def test_embedded_release_config_must_match_candidate_config(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate.nested_summary["release_config"] = copy.deepcopy(candidate.release_config)
    candidate.nested_summary["release_config"]["model_sha256"] = _sha("different-model")

    result = finalize_rgpc_release(candidate)

    assert result["promoted"] is False
    assert "release_config" in result["reasons"]


def test_release_config_schema_is_validated(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate.release_config["schema_version"] = "bad.schema"

    result = finalize_rgpc_release(candidate)

    assert result["promoted"] is False
    assert "release_config" in result["reasons"]


def test_continuous_gate_checks_are_required(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate.continuous_gate.pop("checks")

    result = finalize_rgpc_release(candidate)

    assert result["promoted"] is False
    assert "continuous" in result["reasons"]


def test_seed_values_must_be_numeric(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate.nested_summary["seeds"] = ["one", "two", "three"]

    result = finalize_rgpc_release(candidate)

    assert result["promoted"] is False
    assert "seeds" in result["reasons"]


def test_zero_coverage_floor_is_not_replaced_by_default(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate.continuous_gate["constraints"]["minimum_coverage"] = 0.0
    candidate.nested_summary["aggregate_metrics"]["coverage"] = 0.0

    result = finalize_rgpc_release(candidate)

    assert result["promoted"] is True


def test_negative_coverage_floor_is_rejected(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    candidate.continuous_gate["constraints"]["minimum_coverage"] = -0.1

    result = finalize_rgpc_release(candidate)

    assert result["promoted"] is False
    assert "coverage" in result["reasons"]
