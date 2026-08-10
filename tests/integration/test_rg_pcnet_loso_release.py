from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_rg_pcnet_loso import evaluate_rg_pcnet_loso


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _inner_records() -> list[dict[str, object]]:
    return [
        {"subject_id": "s1", "label": 1, "fall_logit": 3.0, "reliability": 0.9},
        {"subject_id": "s1", "label": 0, "fall_logit": -3.0, "reliability": 0.9},
        {"subject_id": "s2", "label": 1, "fall_logit": 2.0, "reliability": 0.8},
        {"subject_id": "s2", "label": 0, "fall_logit": -2.0, "reliability": 0.8},
        {"subject_id": "s3", "label": 1, "fall_logit": 1.5, "reliability": 0.7},
        {"subject_id": "s3", "label": 0, "fall_logit": -1.5, "reliability": 0.7},
    ]


def _outer_records() -> list[dict[str, object]]:
    return [
        {"subject_id": "s4", "label": 1, "fall_logit": 2.5, "reliability": 0.9},
        {"subject_id": "s4", "label": 0, "fall_logit": -2.5, "reliability": 0.9},
    ]


def test_nested_loso_release_uses_inner_selection_and_writes_auditable_artifacts(tmp_path: Path) -> None:
    inner = tmp_path / "inner.jsonl"
    outer = tmp_path / "outer.jsonl"
    output = tmp_path / "release"
    _write_records(inner, _inner_records())
    _write_records(outer, _outer_records())

    result = evaluate_rg_pcnet_loso(inner, outer, output)

    assert result["outer_subject"] == "s4"
    assert "s4" not in result["calibration_subjects"]
    assert result["release_config"]["fall_threshold"] == result["selection"]["fall_threshold"]
    assert result["release_config"]["reliability_threshold"] == result["selection"]["reliability_threshold"]
    assert result["promoted"] is False
    assert result["reason"] == "continuous event evaluation is required"
    assert (output / "outer_predictions.jsonl").exists()
    assert (output / "calibration.json").exists()
    assert (output / "selection.json").exists()
    assert (output / "release_config.json").exists()


def test_nested_loso_rejects_outer_subject_leakage(tmp_path: Path) -> None:
    inner = tmp_path / "inner.jsonl"
    outer = tmp_path / "outer.jsonl"
    _write_records(inner, _inner_records() + [
        {"subject_id": "s4", "label": 0, "fall_logit": 0.0, "reliability": 0.5},
    ])
    _write_records(outer, _outer_records())

    with pytest.raises(ValueError, match="outer subject leaked into calibration"):
        evaluate_rg_pcnet_loso(inner, outer, tmp_path / "release")
