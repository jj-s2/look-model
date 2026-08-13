from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_rg_pcnet_loso import evaluate_rg_pcnet_loso
from risk.phase_model.release_config import load_release_config


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


def test_nested_loso_normalizes_subject_whitespace_before_leakage_check(tmp_path: Path) -> None:
    inner = tmp_path / "inner.jsonl"
    outer = tmp_path / "outer.jsonl"
    _write_records(inner, _inner_records() + [
        {"subject_id": "s4", "label": 0, "fall_logit": 0.0, "reliability": 0.5},
    ])
    _write_records(outer, [
        {"subject_id": " s4 ", "label": 1, "fall_logit": 2.5, "reliability": 0.9},
        {"subject_id": " s4 ", "label": 0, "fall_logit": -2.5, "reliability": 0.9},
    ])

    with pytest.raises(ValueError, match="outer subject leaked into calibration"):
        evaluate_rg_pcnet_loso(inner, outer, tmp_path / "release")


def test_infeasible_selection_emits_loadable_disabled_release_config(tmp_path: Path) -> None:
    inner = tmp_path / "inner.jsonl"
    outer = tmp_path / "outer.jsonl"
    # Deliberately invert both classes so no threshold can satisfy the default
    # recall/FPR constraints.
    _write_records(inner, [
        {"subject_id": "s1", "label": 1, "fall_logit": -3.0, "reliability": 0.9},
        {"subject_id": "s1", "label": 0, "fall_logit": 3.0, "reliability": 0.9},
        {"subject_id": "s2", "label": 1, "fall_logit": -2.0, "reliability": 0.8},
        {"subject_id": "s2", "label": 0, "fall_logit": 2.0, "reliability": 0.8},
    ])
    _write_records(outer, _outer_records())

    output = tmp_path / "release"
    result = evaluate_rg_pcnet_loso(inner, outer, output)

    assert result["selection"]["feasible"] is False
    assert result["promotion_checks"]["selection_feasible"] is False
    assert result["promoted"] is False
    assert result["reason"] == "continuous event evaluation is required"
    config = load_release_config(output / "release_config.json")
    assert 0.5 <= config.temperature <= 5.0
    assert 0.0 <= config.fall_threshold <= 1.0
    assert 0.0 <= config.reliability_threshold <= 1.0


def test_failed_publish_preserves_all_preexisting_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inner = tmp_path / "inner.jsonl"
    outer = tmp_path / "outer.jsonl"
    _write_records(inner, _inner_records())
    _write_records(outer, _outer_records())
    output = tmp_path / "release"
    output.mkdir()
    names = ("outer_predictions.jsonl", "calibration.json", "selection.json", "release_config.json", "evaluation.json")
    originals = {name: f"old-{name}".encode() for name in names}
    for name, payload in originals.items():
        (output / name).write_bytes(payload)

    import scripts.evaluate_rg_pcnet_loso as module
    original_write_json = module._write_json
    calls = 0

    def fail_on_second_json(path: Path, value: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write_json(path, value)
        raise OSError("injected publish failure")

    monkeypatch.setattr(module, "_write_json", fail_on_second_json)
    with pytest.raises(OSError, match="injected publish failure"):
        evaluate_rg_pcnet_loso(inner, outer, output)
    assert {name: (output / name).read_bytes() for name in names} == originals


def test_record_order_and_equivalent_json_spellings_are_canonicalized(tmp_path: Path) -> None:
    inner = tmp_path / "inner.jsonl"
    outer = tmp_path / "outer.jsonl"
    inner_rows = _inner_records()
    outer_rows = _outer_records()
    _write_records(inner, inner_rows)
    _write_records(outer, outer_rows)
    first = evaluate_rg_pcnet_loso(inner, outer, tmp_path / "first")
    first_bytes = {
        name: (tmp_path / "first" / name).read_bytes()
        for name in ("outer_predictions.jsonl", "calibration.json", "selection.json", "release_config.json", "evaluation.json")
    }

    # Same normalized records, deliberately shuffled and using the supported
    # ``logit`` alias plus integral/float spellings.
    shuffled_inner = [
        {"subject_id": "s3", "label": 0.0, "logit": -1.5, "reliability": 0.7},
        {"subject_id": "s2", "label": 0.0, "logit": -2.0, "reliability": 0.8},
        {"subject_id": "s1", "label": 0, "fall_logit": -3, "reliability": 0.9},
        {"subject_id": "s3", "label": 1, "fall_logit": 1.5, "reliability": 0.7},
        {"subject_id": "s2", "label": 1, "fall_logit": 2, "reliability": 0.8},
        {"subject_id": "s1", "label": 1.0, "logit": 3.0, "reliability": 0.9},
    ]
    shuffled_outer = [
        {"subject_id": "s4", "label": 0.0, "logit": -2.5, "reliability": 0.9},
        {"subject_id": "s4", "label": 1, "fall_logit": 2.5, "reliability": 0.9},
    ]
    _write_records(inner, shuffled_inner)
    _write_records(outer, shuffled_outer)
    second = evaluate_rg_pcnet_loso(inner, outer, tmp_path / "second")
    second_bytes = {
        name: (tmp_path / "second" / name).read_bytes()
        for name in first_bytes
    }
    assert first == second
    assert first_bytes == second_bytes
