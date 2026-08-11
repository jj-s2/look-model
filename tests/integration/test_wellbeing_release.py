import json
from pathlib import Path

import pytest

from scripts.evaluate_wellbeing_release import evaluate_release, load_release_config


CONFIG = Path(__file__).parents[2] / "configs" / "screening" / "pace_wb_v1.json"


def test_default_release_requires_elderly_external_validation_and_forbids_external_delivery(tmp_path: Path) -> None:
    result = evaluate_release(CONFIG, {"metrics": {}}, tmp_path / "release")
    assert result["promoted"] is False
    assert "elderly_external_validation_missing" in result["reasons"]
    assert result["delivery"]["wellbeing_external"] is False
    assert (tmp_path / "release" / "promotion_result.json").exists()


def test_release_config_rejects_invalid_thresholds_and_duplicate_keys(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["thresholds"]["min_auprc"] = 1.1
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="min_auprc"):
        load_release_config(invalid)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"pace-wb.release.v1","schema_version":"bad"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_release_config(duplicate)


def test_release_output_is_deterministic_and_always_unpromoted_without_old_age_data(tmp_path: Path) -> None:
    evidence = {"metrics": {"auprc": 0.95, "brier": 0.1, "ece": 0.05}, "elderly_external_validation": {"status": "missing"}}
    first = tmp_path / "one"
    second = tmp_path / "two"
    result_one = evaluate_release(CONFIG, evidence, first)
    result_two = evaluate_release(CONFIG, evidence, second)
    assert result_one == result_two
    assert (first / "promotion_result.json").read_bytes() == (second / "promotion_result.json").read_bytes()
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_research_shadow_cannot_become_promoted_even_with_good_metrics(tmp_path: Path) -> None:
    evidence = {
        "model_mode": "research_shadow",
        "metrics": {"auprc": 0.99, "brier": 0.01, "ece": 0.01, "false_invites_per_person_month": 0.0},
        "elderly_external_validation": {"status": "passed", "subject_count": 100},
    }
    result = evaluate_release(CONFIG, evidence, tmp_path / "release")
    assert result["promoted"] is False
    assert "model_mode_not_promotable" in result["reasons"]


def test_atomic_publish_restores_previous_release_on_replace_failure(tmp_path: Path, monkeypatch) -> None:
    import scripts.evaluate_wellbeing_release as release

    output = tmp_path / "release"
    evaluate_release(CONFIG, {"metrics": {}}, output)
    previous = (output / "promotion_result.json").read_bytes()
    real_replace = release.os.replace
    calls = {"count": 0}

    def fail_publish(source, destination):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(release.os, "replace", fail_publish)
    with pytest.raises(OSError, match="simulated"):
        evaluate_release(CONFIG, {"metrics": {"auprc": 0.1}}, output)
    assert (output / "promotion_result.json").read_bytes() == previous
