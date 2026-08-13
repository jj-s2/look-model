import hashlib
import json
import math
import warnings
from pathlib import Path

import pytest

import scripts.plot_rg_pcnet_results as plotting
from scripts.plot_rg_pcnet_results import plot_rg_pcnet_results


def _write_inputs(root: Path, *, shuffled: bool = False) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "subject_count": 3,
        "subject_f1": {"s3": 0.70, "s1": 0.90, "s2": 0.80},
    }
    continuous = {
        "event_recall": 0.90,
        "false_alerts_per_hour": 0.10,
        "median_delay_seconds": 1.5,
        "coverage": 0.88,
        "subject_count": 3,
    }
    calibration = {
        "risk_coverage": {
            "clean": {"coverage": [0.0, 0.5, 1.0], "risk": [0.0, 0.1, 0.2]},
            "corrupted": {"coverage": [0.0, 0.5, 1.0], "risk": [0.0, 0.2, 0.4]},
        }
    }
    ablation = {
        "models": {
            "full_rg_pcnet": {"f1": 0.91},
            "baseline_tcn": {"f1": 0.76},
            "phase_only": {"f1": 0.80},
            "reliability_only": {"f1": 0.84},
            "distillation": {"f1": 0.88},
        }
    }
    values = {"aggregate": aggregate, "continuous": continuous, "calibration": calibration, "ablation": ablation}
    paths = {}
    for name, value in values.items():
        path = root / f"{name}.json"
        if shuffled:
            # Deliberately use a different insertion order while preserving values.
            value = json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths[name] = path
    return paths


def test_plot_manifest_hashes_every_metric_source(tmp_path):
    inputs = _write_inputs(tmp_path)
    result = plot_rg_pcnet_results(
        inputs["aggregate"], inputs["continuous"], inputs["calibration"], inputs["ablation"], tmp_path / "out"
    )
    manifest = json.loads((tmp_path / "out" / "plot_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["figures"]) == {"model_comparison", "subject_f1", "risk_coverage", "continuous_events"}
    assert all(len(item["source_sha256"]) == 64 for item in manifest["figures"].values())
    assert manifest["figures"]["subject_f1"]["source_sha256"] == hashlib.sha256(inputs["aggregate"].read_bytes()).hexdigest()
    assert all(Path(path).exists() for path in result.values())
    assert all(Path(item[key]).exists() for item in manifest["figures"].values() for key in ("png", "svg"))
    assert all("subject count:" in item["caption"] and "intervals: descriptive" in item["caption"] for item in manifest["figures"].values())


def test_each_plot_has_png_and_vector_svg(tmp_path):
    inputs = _write_inputs(tmp_path)
    result = plot_rg_pcnet_results(inputs["aggregate"], inputs["continuous"], inputs["calibration"], inputs["ablation"], tmp_path / "out")
    for name in ("model_comparison", "subject_f1", "risk_coverage", "continuous_events"):
        png = result[f"{name}_png"]
        svg = result[f"{name}_svg"]
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert svg.read_text(encoding="utf-8").lstrip().startswith("<?xml")


@pytest.mark.parametrize(
    ("source", "mutation"),
    [
        ("aggregate", lambda value: value["subject_f1"].pop("s1")),
        ("continuous", lambda value: value.pop("coverage")),
        ("calibration", lambda value: value["risk_coverage"]["clean"].update({"risk": [0.0, math.nan, 0.2]})),
        ("ablation", lambda value: value["models"].pop("full_rg_pcnet")),
    ],
)
def test_missing_or_nonfinite_metric_is_rejected(tmp_path, source, mutation):
    inputs = _write_inputs(tmp_path)
    value = json.loads(inputs[source].read_text(encoding="utf-8"))
    mutation(value)
    inputs[source].write_text(json.dumps(value, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError):
        plot_rg_pcnet_results(inputs["aggregate"], inputs["continuous"], inputs["calibration"], inputs["ablation"], tmp_path / "out")


def test_input_key_order_does_not_change_plot_data(tmp_path):
    first = _write_inputs(tmp_path / "first")
    second = _write_inputs(tmp_path / "second", shuffled=True)
    first_result = plot_rg_pcnet_results(first["aggregate"], first["continuous"], first["calibration"], first["ablation"], tmp_path / "out1")
    second_result = plot_rg_pcnet_results(second["aggregate"], second["continuous"], second["calibration"], second["ablation"], tmp_path / "out2")
    first_manifest = json.loads(first_result["plot_manifest"].read_text(encoding="utf-8"))
    second_manifest = json.loads(second_result["plot_manifest"].read_text(encoding="utf-8"))
    for name in first_manifest["figures"]:
        assert first_manifest["figures"][name]["caption"] == second_manifest["figures"][name]["caption"]


def test_negative_median_delay_is_valid_but_negative_false_alert_rate_is_rejected(tmp_path):
    inputs = _write_inputs(tmp_path)
    continuous = json.loads(inputs["continuous"].read_text(encoding="utf-8"))
    continuous["median_delay_seconds"] = -2.0
    inputs["continuous"].write_text(json.dumps(continuous), encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = plot_rg_pcnet_results(inputs["aggregate"], inputs["continuous"], inputs["calibration"], inputs["ablation"], tmp_path / "valid")
    assert not any("Glyph" in str(item.message) for item in caught)
    assert result["continuous_events_png"].exists()

    continuous["false_alerts_per_hour"] = -1.0
    inputs["continuous"].write_text(json.dumps(continuous), encoding="utf-8")
    with pytest.raises(ValueError):
        plot_rg_pcnet_results(inputs["aggregate"], inputs["continuous"], inputs["calibration"], inputs["ablation"], tmp_path / "invalid")


def test_plot_failure_rolls_back_new_output_and_preserves_existing_output(tmp_path, monkeypatch):
    inputs = _write_inputs(tmp_path / "inputs")
    original = plotting._save_figure
    calls = 0

    def fail_on_second(fig, output_dir, name):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected figure failure")
        return original(fig, output_dir, name)

    monkeypatch.setattr(plotting, "_save_figure", fail_on_second)
    new_output = tmp_path / "new-output"
    with pytest.raises(OSError, match="injected figure failure"):
        plot_rg_pcnet_results(inputs["aggregate"], inputs["continuous"], inputs["calibration"], inputs["ablation"], new_output)
    assert not new_output.exists()
    assert not list(tmp_path.glob(".new-output.*"))

    existing = tmp_path / "existing-output"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("old bytes", encoding="utf-8")
    calls = 0
    with pytest.raises(OSError, match="injected figure failure"):
        plot_rg_pcnet_results(inputs["aggregate"], inputs["continuous"], inputs["calibration"], inputs["ablation"], existing)
    assert sentinel.read_text(encoding="utf-8") == "old bytes"
    assert not list(tmp_path.glob(".existing-output.*"))


def test_manifest_records_font_selection_and_fallback_state(tmp_path):
    inputs = _write_inputs(tmp_path)
    output = plot_rg_pcnet_results(inputs["aggregate"], inputs["continuous"], inputs["calibration"], inputs["ablation"], tmp_path / "out")
    manifest = json.loads(output["plot_manifest"].read_text(encoding="utf-8"))
    assert isinstance(manifest["font"]["family"], str) and manifest["font"]["family"]
    assert type(manifest["font"]["fallback"]) is bool
