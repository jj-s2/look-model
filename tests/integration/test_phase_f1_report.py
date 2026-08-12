import json
import sys
from pathlib import Path

import pytest

from scripts.generate_f1_report import generate_f1_report, main


@pytest.fixture
def experiment_data():
    return {
        "promoted": True,
        "temperature": 1.8,
        "threshold": 0.62,
        "fold_thresholds": [0.60, 0.63, 0.61],
        "inner_metrics": {
            "inner_mean_f1": 0.82,
            "inner_mean_recall": 0.78,
            "worst_subject_recall": 0.65,
            "subject_macro_f1": 0.81,
            "ece": 0.08,
            "brier": 0.12,
        },
        "confirmation_metrics": {
            "inner_mean_f1": 0.83,
            "inner_mean_recall": 0.79,
            "worst_subject_recall": 0.66,
            "subject_macro_f1": 0.82,
            "ece": 0.07,
            "brier": 0.11,
        },
        "candidate": {
            "inner_mean_f1": 0.82,
            "inner_mean_recall": 0.78,
            "worst_subject_recall": 0.65,
            "confirmation_f1": 0.83,
            "confirmation_recall": 0.79,
            "ece": 0.07,
            "brier": 0.11,
        },
        "baseline": {
            "inner_mean_f1": 0.75,
            "brier": 0.20,
        },
    }


def test_generate_f1_report_contains_required_sections(experiment_data):
    report = generate_f1_report(experiment_data)
    assert "PA-DTSF F1 优化实验报告" in report
    assert "temperature" in report
    assert "threshold" in report
    assert "promoted" in report
    assert "inner F1 增益" in report
    assert "confirmation F1" in report
    assert "证据边界" in report


def test_generate_f1_report_cli(tmp_path, experiment_data, monkeypatch):
    experiment_path = tmp_path / "experiment.json"
    experiment_path.write_text(json.dumps(experiment_data), encoding="utf-8")
    output_path = tmp_path / "report.md"

    monkeypatch.setattr(sys, "argv", ["generate_f1_report", "--experiment", str(experiment_path), "--output", str(output_path)])
    assert main() == 0
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "1.8" in content or "1.80" in content
    assert "0.62" in content
