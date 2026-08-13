from __future__ import annotations

from pathlib import Path

from scripts.plot_phase_results import build_parser, generate_artifacts, summarize_predictions


def _records() -> list[dict[str, object]]:
    return [
        {"clip_id": "a", "subject_id": "s1", "dataset": "GMDCSA24", "label": 1, "score": 0.95},
        {"clip_id": "b", "subject_id": "s1", "dataset": "GMDCSA24", "label": 1, "score": 0.80},
        {"clip_id": "c", "subject_id": "s1", "dataset": "GMDCSA24", "label": 0, "score": 0.20},
        {"clip_id": "d", "subject_id": "s1", "dataset": "GMDCSA24", "label": 0, "score": 0.05},
    ]


def test_phase_plot_parser_accepts_release_paths():
    args = build_parser().parse_args(["--checkpoint", "checkpoint.pt", "--output-dir", "plots"])
    assert args.checkpoint == Path("checkpoint.pt")
    assert args.output_dir == Path("plots")


def test_phase_plot_summary_contains_binary_metrics():
    summary = summarize_predictions(_records())
    assert summary["samples"] == 4
    assert summary["positive_samples"] == 2
    assert summary["roc_auc"] == 1.0
    assert summary["f1_at_threshold"] == 1.0
    assert sum(row["count"] for row in summary["calibration_bins"]) == 4


def test_phase_plot_artifacts_are_written(tmp_path: Path):
    paths = generate_artifacts(_records(), tmp_path, metadata={"release_id": "test"})
    expected = {"metrics.json", "predictions.jsonl", "report.md", "roc_pr_curves.png", "confusion_matrix.png", "threshold_curves.png", "calibration.png", "score_distribution.png"}
    assert expected.issubset({path.name for path in paths.values()})
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
