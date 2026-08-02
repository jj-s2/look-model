"""Build an auditable Markdown evaluation report and enforce release gates."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


MIN_FALL_F1 = 0.90
MIN_FALL_RECALL = 0.88
MAX_P95_LATENCY_SECONDS = 2.0
MAX_FALSE_ALARMS_PER_HOUR = 1.0


class ReleaseGateError(RuntimeError):
    """Raised when evidence does not meet the published release criteria."""


def _number(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        return None
    return float(value)


def release_gate_failures(metrics: Mapping[str, Any]) -> list[str]:
    """Return all missing or failed gates; missing evidence never passes."""
    requirements = (
        ("fall_f1", MIN_FALL_F1, "at least", "F1"),
        ("fall_recall", MIN_FALL_RECALL, "at least", "recall"),
        ("p95_latency_seconds", MAX_P95_LATENCY_SECONDS, "at most", "P95 latency"),
        ("false_alarms_per_hour", MAX_FALSE_ALARMS_PER_HOUR, "at most", "false alarms/hour"),
    )
    failures: list[str] = []
    for key, threshold, relation, label in requirements:
        value = _number(metrics, key)
        if value is None:
            failures.append(f"{label} evidence is missing")
        elif relation == "at least" and value < threshold:
            failures.append(f"{label} {value:.3f} is below {threshold:.3f}")
        elif relation == "at most" and value > threshold:
            failures.append(f"{label} {value:.3f} exceeds {threshold:.3f}")
    release_id = metrics.get("release_id")
    if not isinstance(release_id, str) or not release_id.strip():
        failures.append("release provenance is missing a release_id")
    if metrics.get("pipeline_scope") != "full_inference":
        failures.append("pipeline_scope must be full_inference")
    if metrics.get("benchmark_release_gate_passed") is not True:
        failures.append("benchmark release_gate_passed evidence is missing or false")
    return failures


def _render_report(metrics: Mapping[str, Any], failures: Sequence[str], sources: Sequence[Path] = ()) -> str:
    def show(key: str) -> str:
        value = metrics.get(key)
        return "unavailable" if value is None else str(value)

    gate = "PASS" if not failures else "FAIL"
    source_lines = "\n".join(f"- `{source.as_posix()}`" for source in sources) or "- No metrics files were supplied."
    failure_lines = "\n".join(f"- {reason}" for reason in failures) or "- All global gates are met by the supplied evidence."
    confusion = metrics.get("confusion_matrix", "unavailable")
    matrices = metrics.get("per_class_confusion_matrix")

    def matrix_row(label: str) -> str:
        matrix = matrices.get(label) if isinstance(matrices, Mapping) else None
        if not isinstance(matrix, Mapping):
            return f"| {label} | unavailable | unavailable | unavailable | unavailable |"
        return "| {label} | {tn} | {fp} | {fn} | {tp} |".format(
            label=label, tn=matrix.get("tn", "unavailable"), fp=matrix.get("fp", "unavailable"),
            fn=matrix.get("fn", "unavailable"), tp=matrix.get("tp", "unavailable"),
        )
    return f"""# Evaluation report

## Release gate: {gate}

This is an evidence report for a research and competition prototype. It does not
establish clinical performance and the wellbeing module is a screening/change
prompt only; it never produces a diagnosis.

### Global thresholds

| Metric | Observed | Threshold |
| --- | ---: | ---: |
| Fall F1 | {show('fall_f1')} | ≥ {MIN_FALL_F1:.2f} |
| Fall recall | {show('fall_recall')} | ≥ {MIN_FALL_RECALL:.2f} |
| End-to-end P95 latency (s) | {show('p95_latency_seconds')} | ≤ {MAX_P95_LATENCY_SECONDS:.1f} |
| False alarms/hour | {show('false_alarms_per_hour')} | ≤ {MAX_FALSE_ALARMS_PER_HOUR:.1f} |

### Gate findings

{failure_lines}

### Classification evidence

| Metric | Observed |
| --- | ---: |
| Fall precision | {show('fall_precision')} |
| Fall F1 | {show('fall_f1')} |
| Fall recall | {show('fall_recall')} |
| Confusion matrix (TN, FP, FN, TP) | {confusion} |
| Dataset | {show('dataset')} |
| Split strategy | {show('split_strategy')} |
| Decision threshold | {show('threshold')} |
| Random seed | {show('random_seed')} |

### Per-class confusion matrices

| Class | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
{matrix_row('fall')}
{matrix_row('adl')}

### Performance evidence

| Metric | Observed |
| --- | ---: |
| P50 latency (s) | {show('p50_latency_seconds')} |
| P95 latency (s) | {show('p95_latency_seconds')} |
| Measurement scope | {show('pipeline_scope')} |
| Hardware | {show('hardware')} |
| Python | {show('python_version')} |
| CUDA | {show('cuda_version')} |
| Model versions | {show('model_versions')} |

### Evidence files

{source_lines}

## Interpretation limits

- Passing requires subject-level evaluation and a full end-to-end measurement;
  a capture/decode-only replay benchmark is explicitly not release eligible.
- No connected C6c or SDNL1 result is claimed in this report unless a separately
  recorded live-device evidence file says so. Unavailable capability remains
  unavailable, never demo data.
- A release-gate failure is intentional and must block performance claims until
  the missing or failing evidence is rerun.
"""


def build_report(metrics: Mapping[str, Any]) -> str:
    """Validate a normalized metrics mapping and return a passing Markdown report."""
    failures = release_gate_failures(metrics)
    if failures:
        raise ReleaseGateError("release gate failed: " + "; ".join(failures))
    return _render_report(metrics, ())


def _extract_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    overall = payload.get("overall_argmax")
    source: Mapping[str, Any] = overall if isinstance(overall, Mapping) else payload
    aliases = {"fall_precision": "fall_precision", "fall_f1": "fall_f1", "fall_recall": "fall_recall"}
    for source_key, target_key in aliases.items():
        if source_key in source and source[source_key] is not None:
            result[target_key] = source[source_key]
    if all(key in source for key in ("tn", "fp", "fn", "tp")):
        result["confusion_matrix"] = tuple(source[key] for key in ("tn", "fp", "fn", "tp"))
        result["per_class_confusion_matrix"] = {
            "fall": {"tn": source["tn"], "fp": source["fp"], "fn": source["fn"], "tp": source["tp"]},
            "adl": {"tn": source["tp"], "fp": source["fn"], "fn": source["fp"], "tp": source["tn"]},
        }
    for key in (
        "p50_latency_seconds", "p95_latency_seconds", "false_alarms_per_hour",
        "pipeline_scope", "hardware", "python_version", "cuda_version", "model_versions",
        "dataset", "threshold", "random_seed", "split_strategy", "per_class_confusion_matrix",
    ):
        if key in payload and payload[key] is not None:
            result[key] = payload[key]
    if "split_strategy" not in result and "overall_argmax" in payload:
        result["split_strategy"] = "leave-one-subject-out (reported source)"
    return result


def _expand_metric_files(metrics_files: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for candidate in metrics_files:
        if candidate.is_dir():
            files.extend(sorted(path for path in candidate.rglob("*.json") if path.is_file()))
        elif candidate.is_file() and candidate.suffix.lower() == ".json":
            files.append(candidate)
    return files


def _collect_release_metrics(metrics_files: Sequence[Path]) -> tuple[dict[str, Any], list[Path], list[str]]:
    """Accept exactly one classification and one benchmark artifact for one release.

    Aggregating arbitrary JSON would allow unrelated results to manufacture a
    passing report.  These two typed artifacts must declare the same release ID.
    """
    classifications: list[tuple[Path, Mapping[str, Any]]] = []
    benchmarks: list[tuple[Path, Mapping[str, Any]]] = []
    for path in _expand_metric_files(metrics_files):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        if payload.get("kind") == "classification_evaluation":
            classifications.append((path, payload))
        elif payload.get("kind") == "pipeline_benchmark":
            benchmarks.append((path, payload))

    failures: list[str] = []
    if len(classifications) != 1:
        failures.append("release provenance requires exactly one classification_evaluation artifact")
    if len(benchmarks) != 1:
        failures.append("release provenance requires exactly one pipeline_benchmark artifact")
    if failures:
        return {}, [], failures

    classification_path, classification = classifications[0]
    benchmark_path, benchmark = benchmarks[0]
    classification_id = classification.get("release_id")
    benchmark_id = benchmark.get("release_id")
    if not isinstance(classification_id, str) or not classification_id.strip():
        failures.append("classification release provenance is missing release_id")
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        failures.append("benchmark release provenance is missing release_id")
    if not failures and classification_id != benchmark_id:
        failures.append("classification and benchmark release_id values do not match")
    if failures:
        return {}, [classification_path, benchmark_path], failures

    merged = _extract_metrics(classification)
    merged.update(_extract_metrics(benchmark))
    merged["release_id"] = classification_id
    merged["pipeline_scope"] = benchmark.get("pipeline_scope")
    merged["benchmark_release_gate_passed"] = benchmark.get("release_gate_passed")
    return merged, [classification_path, benchmark_path], []


def generate_evaluation_report(metrics_files: Sequence[Path]) -> str:
    """Load metrics JSON files, enforce the gates, and return Markdown text."""
    merged, valid_files, provenance_failures = _collect_release_metrics(metrics_files)
    failures = [*provenance_failures, *release_gate_failures(merged)]
    if failures:
        raise ReleaseGateError("release gate failed: " + "; ".join(failures))
    return _render_report(merged, (), valid_files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path, nargs="+", help="JSON files or directories")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    merged, valid_files, provenance_failures = _collect_release_metrics(args.metrics)
    failures = [*provenance_failures, *release_gate_failures(merged)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_report(merged, failures, valid_files), encoding="utf-8")
    if failures:
        print("release gate: FAIL - " + "; ".join(failures), file=sys.stderr)
        return 1
    print(f"release gate: PASS - report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
