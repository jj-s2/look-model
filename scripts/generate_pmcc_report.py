"""Render a compact PMCC evidence-boundary report from an evaluation artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _read(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError("metrics must be a JSON evaluation artifact")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("metrics must be valid JSON") from error
    if not isinstance(result, Mapping) or result.get("schema_version") != "pmcc.evaluation.v1":
        raise ValueError("metrics must be a pmcc.evaluation.v1 artifact")
    return result


def render(metrics: Mapping[str, Any]) -> str:
    provenance = metrics.get("provenance", {})
    rows = []
    def cell(value: Any) -> str:
        return "unavailable" if value is None else str(value)

    for model, result in metrics.get("model_rows", {}).items():
        for horizon, values in result.get("horizons", {}).items():
            rows.append(f"| {model} | {horizon} | {cell(values.get('auroc'))} | {cell(values.get('auprc'))} | {cell(values.get('brier'))} |")
    status = "eligible for research evaluation only" if metrics.get("release_metrics_eligible") else "not release eligible"
    return "\n".join((
        "# U-PMCC evaluation audit", "", f"Claim boundary: `{metrics.get('claim_boundary', 'research_only')}` — {status}.", "",
        "## Provenance", "", f"- Release ID: `{provenance.get('release_id', 'unavailable')}`", f"- Evidence tier: `{provenance.get('evidence_tier', 'unavailable')}`", "- Method: subject-grouped outer splits; calibration within training folds only.", "- Temporal chains are associations, not medical causality; this is not a clinical-performance claim.", "",
        "## Horizon metrics", "", "| Model | Horizon | AUROC | AUPRC | Brier |", "| --- | --- | ---: | ---: | ---: |", *rows, "",
        "## Evidence boundary", "", "Synthetic and offline artifacts remain `research_only` and must not be used as release evidence. Missing subgroup cells are reported as unavailable, never as zero.", "",
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a PMCC audit report.")
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(_read(args.metrics)), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(f"report generation failed: {error}")
