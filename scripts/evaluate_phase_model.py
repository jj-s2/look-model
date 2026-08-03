"""Evaluate frozen prediction JSONL without fitting or changing thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from risk.phase_model.evaluation import evaluate_predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=.5)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = evaluate_predictions(records, threshold=args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metrics": result.metrics, "per_subject": result.per_subject, "per_dataset": result.per_dataset}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
