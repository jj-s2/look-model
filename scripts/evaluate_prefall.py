"""Run group-isolated validation for a labelled pre-fall CSV dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk.prefall_evaluation import evaluate_subject_wise, should_promote


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--subject-column", default="subject_id")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--leave-one-subject-out", action="store_true")
    args = parser.parse_args()
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("pandas is required to evaluate a pre-fall CSV dataset.") from error
    data = pd.read_csv(args.input_csv)
    required = {args.label_column, args.subject_column}
    missing = required.difference(data.columns)
    if missing:
        parser.error(f"input is missing required columns: {sorted(missing)}")
    report = evaluate_subject_wise(
        data.drop(columns=[args.label_column, args.subject_column]), data[args.label_column],
        data[args.subject_column], threshold=args.threshold,
        leave_one_subject_out=args.leave_one_subject_out,
    )
    result = {"validation": report.as_dict(), "promoted": should_promote(
        report.metrics, {"f1": 0.90, "recall": 0.88})}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        raise SystemExit(f"Evaluation unavailable: {error}")
