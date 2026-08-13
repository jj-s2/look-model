"""Train a pre-fall model only after subject-wise validation."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk.prefall_evaluation import evaluate_subject_wise, should_promote
from risk.prefall_model import PrefallModel


class NamedRows:
    """Minimal NumPy-compatible matrix retaining a strict feature schema."""

    def __init__(self, rows, columns):
        self._rows = [list(row) for row in rows]
        self.columns = tuple(columns)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]

    def __iter__(self):
        return iter(self._rows)

    def __array__(self, dtype=None):
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("NumPy is required for synthetic pre-fall smoke training.") from error
        return np.asarray(self._rows, dtype=dtype)

    def take_rows(self, indices):
        return NamedRows([self._rows[index] for index in indices], self.columns)


def _pandas():
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("pandas is required to load or create pre-fall training data.") from error
    return pd


def _synthetic_data():
    features, labels, subjects = [], [], []
    for subject in range(5):
        for observation in range(12):
            label = int(observation >= 8)
            subjects.append(f"synthetic-{subject}")
            labels.append(label)
            features.append([0.10 + label * 0.60 + subject * 0.002,
                             0.20 + label * 0.35 + observation * 0.001])
    return NamedRows(features, ("sway", "step_width")), labels, subjects


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--subject-column", default="subject_id")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dataset-summary", help="JSON audit emitted by the real-window dataset builder")
    parser.add_argument("--leave-one-subject-out", action="store_true",
                        help="evaluate each subject as a held-out fold")
    parser.add_argument("--estimator", choices=("logistic_regression", "extra_trees"),
                        default="logistic_regression")
    parser.add_argument("--synthetic-smoke-test", action="store_true")
    args = parser.parse_args()
    if bool(args.input_csv) == bool(args.synthetic_smoke_test):
        parser.error("provide exactly one of --input-csv or --synthetic-smoke-test")
    dataset_audit = None
    if args.dataset_summary:
        try:
            dataset_audit = json.loads(Path(args.dataset_summary).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read --dataset-summary: {error}")
        if not isinstance(dataset_audit, dict):
            parser.error("--dataset-summary must contain a JSON object")
    if args.synthetic_smoke_test:
        features, labels, subject_ids = _synthetic_data()
    else:
        data = _pandas().read_csv(args.input_csv)
        required = {args.label_column, args.subject_column}
        missing = required.difference(data.columns)
        if missing:
            parser.error(f"input is missing required columns: {sorted(missing)}")
        features = data.drop(columns=[args.label_column, args.subject_column])
        labels, subject_ids = data[args.label_column], data[args.subject_column]
    report = evaluate_subject_wise(features, labels, subject_ids, threshold=args.threshold,
                                   random_seed=args.random_seed,
                                   leave_one_subject_out=args.leave_one_subject_out,
                                   estimator_name=args.estimator)
    baseline = {"f1": 0.90, "recall": 0.88}
    promoted = should_promote(report.metrics, baseline)
    promotion_blockers: list[str] = []
    if dataset_audit is not None:
        guard_sec = dataset_audit.get("guard_sec")
        if not isinstance(guard_sec, (int, float)) or isinstance(guard_sec, bool) or guard_sec < 0.5:
            promotion_blockers.append("guard_sec below 0.5")
    if promotion_blockers:
        promoted = False
    provenance = {
        "synthetic": args.synthetic_smoke_test,
        "not_for_clinical_performance": args.synthetic_smoke_test,
        "source": "synthetic-smoke-test" if args.synthetic_smoke_test else str(args.input_csv),
        "fixture": "deterministic-two-feature-v1" if args.synthetic_smoke_test else None,
        "dataset_audit": dataset_audit,
        "promotion_blockers": promotion_blockers,
    }
    # Smoke fixtures prove wiring only. They can never become a clinical model.
    if args.synthetic_smoke_test:
        promoted = False
    model = PrefallModel(
        random_seed=args.random_seed, threshold=args.threshold, estimator_name=args.estimator,
    ).fit(features, labels)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "prefall_model.pkl").open("wb") as stream:
        pickle.dump(model, stream)
    metrics = {"validation": report.as_dict(), "baseline": baseline, "promoted": promoted,
               **provenance}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    card = {**model.metadata(metrics=report.metrics, promoted=promoted), **provenance}
    (output / "model_card.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        raise SystemExit(f"Training unavailable: {error}")
