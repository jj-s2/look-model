"""End-to-end PA-DTSF F1-optimization experiment.

Given a trained checkpoint, the script:

1. Runs the inner 3-fold threshold search on the validation split.
2. Evaluates the calibrated model on the test split.
3. Checks the F1 promotion gate against a baseline.
4. If promoted, writes a release bundle with calibration metadata.

When the split_manifest contains ``folds`` (e.g. LOOCV), ``--fold`` selects
which fold's train/validation/test partition to use.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk.phase_model.calibration import Calibrator
from risk.phase_model.evaluation import evaluate_fall_event
from risk.phase_model.release import build_release_bundle, write_release_bundle
from risk.phase_model.selection import candidate_passes
from scripts.optimize_phase_threshold import (
    _load_split_logits,
    _load_validation_logits,
    inner_threshold_search,
)


def _load_config(config_path: Path | None) -> object:
    if config_path is None:
        from configs.skeleton import padtsf_v1_f1 as config

        return config
    import importlib.util

    spec = importlib.util.spec_from_file_location("user_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load config from {config_path}")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def evaluate_and_promote(
    *,
    validation_logits: Sequence[float],
    validation_labels: Sequence[int],
    validation_subjects: Sequence[str],
    test_logits: Sequence[float],
    test_labels: Sequence[int],
    test_subjects: Sequence[str],
    baseline: Mapping[str, object],
    config: object,
) -> dict[str, object]:
    """Run threshold search, test evaluation and the promotion gate."""
    if (
        len(validation_logits) == 0
        or len(validation_logits) != len(validation_labels)
        or len(validation_logits) != len(validation_subjects)
    ):
        raise ValueError(
            "validation_logits, validation_labels, and validation_subjects "
            "must be non-empty and have the same length"
        )
    if (
        len(test_logits) == 0
        or len(test_logits) != len(test_labels)
        or len(test_logits) != len(test_subjects)
    ):
        raise ValueError(
            "test_logits, test_labels, and test_subjects must be non-empty and have the same length"
        )
    if len({int(value) for value in test_labels}) < 2:
        return {
            "promoted": False,
            "reason": "confirmation fold requires both classes",
            "temperature": 1.0,
            "threshold": None,
            "inner_metrics": None,
            "confirmation_metrics": None,
            "candidate": None,
            "baseline": dict(baseline),
            "fold_thresholds": None,
            "fold_temperatures": None,
        }

    seed = int(getattr(config, "SEED", 42))
    try:
        result = inner_threshold_search(
            validation_logits,
            validation_labels,
            validation_subjects,
            n_folds=int(getattr(config, "INNER_FOLDS", 3)),
            recall_floor=float(getattr(config, "INNER_RECALL_FLOOR", 0.75)),
            min_recall_per_subject=float(getattr(config, "INNER_MIN_RECALL_PER_SUBJECT", 0.60)),
            aggregation=str(getattr(config, "INNER_THRESHOLD_AGGREGATION", "mean")),
            seed=seed,
        )
    except ValueError as error:
        if str(error) != "no threshold satisfies recall_floor":
            raise
        return {
            "promoted": False,
            "reason": str(error),
            "temperature": 1.0,
            "threshold": None,
            "inner_metrics": None,
            "confirmation_metrics": None,
            "candidate": None,
            "baseline": dict(baseline),
            "fold_thresholds": None,
            "fold_temperatures": None,
        }
    temperature = float(result["temperature"])
    threshold = float(result["threshold"])
    calibrator = Calibrator(temperature=temperature, threshold=threshold)
    test_probabilities = calibrator.calibrate(list(test_logits))

    inner_metrics = evaluate_fall_event(
        list(validation_labels),
        calibrator.calibrate(list(validation_logits)),
        list(validation_subjects),
        threshold,
    )
    confirmation_metrics = evaluate_fall_event(
        list(test_labels), test_probabilities, list(test_subjects), threshold
    )

    candidate = {
        "inner_mean_f1": float(inner_metrics["inner_mean_f1"]),
        "inner_mean_recall": float(inner_metrics["inner_mean_recall"]),
        "worst_subject_recall": float(inner_metrics["worst_subject_recall"]),
        "confirmation_f1": float(confirmation_metrics["inner_mean_f1"]),
        "confirmation_recall": float(confirmation_metrics["inner_mean_recall"]),
        "ece": float(confirmation_metrics["ece"]),
        "brier": float(confirmation_metrics["brier"]),
    }
    promoted = candidate_passes(candidate, baseline)

    return {
        "promoted": promoted,
        "temperature": temperature,
        "threshold": threshold,
        "inner_metrics": inner_metrics,
        "confirmation_metrics": confirmation_metrics,
        "candidate": candidate,
        "baseline": dict(baseline),
        "fold_thresholds": result.get("fold_thresholds"),
        "fold_temperatures": result.get("fold_temperatures"),
    }


def run_phase_f1_experiment(
    *,
    checkpoint: Path,
    dataset_lock: Path,
    split_manifest: Path,
    output_dir: Path,
    baseline: Mapping[str, object],
    data_root: Path | None = None,
    config_path: Path | None = None,
    release_id: str | None = None,
    fold: str | None = None,
) -> dict[str, object]:
    """Run the full experiment pipeline and optionally emit a promoted release."""

    config = _load_config(config_path)
    seed = int(getattr(config, "SEED", 42))
    random.seed(seed)
    np.random.seed(seed)

    root = data_root or Path.cwd()
    validation_logits, validation_labels, validation_subjects = _load_validation_logits(
        checkpoint, dataset_lock, split_manifest, root, config, fold=fold
    )
    test_logits, test_labels, test_subjects = _load_split_logits(
        checkpoint, dataset_lock, split_manifest, root, config, "test", fold=fold
    )

    experiment = evaluate_and_promote(
        validation_logits=list(validation_logits),
        validation_labels=list(validation_labels),
        validation_subjects=list(validation_subjects),
        test_logits=list(test_logits),
        test_labels=list(test_labels),
        test_subjects=list(test_subjects),
        baseline=baseline,
        config=config,
    )
    experiment["fold"] = fold

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if experiment["promoted"]:
        metrics = {
            "release_id": release_id or "padtsf-f1-promoted",
            "promoted": True,
            **experiment["candidate"],
        }
        calibration = {
            "temperature": experiment["temperature"],
            "threshold": experiment["threshold"],
        }
        bundle = build_release_bundle(
            release_id or "padtsf-f1-promoted",
            checkpoint,
            output_dir,
            inputs={
                "dataset_lock": json.loads(dataset_lock.read_text(encoding="utf-8")),
                "split_manifest": json.loads(split_manifest.read_text(encoding="utf-8")),
                "metrics": metrics,
                "calibration": calibration,
            },
        )
        release_dir = output_dir / "release"
        write_release_bundle(bundle, release_dir)
        experiment["release_dir"] = str(release_dir)

    return experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-lock", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path, help="JSON file with baseline metrics")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--release-id", default=None)
    parser.add_argument("--fold", type=str, default=None)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    run_phase_f1_experiment(
        checkpoint=args.checkpoint,
        dataset_lock=args.dataset_lock,
        split_manifest=args.split_manifest,
        output_dir=args.output,
        baseline=baseline,
        data_root=args.data_root,
        config_path=args.config,
        release_id=args.release_id,
        fold=args.fold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
