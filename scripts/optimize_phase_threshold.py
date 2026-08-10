"""Inner threshold search for PA-DTSF fall-event logits.

Given a trained checkpoint or release, the script:

1. Loads the model.
2. Collects logits and labels for the validation split.
3. Runs a subject-level 3-fold inner loop: each fold fits a temperature
   on its subjects and selects a threshold.
4. Aggregates the per-fold thresholds (mean or median) and writes the
   optimized threshold and temperature to disk.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk.phase_model.selection import apply_temperature, fit_temperature, metrics_at_threshold


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


def _subject_stratified_folds(
    subjects: Sequence[str], n_folds: int, seed: int
) -> list[list[int]]:
    """Return ``n_folds`` index lists, stratified by unique subjects.

    Each list contains the sample indices belonging to one fold.  Subjects
    are shuffled and distributed round-robin so that all folds are
    subject-disjoint.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    unique_subjects = sorted({str(subject) for subject in subjects})
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)
    subject_to_fold = {
        subject: fold_index % n_folds for fold_index, subject in enumerate(unique_subjects)
    }
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for sample_index, subject in enumerate(subjects):
        folds[subject_to_fold[str(subject)]].append(sample_index)
    return folds


def _validate_probability_constraint(name: str, value: object) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite and in [0, 1]") from None
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return normalized


def _validate_search_constraints(
    recall_floor: object,
    min_recall_per_subject: object,
    fpr_ceiling: object | None,
) -> tuple[float, float, float | None]:
    return (
        _validate_probability_constraint("recall_floor", recall_floor),
        _validate_probability_constraint("min_recall_per_subject", min_recall_per_subject),
        None
        if fpr_ceiling is None
        else _validate_probability_constraint("fpr_ceiling", fpr_ceiling),
    )


def _choose_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    subjects: Sequence[str],
    *,
    recall_floor: float = 0.75,
    min_recall_per_subject: float = 0.60,
    fpr_ceiling: float | None = None,
    lower: float = 0.20,
    upper: float = 0.80,
    step: float = 0.01,
) -> float:
    """Select a threshold subject-aware, preserving recall floor."""
    recall_floor, min_recall_per_subject, fpr_ceiling = _validate_search_constraints(
        recall_floor, min_recall_per_subject, fpr_ceiling
    )
    lower = _validate_probability_constraint("lower", lower)
    upper = _validate_probability_constraint("upper", upper)
    try:
        step = float(step)
    except (TypeError, ValueError):
        raise ValueError("step must be positive and finite") from None
    if lower > upper:
        raise ValueError("lower must not exceed upper")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("step must be positive and finite")

    grouped: dict[str, tuple[list[int], list[float]]] = {}
    for subject, label, score in zip(subjects, labels, scores):
        group_labels, group_scores = grouped.setdefault(subject, ([], []))
        group_labels.append(int(label))
        group_scores.append(float(score))

    best_threshold = None
    best_score: tuple[float, float, float] = (-1.0, -1.0, float("inf"))
    candidate_count = int(math.floor((upper - lower) / step + 1e-10)) + 1
    for index in range(candidate_count):
        threshold = round(lower + index * step, 12)
        if threshold > upper + 1e-12:
            continue
        aggregate = metrics_at_threshold(labels, scores, threshold)
        if aggregate.recall + 1e-12 < recall_floor:
            continue
        if fpr_ceiling is not None and aggregate.false_positive_rate > fpr_ceiling + 1e-12:
            continue
        subject_metrics = [
            metrics_at_threshold(group_labels, group_scores, threshold)
            for group_labels, group_scores in grouped.values()
        ]
        if any(metric.recall + 1e-12 < min_recall_per_subject for metric in subject_metrics):
            continue
        subject_macro_f1 = sum(metric.f1 for metric in subject_metrics) / len(subject_metrics)
        worst_subject_f1 = min(metric.f1 for metric in subject_metrics)
        candidate_score = (subject_macro_f1, worst_subject_f1, -aggregate.false_positive_rate)
        if candidate_score > best_score:
            best_score = candidate_score
            best_threshold = threshold

    if best_threshold is None:
        raise ValueError("no threshold satisfies recall_floor")
    return best_threshold


def _build_records(
    logits: Sequence[float],
    labels: Sequence[int],
    subjects: Sequence[str],
    *,
    n_folds: int = 3,
    recall_floor: float = 0.75,
    min_recall_per_subject: float = 0.60,
    fpr_ceiling: float | None = None,
    aggregation: str = "mean",
    seed: int = 42,
) -> dict[str, object]:
    """Select a threshold via subject-stratified cross-validation.

    If fewer unique subjects than ``n_folds`` are available, the function
    falls back to fitting one temperature and threshold on the whole
    validation set (no inner fold).
    """
    if len(logits) == 0 or len(logits) != len(labels) or len(logits) != len(subjects):
        raise ValueError("logits, labels, and subjects must be non-empty and have the same length")
    recall_floor, min_recall_per_subject, fpr_ceiling = _validate_search_constraints(
        recall_floor, min_recall_per_subject, fpr_ceiling
    )
    if aggregation not in {"mean", "median"}:
        raise ValueError(f"unsupported aggregation: {aggregation}")

    unique_subjects = sorted({str(subject) for subject in subjects})
    if len(unique_subjects) < n_folds:
        temperature = fit_temperature(logits, labels)
        probabilities = apply_temperature(logits, temperature)
        threshold = _choose_threshold(
            probabilities,
            labels,
            subjects,
            recall_floor=recall_floor,
            min_recall_per_subject=min_recall_per_subject,
            fpr_ceiling=fpr_ceiling,
        )
        return {
            "threshold": float(threshold),
            "temperature": float(temperature),
            "fold_thresholds": [float(threshold)],
            "fold_temperatures": [float(temperature)],
            "aggregation": aggregation,
            "n_folds": n_folds,
            "recall_floor": recall_floor,
            "min_recall_per_subject": min_recall_per_subject,
        }

    folds = _subject_stratified_folds(subjects, n_folds, seed)
    fold_thresholds: list[float] = []
    fold_temperatures: list[float] = []

    for fold_indices in folds:
        if not fold_indices:
            continue
        fold_mask = np.zeros(len(logits), dtype=bool)
        fold_mask[fold_indices] = True
        train_logits = np.asarray(logits)[~fold_mask]
        train_labels = np.asarray(labels)[~fold_mask]
        val_logits = np.asarray(logits)[fold_mask]
        val_labels = np.asarray(labels)[fold_mask]
        val_subjects = np.asarray(subjects, dtype=object)[fold_mask]

        temperature = fit_temperature(train_logits, train_labels)
        val_probs = apply_temperature(val_logits, temperature)
        threshold = _choose_threshold(
            val_probs,
            val_labels,
            val_subjects,
            recall_floor=recall_floor,
            min_recall_per_subject=min_recall_per_subject,
            fpr_ceiling=fpr_ceiling,
        )
        fold_thresholds.append(float(threshold))
        fold_temperatures.append(float(temperature))

    if not fold_thresholds:
        raise RuntimeError("no thresholds were selected")

    aggregated_threshold = float(np.mean(fold_thresholds)) if aggregation == "mean" else float(np.median(fold_thresholds))
    aggregated_temperature = float(np.mean(fold_temperatures)) if aggregation == "mean" else float(np.median(fold_temperatures))

    return {
        "threshold": aggregated_threshold,
        "temperature": aggregated_temperature,
        "fold_thresholds": fold_thresholds,
        "fold_temperatures": fold_temperatures,
        "aggregation": aggregation,
        "n_folds": n_folds,
        "recall_floor": recall_floor,
        "min_recall_per_subject": min_recall_per_subject,
    }


def _resolve_feature_path(root: Path, clip: Mapping[str, object]) -> Path:
    media_path = Path(str(clip["media_path"]))
    candidates = [root / media_path]
    dataset = str(clip.get("dataset") or "")
    if dataset:
        candidates.append(root / dataset / media_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _resolve_split(split_manifest: dict, split_name: str, fold: str | None = None) -> list:
    """Return the list of clip ids for a split, supporting plain and fold manifests."""
    if "folds" in split_manifest:
        if fold is None:
            raise ValueError("split_manifest contains folds; a fold must be specified")
        if fold not in split_manifest["folds"]:
            raise ValueError(f"fold {fold} not found; available: {sorted(split_manifest['folds'].keys())}")
        return list(split_manifest["folds"][fold].get(split_name, []))
    return list(split_manifest.get("partitions", {}).get(split_name, []))


def _load_split_logits(
    checkpoint_path: Path,
    dataset_lock_path: Path,
    split_manifest_path: Path,
    data_root: Path,
    config: object,
    split_name: str,
    fold: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return logits, labels and subjects for a named split partition."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required to compute validation logits") from error

    from risk.phase_model.model import PhaseAwareFusionModel
    from risk.phase_model.normalization import normalize_pose_array

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseAwareFusionModel(
        short_dim=int(getattr(config, "SHORT_DIM", 512)),
        joints=int(getattr(config, "JOINTS", 17)),
        hidden_dim=int(getattr(config, "HIDDEN_DIM", 128)),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    lock = json.loads(dataset_lock_path.read_text(encoding="utf-8"))
    split = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    clips = {str(item["clip_id"]): item for item in lock.get("clips", []) if isinstance(item, Mapping) and "clip_id" in item}
    split_ids = set(str(value) for value in _resolve_split(split, split_name, fold))

    logits: list[float] = []
    labels: list[int] = []
    subjects: list[str] = []

    with torch.no_grad():
        for clip_id in split_ids:
            clip = clips.get(str(clip_id))
            if clip is None:
                continue
            path = _resolve_feature_path(data_root, clip)
            if path.suffix.lower() != ".npz":
                continue
            cache = np.load(path, allow_pickle=False)
            long_pose = normalize_pose_array(cache["long_pose"])
            short = np.asarray(cache["short_embedding"], dtype=np.float32)
            short_tensor = torch.as_tensor(short, dtype=torch.float32, device=device).reshape(1, -1)
            long_tensor = torch.as_tensor(long_pose, dtype=torch.float32, device=device).reshape(1, 64, 17, 3)
            output = model(short_tensor, long_tensor, torch.zeros(1, device=device), torch.ones(1, device=device))
            fall_logit = float(output.fall_event_logit.squeeze().cpu())
            logits.append(fall_logit)
            labels.append(1 if str(clip.get("coarse_event")) == "fall" else 0)
            subjects.append(str(clip.get("subject_id", "unknown")))

    if not logits:
        raise RuntimeError(f"no {split_name} samples produced logits")
    return np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.int64), np.asarray(subjects, dtype=object)


def _load_validation_logits(
    checkpoint_path: Path,
    dataset_lock_path: Path,
    split_manifest_path: Path,
    data_root: Path,
    config: object,
    fold: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return validation logits, labels and subjects for a checkpoint."""
    return _load_split_logits(
        checkpoint_path, dataset_lock_path, split_manifest_path, data_root, config, "validation", fold=fold
    )


def inner_threshold_search(
    logits: Sequence[float],
    labels: Sequence[int],
    subjects: Sequence[str],
    *,
    n_folds: int = 3,
    recall_floor: float = 0.75,
    min_recall_per_subject: float = 0.60,
    fpr_ceiling: float | None = None,
    aggregation: str = "mean",
    seed: int = 42,
) -> dict[str, object]:
    """Run the inner threshold-search loop and return a serializable result."""
    if len(logits) == 0 or len(logits) != len(labels) or len(logits) != len(subjects):
        raise ValueError("logits, labels, and subjects must be non-empty and have the same length")
    recall_floor, min_recall_per_subject, fpr_ceiling = _validate_search_constraints(
        recall_floor, min_recall_per_subject, fpr_ceiling
    )
    return _build_records(
        logits,
        labels,
        subjects,
        n_folds=n_folds,
        recall_floor=recall_floor,
        min_recall_per_subject=min_recall_per_subject,
        fpr_ceiling=fpr_ceiling,
        aggregation=aggregation,
        seed=seed,
    )


def optimize_phase_threshold(
    *,
    checkpoint: Path,
    dataset_lock: Path,
    split_manifest: Path,
    output_dir: Path,
    data_root: Path | None = None,
    config_path: Path | None = None,
    release_id: str | None = None,
    fold: str | None = None,
) -> dict[str, object]:
    """Run the full inner threshold-search pipeline and write results."""
    config = _load_config(config_path)
    seed = int(getattr(config, "SEED", 42))
    random.seed(seed)
    np.random.seed(seed)

    root = data_root or Path.cwd()
    logits, labels, subjects = _load_validation_logits(
        checkpoint, dataset_lock, split_manifest, root, config, fold=fold
    )

    result = inner_threshold_search(
        logits,
        labels,
        subjects,
        n_folds=int(getattr(config, "INNER_FOLDS", 3)),
        recall_floor=float(getattr(config, "INNER_RECALL_FLOOR", 0.75)),
        min_recall_per_subject=float(getattr(config, "INNER_MIN_RECALL_PER_SUBJECT", 0.60)),
        aggregation=str(getattr(config, "INNER_THRESHOLD_AGGREGATION", "mean")),
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "release_id": release_id or "padtsf-threshold-search",
        "checkpoint": str(checkpoint),
        "fold": fold,
        "temperature": result["temperature"],
        "threshold": result["threshold"],
        "fold_thresholds": result["fold_thresholds"],
        "fold_temperatures": result["fold_temperatures"],
    }
    (output_dir / "threshold_selection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-lock", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--release-id", default=None)
    parser.add_argument("--fold", type=str, default=None)
    args = parser.parse_args()

    optimize_phase_threshold(
        checkpoint=args.checkpoint,
        dataset_lock=args.dataset_lock,
        split_manifest=args.split_manifest,
        output_dir=args.output,
        data_root=args.data_root,
        config_path=args.config,
        release_id=args.release_id,
        fold=args.fold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
