"""Run reproducible multi-seed UR Fall cross-domain stability experiments."""
from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite, sqrt
from pathlib import Path
from typing import Sequence

from risk.urfall_crossdomain_training import train_urfall_crossdomain_experiment
from risk.urfall_threshold_selection import select_recall_constrained_threshold


def run_urfall_multiseed_experiment(
    *,
    fall_manifest: Path,
    fall_root: Path,
    adl_manifest: Path,
    adl_root: Path,
    output_dir: Path,
    seeds: Sequence[int] = (17, 42, 73),
    epochs: int = 80,
    device: str = "auto",
    recall_floor: float = 0.8,
    learning_rate: float = 1e-3,
    hidden_dim: int = 32,
    dropout: float = 0.2,
    adl_folds: int = 5,
) -> dict[str, object]:
    """Train declared seeds and aggregate threshold-selected validation metrics."""
    canonical_seeds = _validate_seeds(seeds)
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict[str, object]] = []
    for seed in canonical_seeds:
        seed_dir = root / f"seed-{seed}"
        training = train_urfall_crossdomain_experiment(
            Path(fall_manifest), Path(fall_root), Path(adl_manifest), Path(adl_root), seed_dir,
            epochs=epochs, learning_rate=learning_rate, hidden_dim=hidden_dim, dropout=dropout,
            device=device, random_seed=seed, adl_folds=adl_folds,
        )
        selection = select_recall_constrained_threshold(
            seed_dir / "fall_predictions.jsonl",
            seed_dir / "adl_predictions.jsonl",
            recall_floor=recall_floor,
        )
        _write_json(seed_dir / "threshold_selection.json", selection)
        per_seed.append({
            "seed": seed,
            "fixed_threshold_metrics": training["metrics"],
            "fixed_threshold_adl_false_positive_rate": training["adl_false_positive_rate"],
            "selected_threshold": selection,
        })

    metric_names = (
        "fall_f1", "fall_precision", "fall_recall", "adl_false_positive_rate", "threshold",
    )
    report: dict[str, object] = {
        "schema_version": "urfall.multiseed.v1",
        "external_experiment": True,
        "promoted": False,
        "claim_boundary": "Public-data stability evidence; not a deployment or clinical claim.",
        "validation": "FallLOSO+ADLGroupedHoldout",
        "seeds": canonical_seeds,
        "seed_count": len(canonical_seeds),
        "epochs": epochs,
        "recall_floor": float(recall_floor),
        "input_sha256": {
            "fall_manifest": _sha256(Path(fall_manifest)),
            "adl_manifest": _sha256(Path(adl_manifest)),
        },
        "per_seed": per_seed,
        "metrics": {
            name: _statistics([
                float(item["selected_threshold"][name])  # type: ignore[index]
                for item in per_seed
            ])
            for name in metric_names
        },
    }
    _write_json(root / "aggregate.json", report)
    return report


def _validate_seeds(seeds: Sequence[int]) -> list[int]:
    if isinstance(seeds, (str, bytes)):
        raise ValueError("seeds must be a non-empty sequence of unique non-negative integers")
    values = list(seeds)
    if not values or any(type(value) is not int or value < 0 for value in values) or len(set(values)) != len(values):
        raise ValueError("seeds must be a non-empty sequence of unique non-negative integers")
    return sorted(values)


def _statistics(values: list[float]) -> dict[str, float | int]:
    if not values or not all(isfinite(value) for value in values):
        raise ValueError("aggregate metrics must be finite")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "population_stddev": sqrt(variance),
        "minimum": min(values),
        "maximum": max(values),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"cannot hash input manifest: {error}") from error
