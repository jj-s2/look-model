"""Immutable experiment specifications for the RG-PCNet evaluation matrix.

The matrix is deliberately a data contract rather than a training loop.  A
runner receives :class:`ExperimentRun` objects and is responsible for writing
machine-readable artifacts.  Keeping the specification immutable makes it
safe to resume a partially completed matrix without silently changing a
variant, seed, or held-out split.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


def _freeze(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a recursively immutable, deterministic mapping."""

    frozen: dict[str, Any] = {}
    for key in sorted(value):
        item = value[key]
        if isinstance(item, Mapping):
            item = _freeze(item)
        elif isinstance(item, list):
            item = tuple(item)
        frozen[str(key)] = item
    return MappingProxyType(frozen)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _normalise_name(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _normalise_datasets(values: Iterable[object], *, held_out: str | None) -> tuple[str, ...]:
    names = {_normalise_name(value, "dataset") for value in values}
    if held_out is not None:
        names.discard(held_out)
    return tuple(sorted(names))


_RAW_VARIANTS: dict[str, dict[str, Any]] = {
    "fall_only": {
        "phase_weight": 0.0,
        "transition_weight": 0.0,
        "reliability_weight": 0.0,
        "teacher_weight": 0.0,
    },
    "phase": {
        "phase_weight": 0.5,
        "transition_weight": 0.0,
        "reliability_weight": 0.0,
        "teacher_weight": 0.0,
    },
    "phase_transition": {
        "phase_weight": 0.5,
        "transition_weight": 0.1,
        "reliability_weight": 0.0,
        "teacher_weight": 0.0,
    },
    "reliability": {
        "phase_weight": 0.0,
        "transition_weight": 0.0,
        "reliability_weight": 0.3,
        "teacher_weight": 0.0,
    },
    "full_rgpc": {
        "phase_weight": 0.5,
        "transition_weight": 0.1,
        "reliability_weight": 0.3,
        "teacher_weight": 0.0,
    },
    "full_rgpc_teacher": {
        "phase_weight": 0.5,
        "transition_weight": 0.1,
        "reliability_weight": 0.3,
        "teacher_weight": 0.2,
    },
    "feature_exclusion": {
        "phase_weight": 0.5,
        "transition_weight": 0.1,
        "reliability_weight": 0.3,
        "teacher_weight": 0.0,
        "exclude_rule_source_features": True,
    },
}

# Public constants are immutable, so callers cannot mutate a later run.
VARIANTS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {name: _freeze(config) for name, config in _RAW_VARIANTS.items()}
)


@dataclass(frozen=True)
class ExperimentRun:
    """One deterministic outer-fold/seed/ablation specification."""

    variant: str
    seed: int
    outer_subject: str
    train_datasets: tuple[str, ...]
    held_out_dataset: str | None
    config_overrides: Mapping[str, Any]
    run_id: str
    config_sha256: str

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown experiment variant: {self.variant}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.outer_subject:
            raise ValueError("outer_subject must be non-empty")
        if self.held_out_dataset and self.held_out_dataset in self.train_datasets:
            raise ValueError("held-out dataset must not be present in train_datasets")
        if not self.run_id or len(self.config_sha256) != 64:
            raise ValueError("run_id and config_sha256 are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "variant": self.variant,
            "seed": self.seed,
            "outer_subject": self.outer_subject,
            "train_datasets": list(self.train_datasets),
            "held_out_dataset": self.held_out_dataset,
            "config_overrides": dict(self.config_overrides),
            "config_sha256": self.config_sha256,
        }


def _run_id_payload(
    *,
    variant: str,
    seed: int,
    outer_subject: str,
    train_datasets: tuple[str, ...],
    held_out_dataset: str | None,
    config_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "variant": variant,
        "seed": seed,
        "outer_subject": outer_subject,
        "train_datasets": list(train_datasets),
        "held_out_dataset": held_out_dataset,
        "config_overrides": dict(config_overrides),
    }


def _make_run(
    variant: str,
    seed: int,
    outer_subject: str,
    train_datasets: tuple[str, ...],
    held_out_dataset: str | None,
) -> ExperimentRun:
    overrides = _freeze(VARIANTS[variant])
    payload = _run_id_payload(
        variant=variant,
        seed=seed,
        outer_subject=outer_subject,
        train_datasets=train_datasets,
        held_out_dataset=held_out_dataset,
        config_overrides=overrides,
    )
    config_hash = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    # Include a readable prefix while retaining a collision-resistant suffix.
    readable = f"{variant}__seed-{seed}__subject-{outer_subject}"
    run_id = f"{readable}__{config_hash[:12]}"
    return ExperimentRun(
        variant=variant,
        seed=seed,
        outer_subject=outer_subject,
        train_datasets=train_datasets,
        held_out_dataset=held_out_dataset,
        config_overrides=overrides,
        run_id=run_id,
        config_sha256=config_hash,
    )


def build_experiment_matrix(
    *,
    outer_subjects: Sequence[str],
    seeds: Sequence[int],
    train_datasets: Sequence[str] = (),
    held_out_dataset: str | None = None,
    variants: Sequence[str] | None = None,
) -> tuple[ExperimentRun, ...]:
    """Build the fixed-seed matrix in canonical order.

    ``variants`` is optional for controlled smoke tests; production calls use
    the complete seven-variant set.  Subjects, seeds, and variants are sorted
    only after duplicate checks, making matrix order independent of caller
    input while preserving the complete requested design.
    """

    subjects = tuple(_normalise_name(value, "outer_subject") for value in outer_subjects)
    if len(set(subjects)) != len(subjects) or not subjects:
        raise ValueError("outer_subjects must be non-empty and unique")
    if any("/" in subject or "\\" in subject for subject in subjects):
        raise ValueError("outer_subjects must not contain path separators")
    if not seeds:
        raise ValueError("seeds must be non-empty")
    seed_values: list[int] = []
    for seed in seeds:
        if type(seed) is not int or seed < 0:
            raise ValueError("seeds must contain non-negative integers")
        seed_values.append(seed)
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be unique")
    selected_variants = tuple(variants) if variants is not None else tuple(VARIANTS)
    if not selected_variants:
        raise ValueError("variants must be non-empty")
    if len(set(selected_variants)) != len(selected_variants):
        raise ValueError("variants must be unique")
    unknown = sorted(set(selected_variants).difference(VARIANTS))
    if unknown:
        raise ValueError(f"unknown experiment variant: {unknown[0]}")
    held_out = None if held_out_dataset is None else _normalise_name(held_out_dataset, "held_out_dataset")
    datasets = _normalise_datasets(train_datasets, held_out=held_out)
    return tuple(
        _make_run(variant, seed, subject, datasets, held_out)
        for variant in sorted(selected_variants, key=lambda item: tuple(VARIANTS).index(item))
        for seed in sorted(seed_values)
        for subject in sorted(subjects)
    )


__all__ = ["ExperimentRun", "VARIANTS", "build_experiment_matrix"]
