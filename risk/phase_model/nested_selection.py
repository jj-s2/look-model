"""Subject-safe calibrated threshold and reliability-gate selection.

The selector calibrates the canonical OOF logits once, evaluates the fixed
61-by-19 grid once, and ranks feasible pairs by subject-aware performance.
Reliability only decides whether a record is selected; it never rescales the
calibrated fall probability.

Binary recall, F1, and false-positive rate are overall metrics on selected
records. An absent selected positive denominator gives recall 0, an absent F1
denominator gives F1 0, and an absent selected negative denominator gives the
conservative false-positive rate 1. Every input subject participates in macro
and worst-subject F1; a subject with no selected records receives F1 0.

Risk-coverage curves include the empty prefix as ``(0, 0)``: with no selected
records there are no observed classification errors. Every non-empty prefix is
then included, and AURC is the trapezoidal area over those points. If no pair
is feasible there is no chosen fall threshold and therefore no curve; AURC is
the conservative sentinel 1 and the point collection is empty.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence

from risk.phase_model.rg_calibration import fit_bounded_temperature


_NO_FEASIBLE_REASON = (
    "no threshold pair satisfies recall, fpr, and coverage constraints"
)
_FEASIBLE_REASON = "threshold pair satisfies recall, fpr, and coverage constraints"
_METRIC_SCOPE = (
    "coverage is selected/all; recall, f1, and fpr are computed over selected "
    "records overall"
)
_ZERO_COVERAGE_CONVENTION = (
    "empty prefix has coverage 0 and risk 0 because it contains no "
    "classification errors"
)
_GRID_ITERATION_ORDER = (
    "fall_threshold_ascending",
    "reliability_threshold_ascending",
)


def _finite_float(value: object, message: str) -> float:
    if isinstance(value, (str, bytes, bool)):
        raise ValueError(message)
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(normalized):
        raise ValueError(message)
    return normalized


@dataclass(frozen=True)
class OOFRecord:
    """One validated prediction from a subject-disjoint OOF split."""

    subject_id: str
    label: int
    fall_logit: float
    reliability: float

    def __post_init__(self) -> None:
        if not isinstance(self.subject_id, str) or not self.subject_id.strip():
            raise ValueError("subject_id must be a non-empty string")
        if isinstance(self.label, (str, bytes, bool)):
            raise ValueError("label must be binary 0 or 1")
        try:
            numeric_label = float(self.label)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("label must be binary 0 or 1") from exc
        if not math.isfinite(numeric_label) or numeric_label not in (0.0, 1.0):
            raise ValueError("label must be binary 0 or 1")

        fall_logit = _finite_float(
            self.fall_logit, "fall_logit must be a finite number"
        )
        reliability = _finite_float(
            self.reliability, "reliability must be finite and in [0, 1]"
        )
        if not 0.0 <= reliability <= 1.0:
            raise ValueError("reliability must be finite and in [0, 1]")

        object.__setattr__(self, "label", int(numeric_label))
        object.__setattr__(self, "fall_logit", fall_logit)
        object.__setattr__(self, "reliability", reliability)


@dataclass(frozen=True)
class SelectiveThreshold:
    """Immutable selection result plus complete JSON-friendly audit metadata."""

    temperature: float
    fall_threshold: float | None
    reliability_threshold: float | None
    coverage: float
    recall: float
    f1: float
    fpr: float
    aurc: float
    feasible: bool
    reason: str
    subject_macro_f1: float
    worst_subject_f1: float
    recall_floor: float
    fpr_ceiling: float
    coverage_floor: float
    evaluated_constraints: tuple[tuple[str, float], ...]
    evaluated_pair_count: int
    feasible_pair_count: int
    calibration_enabled: bool
    calibration_reason: str
    split_hash: str
    fall_threshold_grid: tuple[float, ...]
    reliability_threshold_grid: tuple[float, ...]
    grid_iteration_order: tuple[str, str]
    metric_scope: str
    zero_coverage_convention: str
    risk_coverage_points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        probability_fields = (
            "coverage",
            "recall",
            "f1",
            "fpr",
            "aurc",
            "subject_macro_f1",
            "worst_subject_f1",
            "recall_floor",
            "fpr_ceiling",
            "coverage_floor",
        )
        normalized_probabilities = []
        for name in probability_fields:
            value = _finite_float(
                getattr(self, name), f"{name} must be finite and in [0, 1]"
            )
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            normalized_probabilities.append(value)

        temperature = _finite_float(
            self.temperature, "temperature must be finite and in [0.5, 5.0]"
        )
        if not 0.5 <= temperature <= 5.0:
            raise ValueError("temperature must be finite and in [0.5, 5.0]")

        thresholds = []
        for name in ("fall_threshold", "reliability_threshold"):
            raw_value = getattr(self, name)
            if raw_value is None:
                thresholds.append(None)
                continue
            value = _finite_float(raw_value, f"{name} must be finite and in [0, 1]")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            thresholds.append(value)

        if type(self.feasible) is not bool:
            raise ValueError("feasible must be a boolean")
        if self.feasible and any(value is None for value in thresholds):
            raise ValueError("feasible results require both thresholds")
        if not self.feasible and any(value is not None for value in thresholds):
            raise ValueError("infeasible results cannot contain chosen thresholds")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        if type(self.calibration_enabled) is not bool:
            raise ValueError("calibration_enabled must be a boolean")
        if not isinstance(self.calibration_reason, str) or not self.calibration_reason:
            raise ValueError("calibration_reason must be a non-empty string")
        if (
            not isinstance(self.split_hash, str)
            or len(self.split_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.split_hash)
        ):
            raise ValueError("split_hash must be a lowercase SHA-256 hex digest")
        if (
            type(self.evaluated_pair_count) is not int
            or self.evaluated_pair_count < 0
        ):
            raise ValueError("evaluated_pair_count must be a non-negative integer")
        if (
            type(self.feasible_pair_count) is not int
            or not 0 <= self.feasible_pair_count <= self.evaluated_pair_count
        ):
            raise ValueError("feasible_pair_count must be within evaluated pairs")

        evaluated_constraints = tuple(
            (str(name), float(value)) for name, value in self.evaluated_constraints
        )
        fall_grid = tuple(float(value) for value in self.fall_threshold_grid)
        reliability_grid = tuple(
            float(value) for value in self.reliability_threshold_grid
        )
        risk_points = tuple(
            (float(coverage), float(risk))
            for coverage, risk in self.risk_coverage_points
        )

        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "fall_threshold", thresholds[0])
        object.__setattr__(self, "reliability_threshold", thresholds[1])
        for name, value in zip(probability_fields, normalized_probabilities):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "evaluated_constraints", evaluated_constraints)
        object.__setattr__(self, "fall_threshold_grid", fall_grid)
        object.__setattr__(self, "reliability_threshold_grid", reliability_grid)
        object.__setattr__(self, "grid_iteration_order", tuple(self.grid_iteration_order))
        object.__setattr__(self, "risk_coverage_points", risk_points)


def _fall_thresholds() -> tuple[float, ...]:
    return tuple(index / 100 for index in range(20, 81))


def _reliability_thresholds() -> tuple[float, ...]:
    return tuple(index / 20 for index in range(19))


def _canonical_records(records: Sequence[OOFRecord]) -> tuple[OOFRecord, ...]:
    try:
        normalized = tuple(records)
    except TypeError as exc:
        raise ValueError("records must be non-empty") from exc
    if not normalized:
        raise ValueError("records must be non-empty")
    if any(not isinstance(record, OOFRecord) for record in normalized):
        raise ValueError("records must contain only OOFRecord values")
    if len({record.subject_id for record in normalized}) < 2:
        raise ValueError("records must contain at least two non-empty subjects")
    if {record.label for record in normalized} != {0, 1}:
        raise ValueError("records must contain both binary classes")
    return tuple(
        sorted(
            normalized,
            key=lambda record: (
                record.subject_id,
                record.label,
                record.fall_logit,
                record.reliability,
            ),
        )
    )


def _split_hash(records: Sequence[OOFRecord]) -> str:
    payload = [
        {
            "fall_logit": record.fall_logit,
            "label": record.label,
            "reliability": record.reliability,
            "subject_id": record.subject_id,
        }
        for record in records
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _constraint(value: object, name: str) -> float:
    message = f"{name} must be finite and in [0, 1]"
    normalized = _finite_float(value, message)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(message)
    return normalized


def _binary_metrics(
    labels: Sequence[int], predictions: Sequence[int]
) -> tuple[float, float, float]:
    """Return selected-set ``(recall, f1, fpr)`` conservatively."""
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    true_positive = sum(
        label == 1 and prediction == 1
        for label, prediction in zip(labels, predictions)
    )
    false_positive = sum(
        label == 0 and prediction == 1
        for label, prediction in zip(labels, predictions)
    )
    false_negative = sum(
        label == 1 and prediction == 0
        for label, prediction in zip(labels, predictions)
    )
    true_negative = sum(
        label == 0 and prediction == 0
        for label, prediction in zip(labels, predictions)
    )

    positive_count = true_positive + false_negative
    negative_count = false_positive + true_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    recall = true_positive / positive_count if positive_count else 0.0
    f1 = 2 * true_positive / f1_denominator if f1_denominator else 0.0
    fpr = false_positive / negative_count if negative_count else 1.0
    return recall, f1, fpr


def _subject_f1_scores(
    records: Sequence[OOFRecord],
    selected_indices: Sequence[int],
    selected_predictions: Sequence[int],
) -> tuple[float, float]:
    subjects = tuple(sorted({record.subject_id for record in records}))
    predictions_by_index = dict(zip(selected_indices, selected_predictions))
    subject_scores = []
    for subject in subjects:
        labels = []
        predictions = []
        for index, record in enumerate(records):
            if record.subject_id == subject and index in predictions_by_index:
                labels.append(record.label)
                predictions.append(predictions_by_index[index])
        _, subject_f1, _ = _binary_metrics(labels, predictions)
        subject_scores.append(subject_f1)
    return sum(subject_scores) / len(subject_scores), min(subject_scores)


def _risk_coverage_curve(
    records: Sequence[OOFRecord],
    probabilities: Sequence[float],
    fall_threshold: float,
) -> tuple[tuple[tuple[float, float], ...], float]:
    ranked = sorted(
        enumerate(zip(records, probabilities)),
        key=lambda item: -item[1][0].reliability,
    )
    points = [(0.0, 0.0)]
    errors = 0
    total = len(records)
    for prefix_size, (_canonical_index, (record, probability)) in enumerate(
        ranked, start=1
    ):
        prediction = int(probability >= fall_threshold)
        errors += int(prediction != record.label)
        points.append((prefix_size / total, errors / prefix_size))

    aurc = 0.0
    for (left_coverage, left_risk), (right_coverage, right_risk) in zip(
        points, points[1:]
    ):
        aurc += (right_coverage - left_coverage) * (left_risk + right_risk) / 2.0
    return tuple(points), aurc


def _result(
    *,
    temperature: float,
    fall_threshold: float | None,
    reliability_threshold: float | None,
    coverage: float,
    recall: float,
    f1: float,
    fpr: float,
    aurc: float,
    feasible: bool,
    reason: str,
    subject_macro_f1: float,
    worst_subject_f1: float,
    recall_floor: float,
    fpr_ceiling: float,
    coverage_floor: float,
    evaluated_pair_count: int,
    feasible_pair_count: int,
    calibration_enabled: bool,
    calibration_reason: str,
    split_hash: str,
    fall_threshold_grid: tuple[float, ...],
    reliability_threshold_grid: tuple[float, ...],
    risk_coverage_points: tuple[tuple[float, float], ...],
) -> SelectiveThreshold:
    return SelectiveThreshold(
        temperature=temperature,
        fall_threshold=fall_threshold,
        reliability_threshold=reliability_threshold,
        coverage=coverage,
        recall=recall,
        f1=f1,
        fpr=fpr,
        aurc=aurc,
        feasible=feasible,
        reason=reason,
        subject_macro_f1=subject_macro_f1,
        worst_subject_f1=worst_subject_f1,
        recall_floor=recall_floor,
        fpr_ceiling=fpr_ceiling,
        coverage_floor=coverage_floor,
        evaluated_constraints=(
            ("recall_floor", recall_floor),
            ("fpr_ceiling", fpr_ceiling),
            ("coverage_floor", coverage_floor),
        ),
        evaluated_pair_count=evaluated_pair_count,
        feasible_pair_count=feasible_pair_count,
        calibration_enabled=calibration_enabled,
        calibration_reason=calibration_reason,
        split_hash=split_hash,
        fall_threshold_grid=fall_threshold_grid,
        reliability_threshold_grid=reliability_threshold_grid,
        grid_iteration_order=_GRID_ITERATION_ORDER,
        metric_scope=_METRIC_SCOPE,
        zero_coverage_convention=_ZERO_COVERAGE_CONVENTION,
        risk_coverage_points=risk_coverage_points,
    )


def select_oof_thresholds(
    records: Sequence[OOFRecord],
    *,
    recall_floor: float,
    fpr_ceiling: float,
    coverage_floor: float,
) -> SelectiveThreshold:
    """Calibrate once and select a feasible fall/reliability threshold pair."""
    canonical_records = _canonical_records(records)
    normalized_recall_floor = _constraint(recall_floor, "recall_floor")
    normalized_fpr_ceiling = _constraint(fpr_ceiling, "fpr_ceiling")
    normalized_coverage_floor = _constraint(coverage_floor, "coverage_floor")

    split_hash = _split_hash(canonical_records)
    logits = [record.fall_logit for record in canonical_records]
    labels = [record.label for record in canonical_records]
    calibration = fit_bounded_temperature(logits, labels, split_hash=split_hash)
    probabilities = tuple(calibration.calibrate(logits))

    fall_grid = _fall_thresholds()
    reliability_grid = _reliability_thresholds()
    evaluated_pair_count = len(fall_grid) * len(reliability_grid)
    feasible_pair_count = 0
    best_rank: tuple[float, float, float, float, float] | None = None
    best_values: (
        tuple[float, float, float, float, float, float, float, float] | None
    ) = None

    # Strict rank comparison preserves the first pair for any remaining tie.
    # The fixed nested-loop order therefore resolves such ties deterministically.
    for fall_threshold in fall_grid:
        for reliability_threshold in reliability_grid:
            selected_indices = tuple(
                index
                for index, record in enumerate(canonical_records)
                if record.reliability >= reliability_threshold
            )
            selected_labels = tuple(
                canonical_records[index].label for index in selected_indices
            )
            selected_predictions = tuple(
                int(probabilities[index] >= fall_threshold)
                for index in selected_indices
            )
            coverage = len(selected_indices) / len(canonical_records)
            recall, f1, fpr = _binary_metrics(
                selected_labels, selected_predictions
            )
            subject_macro_f1, worst_subject_f1 = _subject_f1_scores(
                canonical_records, selected_indices, selected_predictions
            )

            if (
                recall < normalized_recall_floor
                or fpr > normalized_fpr_ceiling
                or coverage < normalized_coverage_floor
            ):
                continue

            feasible_pair_count += 1
            rank = (
                subject_macro_f1,
                worst_subject_f1,
                -fpr,
                coverage,
                -fall_threshold,
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_values = (
                    fall_threshold,
                    reliability_threshold,
                    coverage,
                    recall,
                    f1,
                    fpr,
                    subject_macro_f1,
                    worst_subject_f1,
                )

    if best_values is None:
        return _result(
            temperature=calibration.temperature,
            fall_threshold=None,
            reliability_threshold=None,
            coverage=0.0,
            recall=0.0,
            f1=0.0,
            fpr=1.0,
            aurc=1.0,
            feasible=False,
            reason=_NO_FEASIBLE_REASON,
            subject_macro_f1=0.0,
            worst_subject_f1=0.0,
            recall_floor=normalized_recall_floor,
            fpr_ceiling=normalized_fpr_ceiling,
            coverage_floor=normalized_coverage_floor,
            evaluated_pair_count=evaluated_pair_count,
            feasible_pair_count=feasible_pair_count,
            calibration_enabled=calibration.enabled,
            calibration_reason=calibration.reason,
            split_hash=split_hash,
            fall_threshold_grid=fall_grid,
            reliability_threshold_grid=reliability_grid,
            risk_coverage_points=(),
        )

    (
        fall_threshold,
        reliability_threshold,
        coverage,
        recall,
        f1,
        fpr,
        subject_macro_f1,
        worst_subject_f1,
    ) = best_values
    risk_coverage_points, aurc = _risk_coverage_curve(
        canonical_records, probabilities, fall_threshold
    )
    return _result(
        temperature=calibration.temperature,
        fall_threshold=fall_threshold,
        reliability_threshold=reliability_threshold,
        coverage=coverage,
        recall=recall,
        f1=f1,
        fpr=fpr,
        aurc=aurc,
        feasible=True,
        reason=_FEASIBLE_REASON,
        subject_macro_f1=subject_macro_f1,
        worst_subject_f1=worst_subject_f1,
        recall_floor=normalized_recall_floor,
        fpr_ceiling=normalized_fpr_ceiling,
        coverage_floor=normalized_coverage_floor,
        evaluated_pair_count=evaluated_pair_count,
        feasible_pair_count=feasible_pair_count,
        calibration_enabled=calibration.enabled,
        calibration_reason=calibration.reason,
        split_hash=split_hash,
        fall_threshold_grid=fall_grid,
        reliability_threshold_grid=reliability_grid,
        risk_coverage_points=risk_coverage_points,
    )
