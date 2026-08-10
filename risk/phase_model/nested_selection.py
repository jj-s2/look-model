"""Subject-safe calibrated threshold and reliability-gate selection.

The selector calibrates the canonical OOF logits once, evaluates the fixed
61-by-19 grid once, and ranks feasible pairs by subject-aware performance.
Reliability only decides whether a record is selected; it never rescales the
calibrated fall probability.

All feasibility comparisons share a ``1e-12`` absolute tolerance so a single
floating-point rounding step cannot flip a boundary decision.

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
_FEASIBILITY_EPSILON = 1e-12
_FALL_THRESHOLD_GRID = tuple(index / 100 for index in range(20, 81))
_RELIABILITY_THRESHOLD_GRID = tuple(index / 20 for index in range(19))
_EVALUATED_PAIR_COUNT = 1159


def _finite_float(value: object, message: str) -> float:
    if isinstance(value, (str, bytes, bool)):
        raise ValueError(message)
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(normalized):
        raise ValueError(message)
    return 0.0 if normalized == 0.0 else normalized


def _probability(value: object, name: str) -> float:
    message = f"{name} must be finite and in [0, 1]"
    normalized = _finite_float(value, message)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(message)
    return normalized


def _pair_sequence(value: object, name: str) -> tuple[tuple[object, object], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence of pairs")
    pairs = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{name} must be a sequence of pairs")
        pairs.append((item[0], item[1]))
    return tuple(pairs)


def _normalized_constraints(
    value: object,
) -> tuple[tuple[str, float], ...]:
    pairs = _pair_sequence(value, "evaluated_constraints")
    normalized = []
    for name, raw_value in pairs:
        if type(name) is not str:
            raise ValueError("evaluated constraint names must be strings")
        normalized.append((name, _probability(raw_value, name)))
    return tuple(normalized)


def _normalized_grid(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(_probability(item, name) for item in value)


def _normalized_string_pair(value: object, name: str) -> tuple[str, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two strings")
    if any(type(item) is not str for item in value):
        raise ValueError(f"{name} must contain exactly two strings")
    return value[0], value[1]


def _normalized_risk_points(value: object) -> tuple[tuple[float, float], ...]:
    pairs = _pair_sequence(value, "risk_coverage_points")
    return tuple(
        (
            _probability(coverage, "risk coverage"),
            _probability(risk, "risk"),
        )
        for coverage, risk in pairs
    )


def _trapezoidal_area(points: Sequence[tuple[float, float]]) -> float:
    return sum(
        (right_coverage - left_coverage) * (left_risk + right_risk) / 2.0
        for (left_coverage, left_risk), (right_coverage, right_risk) in zip(
            points, points[1:]
        )
    )


def _nearest_integer(value: float, lower: int, upper: int) -> int | None:
    candidate = round(value)
    if (
        candidate < lower
        or candidate > upper
        or abs(value - candidate) > _FEASIBILITY_EPSILON
    ):
        return None
    return candidate


def _curve_counts(
    points: Sequence[tuple[float, float]], selected_coverage: float
) -> tuple[int, int]:
    sample_count = len(points) - 1
    cumulative_errors = []
    for prefix_size, (coverage, risk) in enumerate(points):
        if abs(coverage - prefix_size / sample_count) > _FEASIBILITY_EPSILON:
            raise ValueError("risk curve coverage must use every prefix lattice point")
        error_count = _nearest_integer(risk * prefix_size, 0, prefix_size)
        if error_count is None:
            raise ValueError("prefix risk must represent an integer error count")
        if cumulative_errors and error_count - cumulative_errors[-1] not in (0, 1):
            raise ValueError("cumulative prefix errors must increment by zero or one")
        cumulative_errors.append(error_count)

    selected_count = round(selected_coverage * sample_count)
    if (
        not 0 <= selected_count <= sample_count
        or abs(selected_coverage - selected_count / sample_count)
        > _FEASIBILITY_EPSILON
    ):
        raise ValueError("selected coverage must lie on the prefix lattice")
    return selected_count, cumulative_errors[selected_count]


def _confusion_metrics_realizable(
    *,
    selected_count: int,
    selected_errors: int,
    recall: float,
    f1: float,
    fpr: float,
) -> bool:
    for positive_count in range(selected_count + 1):
        negative_count = selected_count - positive_count
        if positive_count == 0:
            if abs(recall) > _FEASIBILITY_EPSILON:
                continue
            true_positive = 0
        else:
            true_positive = round(recall * positive_count)
            if (
                not 0 <= true_positive <= positive_count
                or abs(true_positive / positive_count - recall)
                > _FEASIBILITY_EPSILON
            ):
                continue

        if negative_count == 0:
            if abs(fpr - 1.0) > _FEASIBILITY_EPSILON:
                continue
            false_positive = 0
        else:
            false_positive = round(fpr * negative_count)
            if (
                not 0 <= false_positive <= negative_count
                or abs(false_positive / negative_count - fpr)
                > _FEASIBILITY_EPSILON
            ):
                continue

        false_negative = positive_count - true_positive
        f1_denominator = 2 * true_positive + false_positive + false_negative
        expected_f1 = (
            2 * true_positive / f1_denominator if f1_denominator else 0.0
        )
        if abs(expected_f1 - f1) > _FEASIBILITY_EPSILON:
            continue
        if false_positive + false_negative != selected_errors:
            continue
        return True
    return False


def _constraints_satisfied(
    recall: float,
    fpr: float,
    coverage: float,
    recall_floor: float,
    fpr_ceiling: float,
    coverage_floor: float,
) -> bool:
    """Apply every feasibility boundary with one shared numerical tolerance."""
    return not (
        recall + _FEASIBILITY_EPSILON < recall_floor
        or fpr > fpr_ceiling + _FEASIBILITY_EPSILON
        or coverage + _FEASIBILITY_EPSILON < coverage_floor
    )


def _rank_key(
    subject_macro_f1: float,
    worst_subject_f1: float,
    fpr: float,
    coverage: float,
    fall_threshold: float,
) -> tuple[float, float, float, float, float]:
    """Return the binding five-component feasible-pair rank, unchanged."""
    return (
        subject_macro_f1,
        worst_subject_f1,
        -fpr,
        coverage,
        -fall_threshold,
    )


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
        normalized_probabilities = [
            _probability(getattr(self, name), name) for name in probability_fields
        ]

        temperature = _finite_float(
            self.temperature, "temperature must be finite and in [0.5, 5.0]"
        )
        if not 0.5 <= temperature <= 5.0:
            raise ValueError("temperature must be finite and in [0.5, 5.0]")

        thresholds: list[float | None] = []
        for name in ("fall_threshold", "reliability_threshold"):
            raw_value = getattr(self, name)
            if raw_value is None:
                thresholds.append(None)
                continue
            thresholds.append(_probability(raw_value, name))

        evaluated_constraints = _normalized_constraints(self.evaluated_constraints)
        fall_grid = _normalized_grid(self.fall_threshold_grid, "fall_threshold_grid")
        reliability_grid = _normalized_grid(
            self.reliability_threshold_grid, "reliability_threshold_grid"
        )
        grid_iteration_order = _normalized_string_pair(
            self.grid_iteration_order, "grid_iteration_order"
        )
        risk_points = _normalized_risk_points(self.risk_coverage_points)

        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "fall_threshold", thresholds[0])
        object.__setattr__(self, "reliability_threshold", thresholds[1])
        for name, value in zip(probability_fields, normalized_probabilities):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "evaluated_constraints", evaluated_constraints)
        object.__setattr__(self, "fall_threshold_grid", fall_grid)
        object.__setattr__(self, "reliability_threshold_grid", reliability_grid)
        object.__setattr__(self, "grid_iteration_order", grid_iteration_order)
        object.__setattr__(self, "risk_coverage_points", risk_points)

        _validate_selective_threshold(self)


def _validate_selective_threshold(result: SelectiveThreshold) -> None:
    if type(result.feasible) is not bool:
        raise ValueError("feasible must be a boolean")
    if type(result.calibration_enabled) is not bool:
        raise ValueError("calibration_enabled must be a boolean")
    if type(result.calibration_reason) is not str or not result.calibration_reason:
        raise ValueError("calibration_reason must be a non-empty string")
    if (
        type(result.split_hash) is not str
        or len(result.split_hash) != 64
        or any(
            character not in "0123456789abcdef" for character in result.split_hash
        )
    ):
        raise ValueError("split_hash must be a lowercase SHA-256 hex digest")

    expected_constraints = (
        ("recall_floor", result.recall_floor),
        ("fpr_ceiling", result.fpr_ceiling),
        ("coverage_floor", result.coverage_floor),
    )
    if result.evaluated_constraints != expected_constraints:
        raise ValueError("evaluated_constraints must exactly match configured constraints")
    if result.fall_threshold_grid != _FALL_THRESHOLD_GRID:
        raise ValueError("fall_threshold_grid must be the fixed 61-point grid")
    if result.reliability_threshold_grid != _RELIABILITY_THRESHOLD_GRID:
        raise ValueError("reliability_threshold_grid must be the fixed 19-point grid")
    if result.grid_iteration_order != _GRID_ITERATION_ORDER:
        raise ValueError("grid_iteration_order must match the fixed iteration order")
    if type(result.metric_scope) is not str or result.metric_scope != _METRIC_SCOPE:
        raise ValueError("metric_scope must match the selected-record convention")
    if (
        type(result.zero_coverage_convention) is not str
        or result.zero_coverage_convention != _ZERO_COVERAGE_CONVENTION
    ):
        raise ValueError("zero_coverage_convention must match the empty-prefix rule")
    if (
        type(result.evaluated_pair_count) is not int
        or result.evaluated_pair_count != _EVALUATED_PAIR_COUNT
    ):
        raise ValueError("evaluated_pair_count must equal 1159")
    if (
        type(result.feasible_pair_count) is not int
        or not 0 <= result.feasible_pair_count <= result.evaluated_pair_count
    ):
        raise ValueError("feasible_pair_count must be within evaluated pairs")
    if result.worst_subject_f1 > result.subject_macro_f1 + _FEASIBILITY_EPSILON:
        raise ValueError("worst_subject_f1 cannot exceed subject_macro_f1")

    if result.feasible:
        _validate_feasible_result(result)
    else:
        _validate_infeasible_result(result)


def _validate_feasible_result(result: SelectiveThreshold) -> None:
    if result.feasible_pair_count == 0:
        raise ValueError("feasible results require at least one feasible pair")
    if type(result.reason) is not str or result.reason != _FEASIBLE_REASON:
        raise ValueError("feasible result reason is invalid")
    if result.fall_threshold not in _FALL_THRESHOLD_GRID:
        raise ValueError("fall_threshold must be a fixed grid member")
    if result.reliability_threshold not in _RELIABILITY_THRESHOLD_GRID:
        raise ValueError("reliability_threshold must be a fixed grid member")
    if not _constraints_satisfied(
        result.recall,
        result.fpr,
        result.coverage,
        result.recall_floor,
        result.fpr_ceiling,
        result.coverage_floor,
    ):
        raise ValueError("feasible metrics do not satisfy configured constraints")

    points = result.risk_coverage_points
    if len(points) < 2 or points[0] != (0.0, 0.0) or points[-1][0] != 1.0:
        raise ValueError("feasible risk curve must run from (0, 0) to coverage 1")
    if any(
        right_coverage <= left_coverage
        for (left_coverage, _), (right_coverage, _) in zip(points, points[1:])
    ):
        raise ValueError("risk curve coverage must be strictly increasing")
    if abs(_trapezoidal_area(points) - result.aurc) > _FEASIBILITY_EPSILON:
        raise ValueError("aurc must equal the trapezoidal risk-coverage area")
    selected_count, selected_errors = _curve_counts(points, result.coverage)
    if not _confusion_metrics_realizable(
        selected_count=selected_count,
        selected_errors=selected_errors,
        recall=result.recall,
        f1=result.f1,
        fpr=result.fpr,
    ):
        raise ValueError(
            "selected metrics are not realizable by the risk-curve evidence"
        )


def _validate_infeasible_result(result: SelectiveThreshold) -> None:
    expected_sentinels = (
        result.fall_threshold is None,
        result.reliability_threshold is None,
        result.coverage == 0.0,
        result.recall == 0.0,
        result.f1 == 0.0,
        result.fpr == 1.0,
        result.aurc == 1.0,
        result.subject_macro_f1 == 0.0,
        result.worst_subject_f1 == 0.0,
        result.feasible_pair_count == 0,
        type(result.reason) is str and result.reason == _NO_FEASIBLE_REASON,
        result.risk_coverage_points == (),
    )
    if not all(expected_sentinels):
        raise ValueError("infeasible result sentinels are inconsistent")


def _fall_thresholds() -> tuple[float, ...]:
    return _FALL_THRESHOLD_GRID


def _reliability_thresholds() -> tuple[float, ...]:
    return _RELIABILITY_THRESHOLD_GRID


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
    return _probability(value, name)


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

    immutable_points = tuple(points)
    return immutable_points, _trapezoidal_area(immutable_points)


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
    evaluated_pair_count = 0
    feasible_pair_count = 0
    best_rank: tuple[float, float, float, float, float] | None = None
    best_values: (
        tuple[float, float, float, float, float, float, float, float] | None
    ) = None

    # Strict rank comparison preserves the first pair for any remaining tie.
    # The fixed nested-loop order therefore resolves such ties deterministically.
    for fall_threshold in fall_grid:
        for reliability_threshold in reliability_grid:
            evaluated_pair_count += 1
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

            if not _constraints_satisfied(
                recall,
                fpr,
                coverage,
                normalized_recall_floor,
                normalized_fpr_ceiling,
                normalized_coverage_floor,
            ):
                continue

            feasible_pair_count += 1
            rank = _rank_key(
                subject_macro_f1,
                worst_subject_f1,
                fpr,
                coverage,
                fall_threshold,
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
