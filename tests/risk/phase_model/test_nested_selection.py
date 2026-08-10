import json
import itertools
import math
from dataclasses import FrozenInstanceError, asdict, fields, replace

import pytest

import risk.phase_model.nested_selection as nested_selection
from risk.phase_model.nested_selection import (
    OOFRecord,
    SelectiveThreshold,
    _binary_metrics,
    select_oof_thresholds,
)


EXPECTED_FALL_GRID = tuple(index / 100 for index in range(20, 81))
EXPECTED_RELIABILITY_GRID = tuple(index / 20 for index in range(19))
NO_FEASIBLE_REASON = (
    "no threshold pair satisfies recall, fpr, and coverage constraints"
)


class _IdentityCalibration:
    temperature = 1.0
    enabled = False
    reason = "identity test calibration"

    def __init__(self) -> None:
        self.calibrate_calls = 0

    def calibrate(self, logits):
        self.calibrate_calls += 1
        return [1.0 / (1.0 + math.exp(-value)) for value in logits]


def _install_identity_calibration(monkeypatch):
    artifact = _IdentityCalibration()
    fit_calls = []

    def fake_fit(logits, labels, *, split_hash):
        fit_calls.append((tuple(logits), tuple(labels), split_hash))
        return artifact

    monkeypatch.setattr(nested_selection, "fit_bounded_temperature", fake_fit)
    return artifact, fit_calls


def _logit(probability):
    return math.log(probability / (1.0 - probability))


def _brief_records():
    return [
        OOFRecord("s1", 1, 3.0, 0.9),
        OOFRecord("s1", 0, -2.0, 0.8),
        OOFRecord("s2", 1, 2.0, 0.7),
        OOFRecord("s2", 0, 1.0, 0.2),
    ]


def test_oof_selection_enforces_recall_fpr_and_coverage():
    result = select_oof_thresholds(
        _brief_records(),
        recall_floor=1.0,
        fpr_ceiling=0.0,
        coverage_floor=0.5,
    )

    assert result.feasible
    assert result.recall == 1.0
    assert result.fpr == 0.0
    assert result.coverage >= 0.5


def test_no_feasible_oof_gate_returns_explicit_failure():
    records = [
        OOFRecord("s1", 1, -3.0, 0.1),
        OOFRecord("s2", 0, 3.0, 0.9),
    ]

    result = select_oof_thresholds(
        records,
        recall_floor=1.0,
        fpr_ceiling=0.0,
        coverage_floor=1.0,
    )

    assert result.feasible is False
    assert result.reason == NO_FEASIBLE_REASON
    assert result.fall_threshold is None
    assert result.reliability_threshold is None
    assert result.feasible_pair_count == 0
    assert result.aurc == 1.0
    assert result.risk_coverage_points == ()


def test_exact_grids_pair_count_and_constraints_are_audited():
    result = select_oof_thresholds(
        _brief_records(),
        recall_floor=1.0,
        fpr_ceiling=0.0,
        coverage_floor=0.5,
    )

    assert result.fall_threshold_grid == EXPECTED_FALL_GRID
    assert result.reliability_threshold_grid == EXPECTED_RELIABILITY_GRID
    assert result.evaluated_pair_count == 61 * 19 == 1159
    assert 0 < result.feasible_pair_count <= result.evaluated_pair_count
    assert result.evaluated_constraints == (
        ("recall_floor", 1.0),
        ("fpr_ceiling", 0.0),
        ("coverage_floor", 0.5),
    )
    assert result.grid_iteration_order == (
        "fall_threshold_ascending",
        "reliability_threshold_ascending",
    )


def test_hand_counted_fixture_has_exact_feasible_pair_count(monkeypatch):
    _install_identity_calibration(monkeypatch)
    records = [
        OOFRecord("positive", 1, _logit(0.60), 1.0),
        OOFRecord("negative", 0, _logit(0.40), 1.0),
    ]

    result = select_oof_thresholds(
        records,
        recall_floor=1.0,
        fpr_ceiling=0.0,
        coverage_floor=1.0,
    )

    # Exactly thresholds 0.41..0.60 classify both records correctly: 20 fall
    # thresholds times all 19 reliability thresholds gives 380 feasible pairs.
    assert result.feasible_pair_count == 380


def test_reliability_is_a_gate_and_never_a_probability_multiplier(monkeypatch):
    _install_identity_calibration(monkeypatch)
    records = [
        OOFRecord("positive", 1, _logit(0.60), 0.50),
        OOFRecord("negative", 0, _logit(0.40), 1.00),
    ]

    result = select_oof_thresholds(
        records,
        recall_floor=1.0,
        fpr_ceiling=0.0,
        coverage_floor=1.0,
    )

    assert result.feasible is True
    assert result.fall_threshold == pytest.approx(0.41)
    assert result.reliability_threshold == 0.0
    assert result.recall == 1.0
    assert result.fpr == 0.0


def test_subject_aware_ranking_beats_aggregate_only_choice(monkeypatch):
    _install_identity_calibration(monkeypatch)
    records = []
    records.extend(
        OOFRecord("large", 1, _logit(0.44), 1.0) for _ in range(20)
    )
    records.extend(
        OOFRecord("large", 1, _logit(0.80), 1.0) for _ in range(5)
    )
    records.append(OOFRecord("small", 1, _logit(0.80), 1.0))
    records.extend(
        OOFRecord("small", 0, _logit(0.44), 1.0) for _ in range(10)
    )

    result = select_oof_thresholds(
        records,
        recall_floor=0.0,
        fpr_ceiling=1.0,
        coverage_floor=1.0,
    )

    # Aggregate F1 alone prefers 0.20 (26 TP, 10 FP). Subject macro F1
    # prefers 0.45: the small subject improves from 1/6 to 1 while the large
    # subject falls from 1 to 1/3, so macro F1 rises from 7/12 to 2/3.
    assert result.fall_threshold == pytest.approx(0.45)
    assert result.subject_macro_f1 == pytest.approx(2.0 / 3.0)
    assert result.worst_subject_f1 == pytest.approx(1.0 / 3.0)
    assert result.f1 == pytest.approx(0.375)


@pytest.mark.parametrize(
    ("labels", "predictions", "expected"),
    [
        ((), (), (0.0, 0.0, 1.0)),
        ((1,), (1,), (1.0, 1.0, 1.0)),
        ((0,), (0,), (0.0, 0.0, 0.0)),
        ((1,), (0,), (0.0, 0.0, 1.0)),
    ],
)
def test_selected_binary_metrics_use_conservative_zero_denominators(
    labels, predictions, expected
):
    assert _binary_metrics(labels, predictions) == expected


def test_no_selected_subject_is_included_with_zero_f1(monkeypatch):
    _install_identity_calibration(monkeypatch)
    records = [
        OOFRecord("stable", 1, _logit(0.90), 0.90),
        OOFRecord("improves", 1, _logit(0.90), 0.90),
        OOFRecord("abstained", 1, _logit(0.90), 0.10),
    ]
    records.extend(
        OOFRecord("improves", 0, _logit(0.90), 0.10) for _ in range(4)
    )
    records.extend(
        OOFRecord("abstained", 0, _logit(0.90), 0.10) for _ in range(4)
    )

    result = select_oof_thresholds(
        records,
        recall_floor=1.0,
        fpr_ceiling=1.0,
        coverage_floor=0.1,
    )

    assert result.reliability_threshold == pytest.approx(0.15)
    assert result.coverage == pytest.approx(2.0 / 11.0)
    assert result.subject_macro_f1 == pytest.approx(2.0 / 3.0)
    assert result.worst_subject_f1 == 0.0


def test_remaining_rank_ties_use_fixed_grid_iteration(monkeypatch):
    _install_identity_calibration(monkeypatch)
    records = [
        OOFRecord("s1", 1, _logit(0.95), 1.0),
        OOFRecord("s2", 0, _logit(0.05), 1.0),
    ]

    result = select_oof_thresholds(
        records,
        recall_floor=1.0,
        fpr_ceiling=0.0,
        coverage_floor=1.0,
    )

    assert result.fall_threshold == 0.20
    assert result.reliability_threshold == 0.0


def test_rank_key_is_exact_and_each_component_decides_lexicographically():
    rank_key = nested_selection._rank_key
    baseline = rank_key(
        subject_macro_f1=0.7,
        worst_subject_f1=0.4,
        fpr=0.2,
        coverage=0.6,
        fall_threshold=0.5,
    )

    assert baseline == (0.7, 0.4, -0.2, 0.6, -0.5)
    assert rank_key(0.8, 0.0, 1.0, 0.0, 0.8) > baseline
    assert rank_key(0.7, 0.5, 1.0, 0.0, 0.8) > baseline
    assert rank_key(0.7, 0.4, 0.1, 0.0, 0.8) > baseline
    assert rank_key(0.7, 0.4, 0.2, 0.7, 0.8) > baseline
    assert rank_key(0.7, 0.4, 0.2, 0.6, 0.4) > baseline


def test_calibration_fit_and_probability_conversion_each_run_once(monkeypatch):
    artifact, fit_calls = _install_identity_calibration(monkeypatch)

    result = select_oof_thresholds(
        _brief_records(),
        recall_floor=0.0,
        fpr_ceiling=1.0,
        coverage_floor=0.0,
    )

    assert len(fit_calls) == 1
    assert artifact.calibrate_calls == 1
    assert len(fit_calls[0][2]) == 64
    assert set(fit_calls[0][2]) <= set("0123456789abcdef")
    assert result.split_hash == fit_calls[0][2]
    assert result.calibration_enabled is False
    assert result.calibration_reason == "identity test calibration"


def test_risk_coverage_curve_and_aurc_match_hand_calculation(monkeypatch):
    _install_identity_calibration(monkeypatch)
    records = [
        OOFRecord("s1", 1, _logit(0.90), 0.90),
        OOFRecord("s1", 0, _logit(0.90), 0.80),
        OOFRecord("s2", 0, _logit(0.10), 0.70),
    ]

    result = select_oof_thresholds(
        records,
        recall_floor=0.0,
        fpr_ceiling=1.0,
        coverage_floor=1.0,
    )

    assert result.risk_coverage_points == (
        (0.0, 0.0),
        (1.0 / 3.0, 0.0),
        (2.0 / 3.0, 0.5),
        (1.0, 1.0 / 3.0),
    )
    assert result.aurc == pytest.approx(2.0 / 9.0)
    assert result.zero_coverage_convention == (
        "empty prefix has coverage 0 and risk 0 because it contains no "
        "classification errors"
    )


def test_signed_zero_is_canonical_in_records_hash_order_and_result():
    records = (
        OOFRecord("s1", 0, -0.0, -0.0),
        OOFRecord("s1", 0, +0.0, +0.0),
        OOFRecord("s2", 1, -0.0, +0.0),
        OOFRecord("s2", 1, +0.0, -0.0),
    )

    assert all(record.fall_logit == 0.0 for record in records)
    assert all(math.copysign(1.0, record.fall_logit) == 1.0 for record in records)
    assert all(math.copysign(1.0, record.reliability) == 1.0 for record in records)

    expected_order = nested_selection._canonical_records(records)
    expected_hash = nested_selection._split_hash(expected_order)
    expected_result = select_oof_thresholds(
        records, recall_floor=0.0, fpr_ceiling=1.0, coverage_floor=0.0
    )
    for permutation in itertools.permutations(records):
        canonical = nested_selection._canonical_records(permutation)
        result = select_oof_thresholds(
            permutation, recall_floor=0.0, fpr_ceiling=1.0, coverage_floor=0.0
        )
        assert canonical == expected_order
        assert nested_selection._split_hash(canonical) == expected_hash
        assert result == expected_result


@pytest.mark.parametrize(
    ("metrics", "constraints"),
    [
        ((0.5, 0.5, 0.5), (math.nextafter(0.5, 1.0), 0.5, 0.5)),
        ((0.5, 0.5, 0.5), (0.5, math.nextafter(0.5, 0.0), 0.5)),
        ((0.5, 0.5, 0.5), (0.5, 0.5, math.nextafter(0.5, 1.0))),
    ],
)
def test_constraint_comparisons_accept_one_ulp_roundoff(metrics, constraints):
    assert nested_selection._constraints_satisfied(*metrics, *constraints)


@pytest.mark.parametrize(
    ("metrics", "constraints"),
    [
        ((0.5, 0.5, 0.5), (0.5 + 2e-12, 0.5, 0.5)),
        ((0.5, 0.5, 0.5), (0.5, 0.5 - 2e-12, 0.5)),
        ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5 + 2e-12)),
    ],
)
def test_constraint_comparisons_reject_beyond_shared_tolerance(metrics, constraints):
    assert not nested_selection._constraints_satisfied(*metrics, *constraints)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("subject_id", "", "subject_id must be a non-empty string"),
        ("subject_id", "   ", "subject_id must be a non-empty string"),
        ("subject_id", None, "subject_id must be a non-empty string"),
        ("label", 2, "label must be binary 0 or 1"),
        ("label", "1", "label must be binary 0 or 1"),
        ("fall_logit", math.nan, "fall_logit must be a finite number"),
        ("fall_logit", math.inf, "fall_logit must be a finite number"),
        ("fall_logit", object(), "fall_logit must be a finite number"),
        ("reliability", -0.01, "reliability must be finite and in [0, 1]"),
        ("reliability", 1.01, "reliability must be finite and in [0, 1]"),
        ("reliability", math.nan, "reliability must be finite and in [0, 1]"),
    ],
)
def test_oof_record_rejects_invalid_fields_deterministically(field, value, message):
    values = {
        "subject_id": "s1",
        "label": 1,
        "fall_logit": 0.0,
        "reliability": 0.5,
    }
    values[field] = value

    with pytest.raises(ValueError, match=rf"^{message.replace('[', r'\[').replace(']', r'\]')}$"):
        OOFRecord(**values)


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([], "records must be non-empty"),
        (
            [OOFRecord("s1", 0, -1.0, 0.5), OOFRecord("s1", 1, 1.0, 0.5)],
            "records must contain at least two non-empty subjects",
        ),
        (
            [OOFRecord("s1", 1, 1.0, 0.5), OOFRecord("s2", 1, 2.0, 0.5)],
            "records must contain both binary classes",
        ),
        ([object()], "records must contain only OOFRecord values"),
    ],
)
def test_selection_rejects_invalid_record_collections(records, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        select_oof_thresholds(
            records,
            recall_floor=0.0,
            fpr_ceiling=1.0,
            coverage_floor=0.0,
        )


@pytest.mark.parametrize("field", ["recall_floor", "fpr_ceiling", "coverage_floor"])
@pytest.mark.parametrize("invalid", [-0.01, 1.01, math.nan, math.inf, "bad", True])
def test_selection_rejects_invalid_constraints_deterministically(field, invalid):
    constraints = {
        "recall_floor": 0.0,
        "fpr_ceiling": 1.0,
        "coverage_floor": 0.0,
    }
    constraints[field] = invalid

    with pytest.raises(
        ValueError, match=rf"^{field} must be finite and in \[0, 1\]$"
    ):
        select_oof_thresholds(_brief_records(), **constraints)


def test_result_and_records_are_frozen_json_friendly_audit_snapshots():
    record = OOFRecord("s1", 1, 1.0, 0.5)
    with pytest.raises(FrozenInstanceError):
        record.label = 0

    result = select_oof_thresholds(
        _brief_records(),
        recall_floor=1.0,
        fpr_ceiling=0.0,
        coverage_floor=0.5,
    )

    assert [field.name for field in fields(SelectiveThreshold)][:11] == [
        "temperature",
        "fall_threshold",
        "reliability_threshold",
        "coverage",
        "recall",
        "f1",
        "fpr",
        "aurc",
        "feasible",
        "reason",
        "subject_macro_f1",
    ]
    assert result.metric_scope == (
        "coverage is selected/all; recall, f1, and fpr are computed over "
        "selected records overall"
    )
    assert (
        json.loads(json.dumps(asdict(result), allow_nan=False))["evaluated_pair_count"]
        == 1159
    )
    with pytest.raises(FrozenInstanceError):
        result.coverage = 0.0
    with pytest.raises(TypeError):
        result.fall_threshold_grid[0] = 0.0


def _valid_result():
    return select_oof_thresholds(
        _brief_records(),
        recall_floor=1.0,
        fpr_ceiling=0.0,
        coverage_floor=0.5,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: {"evaluated_constraints": tuple(reversed(result.evaluated_constraints))},
        lambda result: {
            "evaluated_constraints": (
                ("recall", result.recall_floor),
                ("fpr_ceiling", result.fpr_ceiling),
                ("coverage_floor", result.coverage_floor),
            )
        },
        lambda result: {
            "evaluated_constraints": (
                ("recall_floor", result.recall_floor + 0.01),
                ("fpr_ceiling", result.fpr_ceiling),
                ("coverage_floor", result.coverage_floor),
            )
        },
        lambda result: {"evaluated_pair_count": 1158},
        lambda result: {"feasible_pair_count": 0},
        lambda result: {"recall": result.recall_floor - 2e-12},
        lambda result: {"fpr": result.fpr_ceiling + 2e-12},
        lambda result: {"coverage": result.coverage_floor - 2e-12},
        lambda result: {"fall_threshold_grid": result.fall_threshold_grid[:-1]},
        lambda result: {
            "reliability_threshold_grid": result.reliability_threshold_grid[:-1]
        },
        lambda result: {"grid_iteration_order": tuple(reversed(result.grid_iteration_order))},
        lambda result: {"grid_iteration_order": (object(), object())},
        lambda result: {"metric_scope": "arbitrary"},
        lambda result: {"metric_scope": object()},
        lambda result: {"zero_coverage_convention": "arbitrary"},
        lambda result: {"fall_threshold": 0.205},
        lambda result: {"reliability_threshold": 0.025},
        lambda result: {"reason": "arbitrary"},
        lambda result: {"risk_coverage_points": ((0.1, 0.0), (1.0, 0.0))},
        lambda result: {
            "risk_coverage_points": ((0.0, 0.0), (0.5, 0.2), (0.5, 0.1), (1.0, 0.1))
        },
        lambda result: {
            "risk_coverage_points": ((0.0, 0.0), (0.5, math.nan), (1.0, 0.0))
        },
        lambda result: {
            "risk_coverage_points": ((0.0, 0.0), (0.5, 1.1), (1.0, 0.0))
        },
        lambda result: {
            "risk_coverage_points": result.risk_coverage_points[:-1]
        },
        lambda result: {"aurc": min(1.0, result.aurc + 0.01)},
        lambda result: {"calibration_reason": object()},
    ],
)
def test_selective_threshold_rejects_inconsistent_or_non_json_audit_state(mutation):
    result = _valid_result()
    with pytest.raises(ValueError):
        replace(result, **mutation(result))


def test_selective_threshold_detaches_and_normalizes_nested_lists():
    result = _valid_result()
    constraints = [list(item) for item in result.evaluated_constraints]
    fall_grid = list(result.fall_threshold_grid)
    reliability_grid = list(result.reliability_threshold_grid)
    grid_order = list(result.grid_iteration_order)
    risk_points = [list(item) for item in result.risk_coverage_points]

    detached = replace(
        result,
        evaluated_constraints=constraints,
        fall_threshold_grid=fall_grid,
        reliability_threshold_grid=reliability_grid,
        grid_iteration_order=grid_order,
        risk_coverage_points=risk_points,
    )
    constraints[0][1] = 0.0
    fall_grid[0] = 0.0
    reliability_grid[0] = 1.0
    grid_order[0] = "changed"
    risk_points[0][0] = 1.0

    assert detached == result
    assert type(detached.evaluated_constraints) is tuple
    assert all(type(item) is tuple for item in detached.evaluated_constraints)
    assert type(detached.fall_threshold_grid) is tuple
    assert type(detached.reliability_threshold_grid) is tuple
    assert type(detached.grid_iteration_order) is tuple
    assert type(detached.risk_coverage_points) is tuple
    assert all(type(item) is tuple for item in detached.risk_coverage_points)
    json.dumps(asdict(detached), allow_nan=False)


@pytest.mark.parametrize(
    "mutation",
    [
        {"coverage": 0.1},
        {"recall": 0.1},
        {"f1": 0.1},
        {"fpr": 0.0},
        {"aurc": 0.0},
        {"subject_macro_f1": 0.1},
        {"worst_subject_f1": 0.1},
        {"feasible_pair_count": 1},
        {"reason": "arbitrary"},
        {"risk_coverage_points": ((0.0, 0.0), (1.0, 0.0))},
    ],
)
def test_infeasible_result_rejects_nonconservative_sentinels(mutation):
    result = select_oof_thresholds(
        [OOFRecord("s1", 1, -3.0, 0.1), OOFRecord("s2", 0, 3.0, 0.9)],
        recall_floor=1.0,
        fpr_ceiling=0.0,
        coverage_floor=1.0,
    )

    with pytest.raises(ValueError):
        replace(result, **mutation)


def test_selection_is_repeatable_and_independent_of_input_order():
    records = _brief_records()

    first = select_oof_thresholds(
        records,
        recall_floor=1.0,
        fpr_ceiling=0.0,
        coverage_floor=0.5,
    )
    second = select_oof_thresholds(
        list(reversed(records)),
        recall_floor=1.0,
        fpr_ceiling=0.0,
        coverage_floor=0.5,
    )

    assert first == second
    assert first.split_hash == second.split_hash
