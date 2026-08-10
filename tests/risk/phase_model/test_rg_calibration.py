import math

import pytest

from risk.phase_model.rg_calibration import (
    CalibrationArtifact,
    _temperature_candidates,
    fit_bounded_temperature,
)


def test_single_class_calibration_falls_back_to_identity():
    result = fit_bounded_temperature([2.0, 3.0], [1, 1], split_hash="abc")
    assert result.temperature == 1.0
    assert result.enabled is False
    assert result.reason == "calibration requires both classes"
    assert result.class_counts == {"0": 0, "1": 2}


def test_temperature_is_bounded_and_only_enabled_when_nll_improves():
    result = fit_bounded_temperature([4.0, -4.0, -2.0, 2.0], [1, 0, 1, 0], split_hash="abc")
    assert 0.5 <= result.temperature <= 5.0
    if result.enabled:
        assert result.nll_after < result.nll_before
    else:
        assert result.temperature == 1.0


def test_temperature_grid_has_exact_count_and_endpoints():
    candidates = _temperature_candidates()

    assert len(candidates) == 181
    assert candidates[0] == pytest.approx(0.5)
    assert candidates[-1] == pytest.approx(5.0)
    assert all(0.5 <= value <= 5.0 for value in candidates)
    assert all(left < right for left, right in zip(candidates, candidates[1:]))


@pytest.mark.parametrize(
    ("logits", "labels"),
    [([], []), ([1.0], []), ([1.0], [0, 1])],
)
def test_fit_rejects_empty_or_mismatched_inputs(logits, labels):
    with pytest.raises(ValueError, match="non-empty and have the same length"):
        fit_bounded_temperature(logits, labels, split_hash="abc")


@pytest.mark.parametrize("labels", [[0, 2], [-1, 1], [0.5, 1], ["0", 1], [math.nan, 1]])
def test_fit_rejects_non_binary_labels(labels):
    with pytest.raises(ValueError, match="binary"):
        fit_bounded_temperature([1.0, -1.0], labels, split_hash="abc")


@pytest.mark.parametrize("split_hash", [None, "", "   ", 123])
def test_fit_rejects_missing_split_hash(split_hash):
    with pytest.raises(ValueError, match="split_hash must be a non-empty string"):
        fit_bounded_temperature([1.0, -1.0], [1, 0], split_hash=split_hash)


def test_nonfinite_fit_data_returns_finite_identity_fallback():
    result = fit_bounded_temperature(
        [math.nan, math.inf, -math.inf, 0.0],
        [1, 0, 1, 0],
        split_hash="abc",
    )

    assert result.temperature == 1.0
    assert result.enabled is False
    assert result.reason == "calibration data contains non-finite logits"
    assert result.nll_after == result.nll_before
    assert result.brier_after == result.brier_before
    assert all(
        math.isfinite(value)
        for value in (
            result.nll_before,
            result.nll_after,
            result.brier_before,
            result.brier_after,
        )
    )


def test_no_improvement_returns_identity_metrics():
    result = fit_bounded_temperature([0.0, 0.0], [0, 1], split_hash="abc")

    assert result.enabled is False
    assert result.temperature == 1.0
    assert result.reason == "calibration did not improve nll or brier"
    assert result.nll_after == result.nll_before
    assert result.brier_after == result.brier_before


def test_artifact_snapshots_and_freezes_caller_owned_class_counts():
    class_counts = {"0": 1, "1": 1}
    artifact = CalibrationArtifact(
        temperature=1.0,
        enabled=False,
        reason="identity",
        sample_count=2,
        class_counts=class_counts,
        nll_before=0.5,
        nll_after=0.5,
        brier_before=0.25,
        brier_after=0.25,
        split_hash="abc",
    )

    class_counts["0"] = 99
    assert artifact.class_counts == {"0": 1, "1": 1}
    with pytest.raises(TypeError):
        artifact.class_counts["0"] = 99
    with pytest.raises(TypeError):
        artifact.class_counts.update({"0": 99})


def test_calibrate_returns_independent_output_and_rejects_nonfinite_logits():
    artifact = fit_bounded_temperature([0.0, 0.0], [0, 1], split_hash="abc")
    logits = [0.0]

    probabilities = artifact.calibrate(logits)
    logits[0] = 10.0

    assert probabilities == [0.5]
    with pytest.raises(ValueError, match="finite"):
        artifact.calibrate([math.nan])
    with pytest.raises(ValueError, match="finite"):
        artifact.calibrate([math.inf])


def test_large_logits_produce_stable_probabilities_and_finite_metrics():
    result = fit_bounded_temperature(
        [1e300, -1e300, 1e300, -1e300],
        [1, 0, 0, 1],
        split_hash="abc",
    )

    assert result.calibrate([1e300, -1e300, 0.0]) == [1.0, 0.0, 0.5]
    assert all(
        math.isfinite(value)
        for value in (
            result.nll_before,
            result.nll_after,
            result.brier_before,
            result.brier_after,
        )
    )


def test_fit_is_deterministic_for_repeated_inputs():
    logits = [4.0, -4.0, -2.0, 2.0]
    labels = [1, 0, 1, 0]

    first = fit_bounded_temperature(logits, labels, split_hash="abc")
    second = fit_bounded_temperature(tuple(logits), tuple(labels), split_hash="abc")

    assert first == second


def test_enabled_artifact_has_real_metric_improvement():
    result = fit_bounded_temperature([4.0, -4.0, -2.0, 2.0], [1, 0, 1, 0], split_hash="abc")

    assert result.enabled is True
    assert 0.5 <= result.temperature <= 5.0
    assert (
        result.nll_after < result.nll_before - 1e-6
        or result.brier_after < result.brier_before - 1e-6
    )
