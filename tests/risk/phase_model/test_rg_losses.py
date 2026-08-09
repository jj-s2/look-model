import pytest

torch = pytest.importorskip("torch")

from risk.phase_model.rg_losses import (
    RGPCLossTargets,
    compute_rgpc_loss,
    transition_consistency_loss,
)
from risk.phase_model.rg_pcnet import RGPCNetOutput


def _output(*, fall=(0.0,), phase=None, reliability=None, valid=None):
    phase = torch.zeros(1, 2, 3) if phase is None else phase
    reliability = torch.zeros(1, 2) if reliability is None else reliability
    valid = torch.ones(1, 2, dtype=torch.bool) if valid is None else valid
    return RGPCNetOutput(
        fall_logits=torch.zeros_like(reliability),
        window_fall_logit=torch.tensor(fall, dtype=torch.float32, device=phase.device, requires_grad=True),
        phase_logits=phase.requires_grad_(True),
        reliability_logits=reliability.requires_grad_(True),
        window_embedding=torch.zeros(1, 1, device=phase.device),
        valid_mask=valid,
    )


def _targets(*, fall=(1.0,), phase=None, phase_mask=None, reliability=None, valid=None, dt=None):
    valid = torch.ones(1, 2, dtype=torch.bool) if valid is None else valid
    return RGPCLossTargets(
        fall_target=torch.tensor(fall, dtype=torch.float32),
        phase_target=torch.full((1, 2), -1, dtype=torch.long) if phase is None else phase,
        phase_mask=torch.zeros(1, 2, dtype=torch.bool) if phase_mask is None else phase_mask,
        reliability_target=torch.zeros(1, 2) if reliability is None else reliability,
        valid_mask=valid,
        dt=torch.tensor([[0.0, 0.1]]) if dt is None else dt,
    )


def test_allowed_normal_to_descent_transition_has_negligible_penalty():
    """Break caught: permitted normal-to-descent edges are treated as forbidden."""
    logits = torch.tensor([[[12.0, -12.0, -12.0], [-12.0, 12.0, -12.0]]])
    loss = transition_consistency_loss(logits, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0.0, 0.1]]))
    assert loss.item() < 1e-6


def test_forbidden_normal_to_postfall_transition_is_penalized():
    """Break caught: forbidden normal-to-postfall edges have no probability penalty."""
    logits = torch.tensor([[[12.0, -12.0, -12.0], [-12.0, -12.0, 12.0]]])
    loss = transition_consistency_loss(logits, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0.0, 0.1]]))
    assert loss.item() > 0.99


def test_forbidden_postfall_to_descent_transition_is_penalized():
    """Break caught: the second forbidden edge is omitted from the transition penalty."""
    logits = torch.tensor([[[-12.0, -12.0, 12.0], [-12.0, 12.0, -12.0]]])
    loss = transition_consistency_loss(logits, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0.0, 0.1]]))
    assert loss.item() > 0.99


@pytest.mark.parametrize("valid", (torch.tensor([[False, True]]), torch.tensor([[True, False]])))
def test_transition_requires_both_endpoints_to_be_valid(valid):
    """Break caught: a forbidden edge attached to padding still receives a penalty."""
    logits = torch.tensor([[[12.0, -12.0, -12.0], [-12.0, -12.0, 12.0]]])
    assert transition_consistency_loss(logits, valid, torch.tensor([[0.0, 0.0]])).item() == 0.0


def test_zero_timestamp_interval_is_adjacent():
    """Break caught: zero-duration neighbouring frames are incorrectly excluded."""
    logits = torch.tensor([[[12.0, -12.0, -12.0], [-12.0, -12.0, 12.0]]])
    assert transition_consistency_loss(logits, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0.0, 0.0]])).item() > 0.99


def test_large_timestamp_gap_is_not_treated_as_adjacent():
    """Break caught: transitions across a timestamp discontinuity are penalized."""
    logits = torch.tensor([[[12.0, -12.0, -12.0], [-12.0, -12.0, 12.0]]])
    loss = transition_consistency_loss(logits, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0.0, 3.0]]), max_dt=0.5)
    assert loss.item() == 0.0


@pytest.mark.parametrize("dt", (torch.tensor([[0.0, -0.1]]), torch.tensor([[0.0, 0.1, 0.2]]), torch.tensor([[0.0, float("nan")]]), torch.tensor([[0.0, float("inf")]])))
def test_transition_rejects_negative_or_misaligned_dt(dt):
    """Break caught: invalid timestamp intervals silently produce an adjacency mask."""
    logits = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="dt"):
        transition_consistency_loss(logits, torch.ones(1, 2, dtype=torch.bool), dt)


def test_masked_components_match_hand_derived_bce_and_ce():
    """Break caught: padding or untrusted phase labels contribute to supervised losses."""
    log_two = float(torch.log(torch.tensor(2.0)))
    output = _output(
        fall=(0.0,),
        phase=torch.zeros(1, 2, 3),
        reliability=torch.zeros(1, 2),
        valid=torch.tensor([[True, False]]),
    )
    targets = _targets(
        phase=torch.tensor([[0, 2]]),
        phase_mask=torch.tensor([[True, True]]),
        reliability=torch.tensor([[1.0, 0.0]]),
        valid=torch.tensor([[True, False]]),
    )
    loss = compute_rgpc_loss(output, targets)
    assert loss.components["fall"].item() == pytest.approx(log_two)
    assert loss.components["phase"].item() == pytest.approx(float(torch.log(torch.tensor(3.0))))
    assert loss.components["reliability"].item() == pytest.approx(log_two)
    assert loss.components["transition"].item() == 0.0


def test_consistency_uses_only_reliable_frames_valid_in_both_outputs():
    """Break caught: unreliable, padding, or one-sided-invalid frames affect symmetric KL."""
    clean = _output(
        phase=torch.tensor([[[5.0, -5.0, -5.0], [5.0, -5.0, -5.0]]]),
        valid=torch.tensor([[True, True]]),
    )
    corrupted = _output(
        phase=torch.tensor([[[-5.0, 5.0, -5.0], [-5.0, 5.0, -5.0]]]),
        valid=torch.tensor([[True, False]]),
    )
    targets = _targets(reliability=torch.tensor([[1.0, 0.0]]), valid=torch.tensor([[True, True]]))
    loss = compute_rgpc_loss(clean, targets, corrupted_output=corrupted)
    assert loss.components["consistency"].item() > 9.0


def test_consistency_is_symmetric_kl_and_includes_half_reliability_threshold():
    """Break caught: KL direction/normalisation is asymmetric or reliability 0.5 is excluded."""
    clean = _output(phase=torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]))
    corrupt = _output(phase=torch.tensor([[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]]))
    targets = _targets(reliability=torch.tensor([[0.5, 0.49]]))
    loss = compute_rgpc_loss(clean, targets, corrupted_output=corrupt)
    p = torch.full((3,), 1.0 / 3.0)
    q = torch.tensor([1.0, 0.0, -1.0]).softmax(dim=0)
    expected = 0.5 * ((p * (p.log() - q.log())).sum() + (q * (q.log() - p.log())).sum())
    assert loss.components["consistency"].item() == pytest.approx(expected.item())


def test_consistency_is_zero_below_reliability_threshold():
    """Break caught: unreliable frames participate in the clean/corrupted KL objective."""
    clean = _output(phase=torch.zeros(1, 2, 3))
    corrupt = _output(phase=torch.tensor([[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]]))
    targets = _targets(reliability=torch.tensor([[0.49, 0.49]]))
    assert compute_rgpc_loss(clean, targets, corrupted_output=corrupt).components["consistency"].item() == 0.0


def test_selective_loss_uses_per_sample_valid_frame_mean_selection():
    """Break caught: selection is averaged per frame instead of per sample before selective risk."""
    output = _output(fall=(0.0,), reliability=torch.tensor([[0.0, 0.0]]), valid=torch.tensor([[True, False]]))
    targets = _targets(valid=torch.tensor([[True, False]]))
    loss = compute_rgpc_loss(output, targets, coverage_target=0.8)
    expected = 0.5 * float(torch.log(torch.tensor(2.0))) + 0.3
    assert loss.components["selective"].item() == pytest.approx(expected)


def test_missing_phase_supervision_returns_a_finite_differentiable_zero_component():
    """Break caught: empty phase masks yield NaN or disconnect their zero from model logits."""
    output = _output()
    targets = _targets(fall=(-1.0,))
    loss = compute_rgpc_loss(output, targets)
    assert torch.isfinite(loss.components["phase"])
    assert loss.components["phase"].item() == 0.0
    loss.components["phase"].backward()
    assert output.phase_logits.grad is not None


def test_loss_rejects_a_batch_with_any_empty_valid_row():
    """Break caught: a mixed batch silently removes an all-padding sample from supervision."""
    valid = torch.tensor([[True, True], [False, False]])
    output = RGPCNetOutput(
        fall_logits=torch.zeros(2, 2), window_fall_logit=torch.zeros(2), phase_logits=torch.zeros(2, 2, 3),
        reliability_logits=torch.zeros(2, 2), window_embedding=torch.zeros(2, 1), valid_mask=valid,
    )
    targets = RGPCLossTargets(torch.tensor([1.0, 1.0]), torch.full((2, 2), -1, dtype=torch.long), torch.zeros(2, 2, dtype=torch.bool), torch.zeros(2, 2), valid, torch.zeros(2, 2))
    with pytest.raises(ValueError, match="at least one valid frame"):
        compute_rgpc_loss(output, targets)


@pytest.mark.parametrize("fall", (torch.tensor([float("nan")]), torch.tensor([float("inf")]), torch.tensor([-2.0])))
def test_loss_rejects_nonfinite_or_non_sentinel_fall_targets(fall):
    """Break caught: malformed fall labels are silently treated as unknown supervision."""
    targets = _targets()
    targets = RGPCLossTargets(fall, targets.phase_target, targets.phase_mask, targets.reliability_target, targets.valid_mask, targets.dt)
    with pytest.raises(ValueError, match="fall_target"):
        compute_rgpc_loss(_output(), targets)


def test_loss_rejects_nonfinite_or_out_of_range_reliability_target():
    """Break caught: reliability NaNs, infinities, or sentinel negatives reach BCE/KL masks."""
    for value in (float("nan"), float("inf"), -1.0, 1.1):
        targets = _targets(reliability=torch.tensor([[value, 1.0]]))
        with pytest.raises(ValueError, match="reliability_target"):
            compute_rgpc_loss(_output(), targets)


def test_loss_requires_long_phase_targets():
    """Break caught: non-long phase tensors get as far as cross entropy and raise a framework error."""
    targets = _targets(phase=torch.zeros(1, 2, dtype=torch.int32), phase_mask=torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="phase_target"):
        compute_rgpc_loss(_output(), targets)


def test_fall_positive_weight_and_total_weights_are_applied_exactly():
    """Break caught: positive-class or documented component weights are silently ignored."""
    output = _output(fall=(0.0,), phase=torch.zeros(1, 2, 3), reliability=torch.zeros(1, 2))
    targets = _targets(phase=torch.tensor([[0, -1]]), phase_mask=torch.tensor([[True, False]]), reliability=torch.tensor([[1.0, 0.0]]))
    loss = compute_rgpc_loss(output, targets, fall_pos_weight=2.0, coverage_target=0.0)
    log_two = float(torch.log(torch.tensor(2.0)))
    assert loss.components["fall"].item() == pytest.approx(2.0 * log_two)
    expected = sum((loss.components["fall"], 0.5 * loss.components["phase"], 0.1 * loss.components["transition"], 0.3 * loss.components["reliability"], 0.1 * loss.components["consistency"], 0.1 * loss.components["selective"]))
    assert loss.total.item() == pytest.approx(expected.item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_loss_rejects_cpu_targets_masks_and_corruption_for_cuda_output():
    """Break caught: loss validation lets CPU tensors mix with CUDA logits before a cryptic operation error."""
    clean = _output(phase=torch.zeros(1, 2, 3, device="cuda"), reliability=torch.zeros(1, 2, device="cuda"), valid=torch.ones(1, 2, dtype=torch.bool, device="cuda"))
    cpu_targets = _targets()
    with pytest.raises(ValueError, match="device"):
        compute_rgpc_loss(clean, cpu_targets)
    cuda_targets = RGPCLossTargets(
        cpu_targets.fall_target.cuda(), cpu_targets.phase_target.cuda(), cpu_targets.phase_mask.cuda(),
        cpu_targets.reliability_target.cuda(), cpu_targets.valid_mask.cuda(), cpu_targets.dt.cuda(),
    )
    with pytest.raises(ValueError, match="device"):
        compute_rgpc_loss(clean, cuda_targets, corrupted_output=_output())


def test_loss_rejects_mismatched_target_shape_and_non_boolean_mask():
    """Break caught: labels cannot be safely aligned with prediction frames."""
    output = _output()
    targets = _targets(valid=torch.ones(1, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match="valid_mask"):
        compute_rgpc_loss(output, targets)
    malformed = _targets(phase=torch.zeros(1, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="phase_target"):
        compute_rgpc_loss(output, malformed)
