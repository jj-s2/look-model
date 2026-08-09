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
        window_fall_logit=torch.tensor(fall, dtype=torch.float32, requires_grad=True),
        phase_logits=phase.requires_grad_(True),
        reliability_logits=reliability.requires_grad_(True),
        window_embedding=torch.zeros(1, 1),
        valid_mask=valid,
    )


def _targets(*, fall=(1.0,), phase=None, phase_mask=None, reliability=None, valid=None, dt=None):
    valid = torch.ones(1, 2, dtype=torch.bool) if valid is None else valid
    return RGPCLossTargets(
        fall_target=torch.tensor(fall, dtype=torch.float32),
        phase_target=torch.full((1, 2), -1, dtype=torch.long) if phase is None else phase,
        phase_mask=torch.zeros(1, 2, dtype=torch.bool) if phase_mask is None else phase_mask,
        reliability_target=torch.full((1, 2), -1.0) if reliability is None else reliability,
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


def test_large_timestamp_gap_is_not_treated_as_adjacent():
    """Break caught: transitions across a timestamp discontinuity are penalized."""
    logits = torch.tensor([[[12.0, -12.0, -12.0], [-12.0, -12.0, 12.0]]])
    loss = transition_consistency_loss(logits, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0.0, 3.0]]), max_dt=0.5)
    assert loss.item() == 0.0


@pytest.mark.parametrize("dt", (torch.tensor([[0.0, -0.1]]), torch.tensor([[0.0, 0.1, 0.2]])))
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


def test_selective_loss_uses_per_sample_valid_frame_mean_selection():
    """Break caught: selection is averaged per frame instead of per sample before selective risk."""
    output = _output(fall=(0.0,), reliability=torch.tensor([[0.0, 0.0]]), valid=torch.tensor([[True, False]]))
    targets = _targets(valid=torch.tensor([[True, False]]))
    loss = compute_rgpc_loss(output, targets, coverage_target=0.8)
    expected = 0.5 * float(torch.log(torch.tensor(2.0))) + 0.3
    assert loss.components["selective"].item() == pytest.approx(expected)


def test_missing_supervision_returns_finite_differentiable_zero_components():
    """Break caught: empty masks yield NaN or disconnect zero losses from model logits."""
    output = _output(valid=torch.tensor([[False, False]]))
    targets = _targets(valid=torch.tensor([[False, False]]))
    loss = compute_rgpc_loss(output, targets)
    assert all(torch.isfinite(value) for value in loss.components.values())
    assert loss.total.item() == 0.0
    loss.total.backward()
    assert output.window_fall_logit.grad is not None
    assert output.phase_logits.grad is not None
    assert output.reliability_logits.grad is not None


def test_loss_rejects_mismatched_target_shape_and_non_boolean_mask():
    """Break caught: labels cannot be safely aligned with prediction frames."""
    output = _output()
    targets = _targets(valid=torch.ones(1, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match="valid_mask"):
        compute_rgpc_loss(output, targets)
    malformed = _targets(phase=torch.zeros(1, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="phase_target"):
        compute_rgpc_loss(output, malformed)
