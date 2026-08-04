import pytest

from risk.phase_model.losses import LossTargets, PhaseModelOutputTensor, compute_multitask_loss


def test_loss_requires_at_least_one_supervised_field():
    torch = pytest.importorskip("torch")
    outputs = PhaseModelOutputTensor(torch.zeros(1, 6), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    targets = LossTargets(torch.tensor([-1]), torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([-1.0]))
    with pytest.raises(ValueError, match="supervision"):
        compute_multitask_loss(outputs, targets, set())


def test_supervision_mask_omits_unknown_phase():
    torch = pytest.importorskip("torch")
    outputs = PhaseModelOutputTensor(torch.zeros(1, 6), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    targets = LossTargets(torch.tensor([-1]), torch.tensor([1.0]), torch.tensor([-1.0]), torch.tensor([-1.0]))
    loss = compute_multitask_loss(outputs, targets, {"fall_event"})
    assert loss.components["phase"].item() == 0.0
    assert loss.components["fall_event"].item() > 0.0


def test_fall_label_smoothing_reduces_extreme_target():
    torch = pytest.importorskip("torch")
    outputs = PhaseModelOutputTensor(torch.zeros(1, 6), torch.tensor([2.0]), torch.zeros(1), torch.zeros(1))
    targets = LossTargets(torch.tensor([-1]), torch.tensor([1.0]), torch.tensor([-1.0]), torch.tensor([-1.0]))

    raw = compute_multitask_loss(outputs, targets, {"fall_event"}, fall_label_smoothing=0.0)
    smooth = compute_multitask_loss(outputs, targets, {"fall_event"}, fall_label_smoothing=0.05)

    assert torch.isfinite(raw.total)
    assert torch.isfinite(smooth.total)
    assert smooth.components["fall_event"].item() > raw.components["fall_event"].item()


@pytest.mark.parametrize("smoothing", (-0.01, 0.5))
def test_loss_rejects_fall_smoothing_outside_half_open_range(smoothing):
    torch = pytest.importorskip("torch")
    outputs = PhaseModelOutputTensor(torch.zeros(1, 6), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    targets = LossTargets(torch.tensor([-1]), torch.tensor([1.0]), torch.tensor([-1.0]), torch.tensor([-1.0]))

    with pytest.raises(ValueError, match="fall_label_smoothing"):
        compute_multitask_loss(outputs, targets, {"fall_event"}, fall_label_smoothing=smoothing)
