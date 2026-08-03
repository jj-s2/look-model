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
