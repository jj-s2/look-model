import pytest

from risk.phase_model.model import PhaseAwareFusionModel


def test_model_module_has_optional_dependency_boundary():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(RuntimeError, match="PyTorch"):
            PhaseAwareFusionModel(short_dim=16, joints=17, hidden_dim=32)
        return
    model = PhaseAwareFusionModel(short_dim=16, joints=17, hidden_dim=32)
    result = model(
        short_embedding=torch.zeros(2, 16),
        long_pose=torch.zeros(2, 64, 17, 3),
        short_quality=torch.tensor([1.0, 0.0]),
        long_quality=torch.tensor([0.0, 1.0]),
    )
    assert result.phase_logits.shape == (2, 6)
    assert result.fall_event_logit.shape == (2,)
    assert result.prefall_logit.shape == (2,)
    assert result.recovery_logit.shape == (2,)


def test_model_rejects_zero_quality_when_torch_is_available():
    torch = pytest.importorskip("torch")
    model = PhaseAwareFusionModel(short_dim=4, joints=17, hidden_dim=8)
    with pytest.raises(ValueError, match="quality"):
        model(torch.zeros(1, 4), torch.zeros(1, 8, 17, 3), torch.zeros(1), torch.zeros(1))
