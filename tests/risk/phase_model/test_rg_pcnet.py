import pytest

torch = pytest.importorskip("torch")

from risk.phase_model.rg_pcnet import RGPCNet


def test_rgpcnet_outputs_all_three_tasks():
    """Breaks if a task head or pooled representation has the wrong public shape."""
    model = RGPCNet(input_dim=112, hidden_dim=16, dropout=0.0)
    output = model(
        torch.zeros(2, 8, 112),
        torch.tensor([[1] * 8, [1] * 5 + [0] * 3], dtype=torch.bool),
    )

    assert output.fall_logits.shape == (2, 8)
    assert output.window_fall_logit.shape == (2,)
    assert output.phase_logits.shape == (2, 8, 3)
    assert output.reliability_logits.shape == (2, 8)
    assert output.window_embedding.shape == (2, 16)


def test_rgpcnet_is_causal_for_all_prefix_frame_outputs():
    """Breaks if a future frame can influence any earlier frame prediction."""
    torch.manual_seed(4)
    model = RGPCNet(input_dim=112, hidden_dim=16, dropout=0.0).eval()
    first = torch.randn(1, 12, 112)
    changed = first.clone()
    changed[:, 8:] = torch.randn_like(changed[:, 8:]) * 20
    mask = torch.ones(1, 12, dtype=torch.bool)

    with torch.no_grad():
        original = model(first, mask)
        modified = model(changed, mask)

    torch.testing.assert_close(original.fall_logits[:, :8], modified.fall_logits[:, :8])
    torch.testing.assert_close(original.phase_logits[:, :8], modified.phase_logits[:, :8])
    torch.testing.assert_close(
        original.reliability_logits[:, :8], modified.reliability_logits[:, :8]
    )


def test_rgpcnet_invalid_suffix_does_not_change_valid_outputs_or_pooling():
    """Breaks if masked frames affect valid frame outputs or either pooled result."""
    torch.manual_seed(9)
    model = RGPCNet(input_dim=112, hidden_dim=16, dropout=0.0).eval()
    valid_prefix = torch.randn(1, 5, 112)
    clean = torch.cat((valid_prefix, torch.zeros(1, 3, 112)), dim=1)
    corrupt_suffix = torch.cat((valid_prefix, torch.randn(1, 3, 112) * 100), dim=1)
    mask = torch.tensor([[1] * 5 + [0] * 3], dtype=torch.bool)

    with torch.no_grad():
        clean_output = model(clean, mask)
        corrupt_output = model(corrupt_suffix, mask)

    torch.testing.assert_close(clean_output.fall_logits[:, :5], corrupt_output.fall_logits[:, :5])
    torch.testing.assert_close(clean_output.window_fall_logit, corrupt_output.window_fall_logit)
    torch.testing.assert_close(clean_output.window_embedding, corrupt_output.window_embedding)


def test_rgpcnet_rejects_an_empty_mask():
    """Breaks if pooling silently accepts a window without valid frames."""
    model = RGPCNet(input_dim=112, hidden_dim=16, dropout=0.0)

    with pytest.raises(ValueError, match="at least one valid frame"):
        model(torch.zeros(2, 4, 112), torch.zeros(2, 4, dtype=torch.bool))


def test_rgpcnet_rejects_mask_shape_mismatch():
    """Breaks if a mask cannot be aligned with its feature sequence."""
    model = RGPCNet(input_dim=112, hidden_dim=16, dropout=0.0)

    with pytest.raises(ValueError, match="valid_mask shape"):
        model(torch.zeros(2, 4, 112), torch.ones(2, 3, dtype=torch.bool))
