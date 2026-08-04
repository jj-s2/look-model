from datetime import datetime, timedelta, timezone

import pytest
import torch

from risk.phase_model.model import PhaseAwareFusionModel
from risk.phase_model.schema import PhaseModelOutput, PoseObservation
from risk.phase_model.torch_predictor import TorchPhasePredictor
from risk.phase_model.windows import DualWindow


def _window(scale: float = 1.0, offset: tuple[float, float] = (0.0, 0.0)) -> DualWindow:
    frames = []
    start = datetime(2026, 8, 4, tzinfo=timezone.utc)
    points = [(100.0, 100.0)] * 17
    points[11] = (100.0, 200.0)
    points[12] = (120.0, 200.0)
    points[5] = (100.0, 100.0)
    points[6] = (120.0, 100.0)
    transformed = tuple(
        (scale * x + offset[0], scale * y + offset[1]) for x, y in points
    )
    for index in range(64):
        frames.append(
            PoseObservation(
                timestamp=start + timedelta(milliseconds=500 * index),
                tracking_id="person-0",
                keypoints=transformed,
                scores=(0.9,) * 17,
                visible_mask=(True,) * 17,
                bbox=(0.0, 0.0, 320.0, 480.0),
                frame_size=(640, 480),
                stream_fresh=True,
            )
        )
    return DualWindow(short=tuple(frames[-48:]), long=tuple(frames))


def _checkpoint(path, model: PhaseAwareFusionModel | None = None) -> None:
    model = model or PhaseAwareFusionModel(short_dim=512, joints=17, hidden_dim=128)
    torch.save({"release_id": "test-release", "model": model.state_dict()}, path)


def test_predictor_loads_checkpoint_and_returns_phase_output(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint)

    predictor = TorchPhasePredictor(checkpoint, device="cpu")
    output = predictor.predict(_window())

    assert isinstance(output, PhaseModelOutput)
    assert len(output.phase_probs) == 6
    assert sum(output.phase_probs) == pytest.approx(1.0)
    assert 0.0 <= output.fall_event_prob <= 1.0
    assert 0.0 <= output.prefall_prob <= 1.0
    assert 0.0 <= output.recovery_prob <= 1.0
    assert output.quality_score == pytest.approx(0.9)
    assert predictor.short_branch_quality == 0.0


def test_predictor_normalizes_translation_and_scale_before_inference(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint)
    predictor = TorchPhasePredictor(checkpoint, device="cpu")

    first = predictor.predict(_window())
    transformed = predictor.predict(_window(scale=2.0, offset=(500.0, -100.0)))

    assert transformed.phase_probs == pytest.approx(first.phase_probs, abs=1e-5)
    assert transformed.fall_event_prob == pytest.approx(first.fall_event_prob, abs=1e-5)


def test_predictor_rejects_missing_state_dict(tmp_path):
    checkpoint = tmp_path / "invalid.pt"
    torch.save({"release_id": "test-release", "model": {}}, checkpoint)

    with pytest.raises(ValueError, match="state dict"):
        TorchPhasePredictor(checkpoint, device="cpu")


def test_predictor_rejects_unavailable_explicit_cuda(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        TorchPhasePredictor(checkpoint, device="cuda:0")
