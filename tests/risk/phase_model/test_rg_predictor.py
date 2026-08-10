from datetime import datetime, timedelta, timezone
import hashlib

import pytest
import torch

from risk.phase_model.rg_pcnet import RGPCNet
from risk.phase_model.rg_predictor import RGPredictor
from risk.phase_model.schema import PhaseModelOutput, PoseObservation
from risk.phase_model.windows import DualWindow
from risk.phase_model.release_config import RGPCReleaseConfig, write_release_config


def _config():
    return RGPCReleaseConfig(
        schema_version="rgpc.release.v1", release_id="r1", model_sha256="a" * 64,
        dataset_sha256="b" * 64, split_sha256="c" * 64, temperature=1.5,
        fall_threshold=0.6, reliability_threshold=0.4, confirm_seconds=0.8,
        recovery_seconds=2.0, cooldown_seconds=10.0, minimum_coverage=0.6,
    )


def _window():
    start = datetime(2026, 8, 10, tzinfo=timezone.utc)
    frames = []
    for index in range(64):
        frames.append(PoseObservation(
            timestamp=start + timedelta(seconds=index), tracking_id="p1",
            keypoints=((0.0, 0.0),) * 17, scores=(0.9,) * 17,
            visible_mask=(True,) * 17, bbox=(0.0, 0.0, 100.0, 100.0),
            frame_size=(640, 480), stream_fresh=True,
        ))
    return DualWindow(short=tuple(frames[-48:]), long=tuple(frames))


def _release(tmp_path):
    model = RGPCNet()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = _config()
    config = RGPCReleaseConfig(**{**config.to_dict(), "model_sha256": digest})
    write_release_config(config, tmp_path / "release_config.json")
    return tmp_path


def test_low_reliability_prediction_abstains_without_multiplying_probability(tmp_path):
    predictor = RGPredictor.from_release(_release(tmp_path), device="cpu")
    predictor._predict_tensors = lambda *_: (0.8, (0.1, 0.8, 0.1), 0.2)
    output = predictor.predict(_window())
    assert output.fall_event_prob == pytest.approx(0.8)
    assert output.fall_decision is None
    assert output.quality_score == pytest.approx(0.2)
    assert output.phase_probs == pytest.approx((0.1, 0.0, 0.8, 0.0, 0.1, 0.0))


def test_predictor_returns_phase_model_output(tmp_path):
    predictor = RGPredictor.from_release(_release(tmp_path), device="cpu")
    output = predictor.predict(_window())
    assert isinstance(output, PhaseModelOutput)
    assert len(output.phase_probs) == 6
    assert sum(output.phase_probs) == pytest.approx(1.0)


def test_predict_rejects_non_64_long_window(tmp_path):
    predictor = RGPredictor.from_release(_release(tmp_path), device="cpu")
    window = _window()
    with pytest.raises(ValueError, match="64"):
        predictor.predict(DualWindow(short=window.short, long=window.long[:-1]))
