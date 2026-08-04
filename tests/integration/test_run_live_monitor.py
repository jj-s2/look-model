from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from risk.phase_model.schema import PhaseModelOutput
from vision.pose_pipeline import PoseFrameResult
from scripts.run_live_monitor import build_parser, run_monitor


class _FakeStream:
    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.frames = [np.zeros((64, 64, 3), dtype=np.uint8)]

    def open(self):
        self.opened = True
        return self

    def close(self):
        self.closed = True

    def read_frame(self):
        return self.frames.pop(0) if self.frames else None


class _FakePosePipeline:
    def process(self, frame, timestamp):
        return PoseFrameResult(
            frame_index=1,
            timestamp=timestamp,
            bboxes=[[1.0, 1.0, 32.0, 48.0]],
            keypoints=[[[float(index), float(index)] for index in range(17)]],
            keypoint_scores=[[1.0] * 17],
            payload={},
        )


class _FakePredictor:
    def predict(self, window):
        return PhaseModelOutput(
            phase_probs=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            fall_event_prob=0.0,
            prefall_prob=0.0,
            recovery_prob=0.0,
            quality_score=0.9,
            embedding_version="test",
            model_version="test",
        )


def test_live_runner_parser_exposes_checkpoint_and_input_flags():
    args = build_parser().parse_args(["--checkpoint", "model.pt", "--input", "video.mp4", "--no-browser"])
    assert args.checkpoint.name == "model.pt"
    assert args.input_source == "video.mp4"
    assert args.no_browser is True


def test_live_runner_rejects_missing_checkpoint_before_opening_stream(tmp_path: Path):
    stream = _FakeStream()
    try:
        run_monitor(
            checkpoint=tmp_path / "missing.pt",
            stream=stream,
            pose_pipeline=_FakePosePipeline(),
            predictor=_FakePredictor(),
            output_dir=tmp_path / "outputs",
            steps=1,
        )
    except FileNotFoundError as error:
        assert "checkpoint" in str(error).lower()
    else:  # pragma: no cover - assertion keeps the test explicit
        raise AssertionError("missing checkpoint must fail before stream opening")
    assert stream.opened is False


def test_live_runner_runs_one_injected_step_and_closes_stream(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test")
    stream = _FakeStream()
    snapshot = run_monitor(
        checkpoint=checkpoint,
        stream=stream,
        pose_pipeline=_FakePosePipeline(),
        predictor=_FakePredictor(),
        output_dir=tmp_path / "outputs",
        steps=1,
    )
    assert stream.opened is True
    assert stream.closed is True
    assert snapshot.camera_health in {"healthy", "degraded", "offline"}
    assert (tmp_path / "outputs" / "local_alerts.jsonl").exists()
