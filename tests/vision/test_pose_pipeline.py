from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import pytest

from vision.pose_buffer import PoseSequenceBuffer
from vision.pose_pipeline import Detection, PoseFrameResult, PoseInferenceOutput, PosePipeline


NOW = datetime(2026, 8, 2, tzinfo=timezone.utc)
FRAME = object()


class FakeDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, frame):
        assert frame is FRAME
        return self._detections


class FakePoseEstimator:
    def __init__(self):
        self.last_boxes = None

    def estimate(self, frame, boxes):
        assert frame is FRAME
        self.last_boxes = boxes
        return PoseInferenceOutput(
            keypoints=[[[10.0, 20.0]]],
            keypoint_scores=[[0.9]],
        )


def frame(index):
    return PoseFrameResult(index, NOW, [], [], [])


def test_pipeline_passes_only_person_boxes_to_pose_estimator():
    """A non-person detection must never reach the pose-model boundary."""
    detector = FakeDetector([
        Detection("person", 0.9, [1, 2, 30, 40]),
        Detection("chair", 0.8, [5, 6, 20, 25]),
    ])
    pose = FakePoseEstimator()

    result = PosePipeline(detector, pose, person_threshold=0.5).process(FRAME, NOW)

    assert pose.last_boxes == [[1, 2, 30, 40]]
    assert result.timestamp == NOW
    assert result.frame_index == 1
    assert result.keypoints == [[[10.0, 20.0]]]


def test_sequence_buffer_emits_fixed_length_overlapping_window():
    """Dropping fewer than a full window after emission would break overlap."""
    buffer = PoseSequenceBuffer(window_size=4, stride=2)

    emitted = [buffer.append(frame(i)) for i in range(6)]

    windows = [window for window in emitted if window is not None]
    assert [len(window) for window in windows] == [4, 4]
    assert [item.frame_index for item in windows[1]] == [2, 3, 4, 5]


def test_pipeline_does_not_call_pose_estimator_without_person_boxes():
    """An empty person set must not become an accidental full-frame pose run."""
    pose = FakePoseEstimator()

    result = PosePipeline(
        FakeDetector([Detection("chair", 0.9, [1, 2, 30, 40])]), pose
    ).process(FRAME, NOW)

    assert pose.last_boxes is None
    assert result.bboxes == []
    assert result.keypoints == []
    assert result.keypoint_scores == []
    assert result.payload == {}


def test_sequence_buffer_stride_one_keeps_exact_overlap():
    buffer = PoseSequenceBuffer(window_size=3, stride=1)

    windows = [window for window in (buffer.append(frame(i)) for i in range(5)) if window]

    assert [[item.frame_index for item in window] for window in windows] == [
        [0, 1, 2], [1, 2, 3], [2, 3, 4],
    ]


def test_sequence_buffer_full_stride_has_no_overlap():
    buffer = PoseSequenceBuffer(window_size=3, stride=3)

    windows = [window for window in (buffer.append(frame(i)) for i in range(6)) if window]

    assert [[item.frame_index for item in window] for window in windows] == [
        [0, 1, 2], [3, 4, 5],
    ]


@pytest.mark.parametrize("window_size,stride", [(0, 1), (3, 0), (3, 4)])
def test_sequence_buffer_rejects_invalid_window_or_stride(window_size, stride):
    with pytest.raises(ValueError):
        PoseSequenceBuffer(window_size=window_size, stride=stride)


def test_pose_modules_import_when_numpy_is_unavailable():
    """Importing testable contracts cannot require production ML dependencies."""
    code = """
import builtins
original_import = builtins.__import__
def no_numpy(name, *args, **kwargs):
    if name == 'numpy' or name.startswith('numpy.'):
        raise ModuleNotFoundError('numpy intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_numpy
import vision.pose_pipeline
import vision.pose_buffer
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
