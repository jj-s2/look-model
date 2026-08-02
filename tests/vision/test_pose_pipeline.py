from datetime import datetime, timezone

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
