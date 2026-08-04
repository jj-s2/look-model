from datetime import datetime, timezone

import numpy as np

from vision.ultralytics_pose import SinglePersonTracker, UltralyticsPosePipeline


class _Boxes:
    def __init__(self, boxes, scores, labels):
        self.xyxy = np.asarray(boxes, dtype=np.float32)
        self.conf = np.asarray(scores, dtype=np.float32)
        self.cls = np.asarray(labels, dtype=np.float32)


class _Keypoints:
    def __init__(self, points, scores):
        self.xy = np.asarray(points, dtype=np.float32)
        self.conf = np.asarray(scores, dtype=np.float32)


class _Result:
    def __init__(self, boxes, points, scores):
        self.boxes = _Boxes(boxes, [0.95] * len(boxes), [0] * len(boxes))
        self.keypoints = _Keypoints(points, scores)


class _FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


def _timestamp():
    return datetime(2026, 8, 4, tzinfo=timezone.utc)


def test_ultralytics_pipeline_converts_person_result_to_pose_frame():
    result = _Result(
        boxes=[[1, 2, 30, 40]],
        points=np.ones((1, 17, 2), dtype=np.float32),
        scores=np.full((1, 17), 0.9, dtype=np.float32),
    )
    model = _FakeModel(result)
    pipeline = UltralyticsPosePipeline(model=model, device="cpu")

    output = pipeline.process(np.zeros((48, 64, 3), dtype=np.uint8), _timestamp())

    assert output.frame_index == 1
    assert output.bboxes == [[1.0, 2.0, 30.0, 40.0]]
    assert len(output.keypoints) == 1
    assert len(output.keypoints[0]) == 17
    assert len(output.keypoint_scores[0]) == 17
    assert model.calls[0]["device"] == "cpu"


def test_ultralytics_pipeline_returns_empty_result_without_people():
    result = _Result(boxes=[], points=np.empty((0, 17, 2)), scores=np.empty((0, 17)))
    pipeline = UltralyticsPosePipeline(model=_FakeModel(result), device="cpu")

    output = pipeline.process(np.zeros((48, 64, 3), dtype=np.uint8), _timestamp())

    assert output.bboxes == []
    assert output.keypoints == []
    assert output.keypoint_scores == []


def test_ultralytics_pipeline_downgrades_malformed_model_output():
    class BrokenModel:
        def predict(self, **kwargs):
            raise RuntimeError("synthetic model failure")

    pipeline = UltralyticsPosePipeline(model=BrokenModel(), device="cpu")

    output = pipeline.process(np.zeros((48, 64, 3), dtype=np.uint8), _timestamp())

    assert output.bboxes == []
    assert output.payload == {"error": "pose_inference_failed"}


def test_single_person_tracker_assigns_stable_index_ids():
    tracker = SinglePersonTracker()
    frame = type("Frame", (), {"bboxes": [[0, 0, 10, 10], [20, 20, 30, 30]]})()

    assert tracker.track(frame) == [
        {"tracking_id": "person-0"},
        {"tracking_id": "person-1"},
    ]
