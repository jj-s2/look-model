"""Run local-video person detection and pose estimation.

Heavy OpenMMLab imports and model initialization occur only after CLI parsing,
so ``--help`` is safe in a minimal test environment.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local pose-estimation pipeline.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", "--input-video", dest="input_video", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _configure_openmmlab_imports(project_root: Path) -> None:
    """Keep OpenMMLab/YAPF cache writes inside the project before importing it."""
    project_dir = str(project_root.expanduser().resolve())
    cache_dir = os.path.join(project_dir, ".cache", "yapf")
    appdata_dir = os.path.join(project_dir, ".cache", "appdata")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(appdata_dir, exist_ok=True)
    os.environ["LOCALAPPDATA"] = appdata_dir
    sys.path[:0] = [project_dir, os.path.join(project_dir, "third_party", "mmpose"), os.path.join(project_dir, "third_party", "mmdetection")]


def create_openmmlab_pipeline(project_root: Path, device: str):
    """Create injected adapters and the visualizer on the real-model path only."""
    _configure_openmmlab_imports(project_root)
    import numpy as np
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_topdown, init_model as init_pose_estimator
    from mmpose.evaluation.functional import nms
    from mmpose.registry import VISUALIZERS
    from mmpose.structures import merge_data_samples
    from mmpose.utils import adapt_mmdet_pipeline
    from paths import DET_CHECKPOINT, POSE_CHECKPOINT
    from vision.pose_pipeline import Detection, PoseInferenceOutput, PosePipeline

    det_config = "third_party/mmpose/demo/mmdetection_cfg/rtmdet_tiny_8xb32-300e_coco.py"
    pose_config = "third_party/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py"
    detector = init_detector(det_config, DET_CHECKPOINT, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(
        pose_config, POSE_CHECKPOINT, device=device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
    )

    class DetectorAdapter:
        def detect(self, frame):
            instances = inference_detector(detector, frame).pred_instances.cpu().numpy()
            boxes = np.concatenate((instances.bboxes, instances.scores[:, None]), axis=1)
            person_boxes = boxes[np.logical_and(instances.labels == 0, instances.scores > 0.3)]
            person_boxes = person_boxes[nms(person_boxes, 0.3), :4]
            return [Detection("person", 1.0, box.tolist()) for box in person_boxes]

    class EstimatorAdapter:
        def estimate(self, frame, boxes: Sequence[Sequence[float]]):
            samples = merge_data_samples(inference_topdown(pose_estimator, frame, np.asarray(boxes)))
            instances = samples.get("pred_instances", None)
            return PoseInferenceOutput(
                keypoints=[] if instances is None else instances.keypoints.tolist(),
                keypoint_scores=[] if instances is None else instances.keypoint_scores.tolist(),
                payload=samples,
            )

    pose_estimator.cfg.visualizer.radius = 3
    pose_estimator.cfg.visualizer.alpha = 0.8
    pose_estimator.cfg.visualizer.line_width = 1
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style="mmpose")
    return PosePipeline(DetectorAdapter(), EstimatorAdapter()), visualizer, pose_estimator.dataset_meta


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    pipeline, visualizer, dataset_meta = create_openmmlab_pipeline(args.project_root, args.device)
    import cv2
    import json_tricks as json
    import mmcv
    import mmengine
    from mmpose.structures import split_instances
    from vision.input_adapter import create_input_adapter

    input_video = str(args.input_video or (args.project_root / "third_party" / "mmpose" / "demo" / "resources" / "demo.mp4"))
    output_dir = str(args.output_dir or (args.project_root / "experiments" / "outputs" / "pipeline_demo"))
    mmengine.mkdir_or_exist(output_dir)
    output_video = os.path.join(output_dir, "demo_pipeline.mp4")
    prediction_path = os.path.join(output_dir, "demo_pipeline_keypoints.json")
    writer = None
    predictions: list[dict[str, Any]] = []

    with create_input_adapter(input_video) as adapter:
        input_meta = adapter.meta()
        for frame in adapter:
            result = pipeline.process(frame, datetime.now(timezone.utc))
            samples = result.payload
            instances = samples.get("pred_instances", None)
            predictions.append(dict(frame_id=result.frame_index, instances=split_instances(instances)))
            image = mmcv.bgr2rgb(frame)
            visualizer.add_datasample("result", image, data_sample=samples, draw_gt=False,
                draw_heatmap=False, draw_bbox=False, show_kpt_idx=False,
                skeleton_style="mmpose", show=False, wait_time=0, kpt_thr=0.3)
            image = visualizer.get_image()
            if writer is None:
                writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), adapter.fps or 25, (image.shape[1], image.shape[0]))
            writer.write(mmcv.rgb2bgr(image))

    if writer:
        writer.release()
    with open(prediction_path, "w") as file:
        json.dump(dict(meta_info=dataset_meta, instance_info=predictions, run_info=dict(
            generated_at=datetime.now(timezone.utc).isoformat(), model_versions={"detector": "RTMDet-tiny", "pose": "RTMPose-m"}, input_quality=input_meta,
        )), file, indent="\t")
    print(f"[{time.time() - t0:.1f}s] pipeline complete: {prediction_path}")


if __name__ == "__main__":
    main()
