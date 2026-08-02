"""端到端 pipeline：本地视频 → InputAdapter → RTMDet 人体检测 → RTMPose 17关键点 → JSON + 可视化视频。

本脚本验证 vision/input_adapter.py 与 OpenMMLab 推理链路的集成。
关键修复：monkey-patch platformdirs，将 YAPF 缓存重定向到项目内目录，
避免 TRAE Sandbox 限制系统目录访问导致 import mmcv 卡住。
"""
import os
import sys
import time
from pathlib import Path

# ---- 0. 先 monkey-patch platformdirs，必须在 import mmcv/mmdet/mmpose 之前 ----
project_dir = r"C:\Users\John\Desktop\look model"
yapf_cache_dir = os.path.join(project_dir, ".cache", "yapf")
os.makedirs(yapf_cache_dir, exist_ok=True)
appdata_dir = os.path.join(project_dir, ".cache", "appdata")
os.makedirs(appdata_dir, exist_ok=True)
# 通过 LOCALAPPDATA 环境变量重定向，使子进程 platformdirs 也指向项目内
os.environ["LOCALAPPDATA"] = appdata_dir

import platformdirs

def _patched_user_cache_dir(appname=None, appauthor=None, version=None,
                            opinion=True, ensure_exists=False):
    path = yapf_cache_dir
    params = []
    if appauthor is not False:
        author = appauthor or appname or ""
        if author:
            params.append(author)
    if appname:
        params.append(appname)
    if opinion:
        params.append("Cache")
    if version:
        params.append(version)
    if params:
        path = os.path.join(path, *params)
    os.makedirs(path, exist_ok=True)
    return path

platformdirs.user_cache_dir = _patched_user_cache_dir
import platformdirs.windows
@property
def _patched_win_cache(self):
    return self._append_parts(yapf_cache_dir, opinion_value="Cache")
platformdirs.windows.Windows.user_cache_dir = _patched_win_cache

# ---- 1. 项目路径设置 ----
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, "third_party", "mmpose"))
sys.path.insert(0, os.path.join(project_dir, "third_party", "mmdetection"))

import cv2
import numpy as np
import json_tricks as json
import mmcv
import mmengine

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

from vision.input_adapter import create_input_adapter

# ---- 2. 配置 ----
# 模型权重已迁移至 F 盘，路径集中在 paths.py 管理
sys.path.insert(0, project_dir)
from paths import DET_CHECKPOINT, POSE_CHECKPOINT
DET_CONFIG = r"third_party/mmpose/demo/mmdetection_cfg/rtmdet_tiny_8xb32-300e_coco.py"
POSE_CONFIG = r"third_party/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py"

INPUT_VIDEO = r"third_party/mmpose/demo/resources/demo.mp4"
OUTPUT_DIR = r"experiments/outputs/pipeline_demo"
DEVICE = "cuda:0"
DET_CAT_ID = 0  # COCO person
BBOX_THR = 0.3
NMS_THR = 0.3
KPT_THR = 0.3


def process_one_frame(frame, detector, pose_estimator, visualizer):
    """单帧推理：检测人体 → 提取关键点 → 可视化。返回 (pred_instances, vis_image)。"""
    # 人体检测
    det_result = inference_detector(detector, frame)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == DET_CAT_ID,
                                   pred_instance.scores > BBOX_THR)]
    bboxes = bboxes[nms(bboxes, NMS_THR), :4]

    # 关键点提取
    pose_results = inference_topdown(pose_estimator, frame, bboxes)
    data_samples = merge_data_samples(pose_results)

    # 可视化
    img_rgb = mmcv.bgr2rgb(frame)
    visualizer.add_datasample(
        'result', img_rgb, data_sample=data_samples,
        draw_gt=False, draw_heatmap=False, draw_bbox=False,
        show_kpt_idx=False, skeleton_style='mmpose',
        show=False, wait_time=0, kpt_thr=KPT_THR)
    vis_image = visualizer.get_image()

    return data_samples.get('pred_instances', None), vis_image


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] === 端到端 pipeline 启动 ===")

    # 创建输出目录
    mmengine.mkdir_or_exist(OUTPUT_DIR)
    output_video = os.path.join(OUTPUT_DIR, "demo_pipeline.mp4")
    pred_save_path = os.path.join(OUTPUT_DIR, "demo_pipeline_keypoints.json")

    # 初始化检测器
    print(f"[{time.time()-t0:.1f}s] 加载 RTMDet-tiny 检测器...")
    detector = init_detector(DET_CONFIG, DET_CHECKPOINT, device=DEVICE)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # 初始化姿态估计器
    print(f"[{time.time()-t0:.1f}s] 加载 RTMPose-m 姿态估计器...")
    pose_estimator = init_pose_estimator(
        POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

    # 初始化可视化器
    pose_estimator.cfg.visualizer.radius = 3
    pose_estimator.cfg.visualizer.alpha = 0.8
    pose_estimator.cfg.visualizer.line_width = 1
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style='mmpose')

    # 用 InputAdapter 打开视频
    print(f"[{time.time()-t0:.1f}s] 用 InputAdapter 打开视频: {INPUT_VIDEO}")
    video_writer = None
    pred_instances_list = []
    frame_idx = 0

    with create_input_adapter(INPUT_VIDEO) as adapter:
        print(f"[{time.time()-t0:.1f}s] 输入元信息: {adapter.meta()}")
        print(f"[{time.time()-t0:.1f}s] 开始逐帧推理 ({adapter.frame_count} 帧)...")

        for frame in adapter:
            frame_idx += 1
            pred_instances, vis_image = process_one_frame(
                frame, detector, pose_estimator, visualizer)

            # 保存关键点
            pred_instances_list.append(
                dict(frame_id=frame_idx,
                     instances=split_instances(pred_instances)))

            # 写入可视化视频
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    output_video, fourcc, adapter.fps or 25,
                    (vis_image.shape[1], vis_image.shape[0]))
            video_writer.write(mmcv.rgb2bgr(vis_image))

            print(f"[{time.time()-t0:.1f}s] frame {frame_idx}/{adapter.frame_count} "
                  f"done", flush=True)

    # 释放视频写入器
    if video_writer:
        video_writer.release()

    # 保存关键点 JSON
    with open(pred_save_path, 'w') as f:
        json.dump(
            dict(meta_info=pose_estimator.dataset_meta,
                 instance_info=pred_instances_list),
            f, indent='\t')

    print(f"[{time.time()-t0:.1f}s] === pipeline 完成 ===")
    print(f"输出视频: {output_video} ({os.path.getsize(output_video)} bytes)")
    print(f"关键点 JSON: {pred_save_path} ({os.path.getsize(pred_save_path)} bytes)")
    print(f"总帧数: {frame_idx}")


if __name__ == '__main__':
    main()
