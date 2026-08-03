"""GMDCSA24 → mmaction2 骨架数据提取脚本。

遍历 datasets/metadata.csv 中的全部视频，使用 InputAdapter + RTMDet + RTMPose
提取 2D 关键点，输出 mmaction2 PoseC3D/STGCN 训练所需的 pickle 标注文件。

pickle 单条记录格式（参考 ntu_pose_extraction.py）:
    {
        'keypoint':        np.ndarray, shape (num_person, T, 17, 2), dtype float32
        'keypoint_score':  np.ndarray, shape (num_person, T, 17),    dtype float32
        'frame_dir':       str, 视频名（唯一标识）
        'img_shape':       tuple (H, W)
        'original_shape':  tuple (H, W)
        'total_frames':    int
        'label':           int, 0=adl, 1=fall
    }

最终聚合为：
    datasets/annotations/gmdcsa24_pose_train.pkl  (list[dict])
    datasets/annotations/gmdcsa24_pose_val.pkl
    datasets/annotations/gmdcsa24_pose_test.pkl
按受试者隔离拆分（Subject 1,2,3 训练；Subject 4 验证；留一法可切换）。

关键修复：monkey-patch platformdirs，将 YAPF 缓存重定向到项目内目录。
"""
import os
import sys
import time
import csv
import pickle
from pathlib import Path

# ---- 0. monkey-patch platformdirs（必须在 import mmcv 之前）----
project_dir = r"C:\Users\John\Desktop\look model"
yapf_cache_dir = os.path.join(project_dir, ".cache", "yapf")
os.makedirs(yapf_cache_dir, exist_ok=True)
appdata_dir = os.path.join(project_dir, ".cache", "appdata")
os.makedirs(appdata_dir, exist_ok=True)
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

# ---- 1. 项目路径 ----
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, "third_party", "mmpose"))
sys.path.insert(0, os.path.join(project_dir, "third_party", "mmdetection"))

import numpy as np
import mmengine
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

from vision.input_adapter import create_input_adapter

# ---- 2. 配置 ----
# 模型权重已迁移至 F 盘，路径集中在 paths.py 管理
from paths import DET_CHECKPOINT, POSE_CHECKPOINT
DET_CONFIG = r"third_party/mmpose/demo/mmdetection_cfg/rtmdet_tiny_8xb32-300e_coco.py"
POSE_CONFIG = r"third_party/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py"

METADATA_CSV = r"datasets/metadata.csv"
ANNO_DIR = r"datasets/annotations"
POSE_DIR = r"datasets/processed/gmdcsa24_pose"  # 单视频 pickle 缓存

DEVICE = "cuda:0"
DET_CAT_ID = 0  # COCO person
BBOX_THR = 0.3
NMS_THR = 0.3
LABEL_MAP = {"adl": 0, "fall": 1}

# 受试者隔离拆分：Subject 4 作为测试集，Subject 3 作为验证集，其余训练
# （可切换为留一法：循环 4 次，每次留一个受试者作测试）
SPLIT = {
    "train": ["1", "2"],
    "val":   ["3"],
    "test":  ["4"],
}


def process_video(video_path, detector, pose_estimator, label, frame_dir):
    """对单个视频提取关键点序列，返回 anno dict 或 None（失败）。"""
    with create_input_adapter(video_path) as adapter:
        if not adapter.is_opened:
            return None

        h, w = adapter.height, adapter.width
        total_frames = adapter.frame_count
        all_keypoints = []  # 每帧 list[(K,2)]
        all_scores = []     # 每帧 list[(K,)]

        for frame in adapter:
            # 人体检测
            det_result = inference_detector(detector, frame)
            pred = det_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate(
                (pred.bboxes, pred.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred.labels == DET_CAT_ID,
                                           pred.scores > BBOX_THR)]
            if bboxes.shape[0] == 0:
                # 当前帧无检测，用零占位
                all_keypoints.append(np.zeros((17, 2), dtype=np.float32))
                all_scores.append(np.zeros((17,), dtype=np.float32))
                continue
            bboxes = bboxes[nms(bboxes, NMS_THR), :4]

            # 关键点提取（取分数最高的一个人，单人场景）
            pose_results = inference_topdown(pose_estimator, frame, bboxes)
            data_samples = merge_data_samples(pose_results)
            pred_instances = data_samples.get('pred_instances', None)
            if pred_instances is None or len(pred_instances) == 0:
                all_keypoints.append(np.zeros((17, 2), dtype=np.float32))
                all_scores.append(np.zeros((17,), dtype=np.float32))
                continue

            # 取分数最高的实例
            # pred_instances 可能是 InstanceData，含 keypoints/keypoint_scores 字段
            kpts = np.asarray(pred_instances.keypoints)  # (N, 17, 2)
            # keypoint_scores 可能存在不同字段名
            if hasattr(pred_instances, 'keypoint_scores'):
                scores = np.asarray(pred_instances.keypoint_scores)
            elif hasattr(pred_instances, 'scores'):
                scores = np.asarray(pred_instances.scores)
            else:
                # 无分数时用 bbox_score 近似
                scores = np.ones((kpts.shape[0], kpts.shape[1]), dtype=np.float32)
            if kpts.ndim == 2:
                kpts = kpts[None]
                scores = scores[None]
            # scores 形状对齐
            if scores.ndim == 1:
                scores = np.tile(scores[:, None], (1, kpts.shape[1]))
            best_idx = int(np.argmax(scores.mean(axis=1)))
            all_keypoints.append(kpts[best_idx].astype(np.float32))
            all_scores.append(scores[best_idx].astype(np.float32))

        if not all_keypoints:
            return None

        # 对齐为 (num_person=1, T, 17, 2)
        kpts_arr = np.stack(all_keypoints, axis=0)[None]  # (1, T, 17, 2)
        scores_arr = np.stack(all_scores, axis=0)[None]    # (1, T, 17)
        actual_frames = kpts_arr.shape[1]

        anno = dict(
            keypoint=kpts_arr,
            keypoint_score=scores_arr,
            frame_dir=frame_dir,
            img_shape=(h, w),
            original_shape=(h, w),
            total_frames=actual_frames,
            label=label,
        )
        return anno


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] === GMDCSA24 姿态提取启动 ===")

    mmengine.mkdir_or_exist(ANNO_DIR)
    mmengine.mkdir_or_exist(POSE_DIR)

    # 读取 metadata
    with open(METADATA_CSV, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    # 小规模验证：通过环境变量 GMDCSA24_MAX_N 控制处理数量，默认全部
    max_n = int(os.environ.get("GMDCSA24_MAX_N", "0"))
    if max_n > 0:
        rows = rows[:max_n]
        print(f"[{time.time()-t0:.1f}s] 小规模验证模式：仅处理前 {max_n} 条")
    print(f"[{time.time()-t0:.1f}s] 读取 {len(rows)} 条 metadata")

    # 初始化模型
    print(f"[{time.time()-t0:.1f}s] 加载 RTMDet-tiny...")
    detector = init_detector(DET_CONFIG, DET_CHECKPOINT, device=DEVICE)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    print(f"[{time.time()-t0:.1f}s] 加载 RTMPose-m...")
    pose_estimator = init_pose_estimator(
        POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

    # 按受试者分组处理
    splits = {"train": [], "val": [], "test": []}
    success_count = 0
    fail_count = 0

    for i, row in enumerate(rows):
        subject = row['subject_id']
        label_str = row['label']
        video_id = row['video_id']
        category = row['category']
        video_path = os.path.join(project_dir, row['file_path'])
        frame_dir = f"S{subject}_{category}{video_id}"
        label = LABEL_MAP.get(label_str.lower())
        if label is None:
            print(f"[{time.time()-t0:.1f}s] 跳过未知标签: {label_str}")
            continue

        # 决定该样本属于哪个 split
        split_name = None
        for s, subs in SPLIT.items():
            if subject in subs:
                split_name = s
                break
        if split_name is None:
            print(f"[{time.time()-t0:.1f}s] 受试者 {subject} 未分配 split，跳过")
            continue

        # 缓存检查
        cache_path = os.path.join(POSE_DIR, f"{frame_dir}.pkl")
        if os.path.isfile(cache_path):
            try:
                anno = mmengine.load(cache_path)
                splits[split_name].append(anno)
                success_count += 1
                print(f"[{time.time()-t0:.1f}s] [{i+1}/{len(rows)}] {frame_dir} (cached) -> {split_name}")
                continue
            except Exception:
                pass  # 缓存损坏则重做

        if not os.path.isfile(video_path):
            print(f"[{time.time()-t0:.1f}s] [{i+1}/{len(rows)}] 文件不存在: {video_path}")
            fail_count += 1
            continue

        t1 = time.time()
        print(f"[{time.time()-t0:.1f}s] [{i+1}/{len(rows)}] 处理 {frame_dir} ({label_str}) ...", flush=True)
        try:
            anno = process_video(video_path, detector, pose_estimator, label, frame_dir)
        except Exception as e:
            print(f"[{time.time()-t0:.1f}s]   FAIL: {e}")
            anno = None

        if anno is None:
            fail_count += 1
            continue

        # 缓存单视频 pickle
        mmengine.dump(anno, cache_path)
        splits[split_name].append(anno)
        success_count += 1
        print(f"[{time.time()-t0:.1f}s]   OK ({anno['total_frames']} frames, {time.time()-t1:.1f}s) -> {split_name}")

    # 聚合输出
    print(f"\n[{time.time()-t0:.1f}s] === 汇总 ===")
    print(f"成功: {success_count}, 失败: {fail_count}")
    for s, annos in splits.items():
        out_path = os.path.join(ANNO_DIR, f"gmdcsa24_pose_{s}.pkl")
        mmengine.dump(annos, out_path)
        # 统计
        labels = [a['label'] for a in annos]
        n_adl = sum(1 for l in labels if l == 0)
        n_fall = sum(1 for l in labels if l == 1)
        print(f"  {s}: {len(annos)} samples (adl={n_adl}, fall={n_fall}) -> {out_path}")

    print(f"\n[{time.time()-t0:.1f}s] === 完成 ===")


if __name__ == '__main__':
    main()
