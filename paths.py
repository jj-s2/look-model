"""集中管理模型权重与训练 checkpoint 路径。

模型文件已迁移至 F 盘（F:\\look model\\），本文件集中定义所有模型路径常量。
修改路径时只需改此文件一处，各脚本通过 import 使用。

目录结构：
  F:\\look model\\models\\pretrained\\              - 预训练权重
  F:\\look model\\experiments\\outputs\\             - 训练 checkpoint
      ├── posec3d_gmdcsa24\\                        - PoseC3D 基线
      └── losocv\\fold_S{1,2,3,4}\\                 - LOSOCV 四折
"""
import os

# ===== F 盘模型根目录 =====
F_ROOT = r"F:\look model"
MODELS_DIR = os.path.join(F_ROOT, "models", "pretrained")
CHECKPOINTS_DIR = os.path.join(F_ROOT, "experiments", "outputs")

# ===== 预训练模型权重 =====
# 人体检测（RTMDet-tiny，COCO）
DET_CHECKPOINT = os.path.join(MODELS_DIR, "rtmdet_tiny_8xb32-300e_coco.pth")
# 关键点提取（RTMPose-m，COCO 17关键点）
POSE_CHECKPOINT = os.path.join(
    MODELS_DIR,
    "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth")
# HRNet（mmpose demo 用）
HRNET_CHECKPOINT = os.path.join(
    MODELS_DIR, "hrnet_w32_coco_256x192-c78dce93_20200708.pth")
# Faster R-CNN（mmpose demo 用）
FASTER_RCNN_CHECKPOINT = os.path.join(
    MODELS_DIR, "faster_rcnn_r50_fpn_2x_coco.pth")
# PoseC3D 预训练（SlowOnly-R50, NTU60）
POSEC3D_PRETRAINED = os.path.join(
    MODELS_DIR, "slowonly_r50_u48_240e_ntu60_xsub_keypoint.pth")
# STGCN 预训练（NTU60）
STGCN_PRETRAINED = os.path.join(
    MODELS_DIR, "stgcn_8xb16-joint-u100-80e_ntu60_xsub-keypoint-2d.pth")

# ===== 训练 checkpoint 目录 =====
# PoseC3D 基线（GMDCSA24）
POSEC3D_GMDCSA24_DIR = os.path.join(CHECKPOINTS_DIR, "posec3d_gmdcsa24")
# LOSOCV 四折
LOSOCV_DIR = os.path.join(CHECKPOINTS_DIR, "losocv")


def fold_dir(fold_name):
    """获取 LOSOCV 某折的目录，如 fold_dir('fold_S1')。"""
    return os.path.join(LOSOCV_DIR, fold_name)


if __name__ == '__main__':
    # 打印所有路径，便于检查
    print("=== 模型路径配置 ===")
    print(f"F_ROOT: {F_ROOT}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")
    print("\n预训练模型:")
    for name, path in [
        ("DET_CHECKPOINT", DET_CHECKPOINT),
        ("POSE_CHECKPOINT", POSE_CHECKPOINT),
        ("HRNET_CHECKPOINT", HRNET_CHECKPOINT),
        ("FASTER_RCNN_CHECKPOINT", FASTER_RCNN_CHECKPOINT),
        ("POSEC3D_PRETRAINED", POSEC3D_PRETRAINED),
        ("STGCN_PRETRAINED", STGCN_PRETRAINED),
    ]:
        exists = "✓" if os.path.isfile(path) else "✗(未找到)"
        print(f"  {name}: {exists}")
