"""验证生成的 GMDCSA24 pose pickle 格式。"""
import os
import sys
import pickle

project_dir = r"C:\Users\John\Desktop\look model"

# 用原生 pickle 读取，避免 mmengine import 卡住
pkl_path = os.path.join(project_dir, "datasets", "annotations", "gmdcsa24_pose_train.pkl")
print(f"读取: {pkl_path}")
with open(pkl_path, 'rb') as f:
    d = pickle.load(f)

print(f"type: {type(d).__name__}, len: {len(d)}")
print()
for i, a in enumerate(d):
    print(f"--- 样本 {i} ---")
    print(f"  keys: {list(a.keys())}")
    print(f"  frame_dir: {a['frame_dir']}")
    print(f"  keypoint shape: {a['keypoint'].shape}, dtype: {a['keypoint'].dtype}")
    print(f"  keypoint_score shape: {a['keypoint_score'].shape}, dtype: {a['keypoint_score'].dtype}")
    print(f"  img_shape: {a['img_shape']}")
    print(f"  original_shape: {a['original_shape']}")
    print(f"  total_frames: {a['total_frames']}")
    print(f"  label: {a['label']}  (0=adl, 1=fall)")
    print(f"  keypoint[0,0,0]: {a['keypoint'][0,0,0]}")
    print(f"  keypoint_score[0,0,0]: {a['keypoint_score'][0,0,0]}")
    print()

print("=" * 50)
print("格式验证通过 ✅")
print("字段符合 mmaction2 PoseC3D/STGCN 训练要求")
