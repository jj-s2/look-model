"""从单视频 pkl 缓存重新生成 train/val/test 汇总 pkl。

extract_gmdcsa24_pose.py 在 155-159 视频处因 CUDA 错误崩溃，
汇总 pkl 写入不完整。本脚本从 datasets/processed/gmdcsa24_pose/
读取已成功生成的单视频 pkl，重新按受试者隔离拆分生成汇总文件。
"""
import os
import sys
import pickle
import csv
import time

project_dir = r"C:\Users\John\Desktop\look model"
POSE_DIR = os.path.join(project_dir, "datasets", "processed", "gmdcsa24_pose")
ANNO_DIR = os.path.join(project_dir, "datasets", "annotations")
METADATA_CSV = os.path.join(project_dir, "datasets", "metadata.csv")

LABEL_MAP = {"adl": 0, "fall": 1}
SPLIT = {
    "train": ["1", "2"],
    "val":   ["3"],
    "test":  ["4"],
}

def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] === 重新生成汇总 pkl ===")

    # 读取 metadata 建立 frame_dir -> (subject, label) 映射
    with open(METADATA_CSV, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    frame_dir_map = {}  # frame_dir -> (subject, label_str, split)
    for row in rows:
        subject = row['subject_id']
        video_id = row['video_id']
        category = row['category']
        label_str = row['label']
        frame_dir = f"S{subject}_{category}{video_id}"
        label = LABEL_MAP.get(label_str.lower())
        if label is None:
            continue
        split_name = None
        for s, subs in SPLIT.items():
            if subject in subs:
                split_name = s
                break
        if split_name is None:
            continue
        frame_dir_map[frame_dir] = (subject, label, split_name)

    print(f"[{time.strftime('%H:%M:%S')}] metadata 映射: {len(frame_dir_map)} 条")

    # 遍历单视频 pkl 缓存
    splits = {"train": [], "val": [], "test": []}
    success = 0
    fail = 0
    missing = []

    for frame_dir, (subject, label, split_name) in frame_dir_map.items():
        cache_path = os.path.join(POSE_DIR, f"{frame_dir}.pkl")
        if not os.path.isfile(cache_path):
            missing.append(frame_dir)
            continue
        try:
            with open(cache_path, 'rb') as f:
                anno = pickle.load(f)
            # 校验关键字段
            assert 'keypoint' in anno
            assert 'keypoint_score' in anno
            assert anno['keypoint'].ndim == 4  # (1, T, 17, 2)
            # 确保 label 正确
            anno['label'] = label
            splits[split_name].append(anno)
            success += 1
        except Exception as e:
            print(f"  [FAIL] {frame_dir}: {e}")
            fail += 1

    print(f"\n[{time.strftime('%H:%M:%S')}] === 统计 ===")
    print(f"成功: {success}")
    print(f"失败: {fail}")
    print(f"缺失: {len(missing)}")
    if missing:
        print(f"缺失列表 (前 10): {missing[:10]}")

    # 生成汇总 pkl
    print(f"\n[{time.strftime('%H:%M:%S')}] === 写入汇总 pkl ===")
    os.makedirs(ANNO_DIR, exist_ok=True)
    for s, annos in splits.items():
        out_path = os.path.join(ANNO_DIR, f"gmdcsa24_pose_{s}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(annos, f)
        n_adl = sum(1 for a in annos if a['label'] == 0)
        n_fall = sum(1 for a in annos if a['label'] == 1)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  {s}: {len(annos)} samples (adl={n_adl}, fall={n_fall}, "
              f"size={size_kb:.0f}KB) -> {out_path}")

    # 验证写入的 pkl 可读
    print(f"\n[{time.strftime('%H:%M:%S')}] === 验证可读性 ===")
    for s in ["train", "val", "test"]:
        out_path = os.path.join(ANNO_DIR, f"gmdcsa24_pose_{s}.pkl")
        with open(out_path, 'rb') as f:
            data = pickle.load(f)
        if data:
            a = data[0]
            print(f"  {s}: {len(data)} samples, 样本0 keypoint={a['keypoint'].shape}, "
                  f"total_frames={a['total_frames']}, label={a['label']}")
        else:
            print(f"  {s}: 空")

    print(f"\n[{time.strftime('%H:%M:%S')}] === 完成 (耗时 {time.time()-t0:.1f}s) ===")


if __name__ == '__main__':
    main()
