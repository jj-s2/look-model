"""
扫描 GMDCSA24 数据集目录、视频分辨率、帧率，生成 datasets/metadata.csv。

输出字段：
  subject_id, category, video_id, file_path, file_size_bytes,
  duration_seconds, fps, width, height, attire, time_of_recording,
  description, classes_raw, source_dataset, label

label 字段统一为 fall / adl。
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:
    print("[ERROR] 需要 opencv-python: pip install opencv-python",
          file=sys.stderr)
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "raw" / "GMDCSA24" / "extracted" / \
    "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
OUTPUT_CSV = PROJECT_ROOT / "datasets" / "metadata.csv"

LABEL_MAP = {
    "Fall": "fall",
    "ADL": "adl",
}


def parse_csv_annotations(csv_path: Path) -> dict[str, dict]:
    """解析 Subject 目录下的 ADL.csv / Fall.csv。返回 {file_name: row}。

    注意：CSV 表头中部分列名带前导空格（如 ` Classes`），
    这里统一去除键的前后空格，避免取不到值。
    """
    if not csv_path.exists():
        return {}
    result = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # 去除每个字段名的前后空格
        reader.fieldnames = [
            (n or "").strip() for n in reader.fieldnames or []
        ]
        for row in reader:
            # 同时去除键的前后空格
            cleaned = {(k or "").strip(): v for k, v in row.items()}
            fname = (cleaned.get("File Name") or "").strip()
            if fname:
                result[fname] = cleaned
    return result


def probe_video(path: Path) -> tuple[Optional[float], Optional[int],
                                    Optional[int], Optional[int]]:
    """返回 (duration_s, fps, width, height)。失败返回 None。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None, None, None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = (n / fps) if (fps and n > 0) else None
        return duration, fps, w, h
    finally:
        cap.release()


def scan_subject(subject_dir: Path, subject_id: str) -> list[dict]:
    rows = []
    for category in ("ADL", "Fall"):
        cat_dir = subject_dir / category
        if not cat_dir.exists():
            continue
        csv_path = subject_dir / f"{category}.csv"
        annotations = parse_csv_annotations(csv_path)

        for video_path in sorted(cat_dir.glob("*.mp4")):
            ann = annotations.get(video_path.name, {})
            duration, fps, w, h = probe_video(video_path)
            rows.append({
                "subject_id": subject_id,
                "category": category,
                "video_id": video_path.stem,
                "file_path": str(video_path.relative_to(PROJECT_ROOT)),
                "file_size_bytes": video_path.stat().st_size,
                "duration_seconds": duration,
                "fps": fps,
                "width": w,
                "height": h,
                "attire": (ann.get("Attire") or "").strip(),
                "time_of_recording": (ann.get("Time of Recording") or "").strip(),
                "description": (ann.get("Description") or "").strip(),
                "classes_raw": (ann.get("Classes") or "").strip(),
                "source_dataset": "GMDCSA24",
                "label": LABEL_MAP[category],
            })
    return rows


def main() -> int:
    if not DATASET_ROOT.exists():
        print(f"[ERROR] 数据集目录不存在: {DATASET_ROOT}", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    for subject_id in ("1", "2", "3", "4"):
        subject_dir = DATASET_ROOT / f"Subject {subject_id}"
        if not subject_dir.exists():
            print(f"[WARN] {subject_dir} 不存在，跳过", file=sys.stderr)
            continue
        rows = scan_subject(subject_dir, subject_id)
        print(f"[Subject {subject_id}] 扫描到 {len(rows)} 个视频")
        all_rows.extend(rows)

    if not all_rows:
        print("[ERROR] 未扫描到任何视频", file=sys.stderr)
        return 1

    fieldnames = list(all_rows[0].keys())
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # 统计汇总
    from collections import Counter
    subject_counts = Counter(r["subject_id"] for r in all_rows)
    label_counts = Counter(r["label"] for r in all_rows)
    fps_values = [r["fps"] for r in all_rows if r["fps"]]
    resolutions = Counter(
        f"{r['width']}x{r['height']}" for r in all_rows
        if r["width"] and r["height"]
    )
    total_size = sum(r["file_size_bytes"] for r in all_rows)

    print("\n========== 扫描汇总 ==========")
    print(f"总视频数: {len(all_rows)}")
    print(f"按受试者: {dict(subject_counts)}")
    print(f"按标签:   {dict(label_counts)}")
    print(f"分辨率:   {dict(resolutions)}")
    print(f"FPS 范围: {min(fps_values):.2f} ~ {max(fps_values):.2f}")
    print(f"总大小:   {total_size / (1<<30):.2f} GB")
    print(f"输出:     {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
