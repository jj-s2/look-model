"""Extract GPU pose caches and normalized metadata from the official GMDCSA24 zip."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import Iterable

import numpy as np


SEGMENT_RE = re.compile(r"([^\[;:]+?)\s*\[\s*([0-9.]+)\s*to\s*([0-9.]+)\s*\]")
METADATA_FIELDS = ("media_path", "subject_id", "camera_id", "event_group_id", "start_sec", "end_sec", "label")


def parse_class_segments(text: str) -> list[tuple[str, float, float]]:
    result = []
    for label, start, end in SEGMENT_RE.findall(text or ""):
        normalized = label.strip().lower()
        coarse = "fall" if normalized.startswith("falling") else "adl"
        result.append((coarse, float(start), float(end)))
    return result


def pad_or_sample_pose(pose: np.ndarray, *, frames: int = 64) -> np.ndarray:
    if pose.ndim != 3 or pose.shape[1:] != (17, 3):
        raise ValueError("pose must have shape (time, 17, 3)")
    if len(pose) == 0:
        return np.zeros((frames, 17, 3), dtype=np.float32)
    indices = np.linspace(0, len(pose) - 1, frames).round().astype(int)
    return pose[indices].astype(np.float32, copy=False)


def _cache_name(subject: str, category: str, stem: str) -> str:
    """Build a collision-free cache name (ADL/ Fall share numeric stems)."""
    return f"{subject}_{category.lower()}_{stem}.npz"


def _video_pose(model, video: Path, *, device: str) -> np.ndarray:
    results = model.predict(source=str(video), stream=True, device=device, verbose=False, imgsz=512, vid_stride=2)
    frames = []
    for result in results:
        if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
            frames.append(np.zeros((17, 3), dtype=np.float32))
            continue
        xy = result.keypoints.xy.detach().cpu().numpy()
        conf = result.keypoints.conf.detach().cpu().numpy() if result.keypoints.conf is not None else np.ones(xy.shape[:2])
        person = int(np.nanmean(conf, axis=1).argmax())
        frames.append(np.concatenate([xy[person], conf[person, :, None]], axis=1).astype(np.float32))
    return np.asarray(frames, dtype=np.float32)


def extract_dataset(dataset_root: Path, output_root: Path, *, device: str = "0", limit: int | None = None) -> Path:
    from ultralytics import YOLO
    model = YOLO("yolo11n-pose.pt")
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = output_root / "metadata.csv"
    existing_rows = {}
    if metadata.exists():
        with metadata.open(encoding="utf-8", newline="") as handle:
            existing_rows = {row.get("media_path", ""): row for row in csv.DictReader(handle)}
    if not metadata.exists():
        with metadata.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=METADATA_FIELDS).writeheader()
    videos = sorted(dataset_root.glob("Subject */*/**/*.mp4"))
    if limit is not None:
        videos = videos[:limit]
    for index, video in enumerate(videos, start=1):
        subject = video.parts[-3].replace("Subject ", "subject-")
        category = video.parent.name
        label_csv = video.parents[1] / ("Fall.csv" if category == "Fall" else "ADL.csv")
        row = next(item for item in csv.DictReader(label_csv.open(encoding="utf-8-sig")) if item["File Name"].strip() == video.name)
        segments = parse_class_segments(row.get(" Classes", row.get("Classes", "")))
        start, end = (0.0, float(row.get("Length (seconds)", 1)))
        fall_segments = [item for item in segments if item[0] == "fall"]
        if fall_segments:
            start, end = fall_segments[0][1], fall_segments[0][2]
        relative = _cache_name(subject, category, video.stem)
        row = {"media_path": relative, "subject_id": subject, "camera_id": "gmdcsa24", "event_group_id": video.stem, "start_sec": start, "end_sec": end, "label": "Falling" if fall_segments else "ADL"}
        if (output_root / relative).exists():
            if relative not in existing_rows:
                with metadata.open("a", encoding="utf-8", newline="") as handle:
                    csv.DictWriter(handle, fieldnames=METADATA_FIELDS).writerow(row)
                existing_rows[relative] = row
            print(f"[{index}/{len(videos)}] {video.name} (cached)")
            continue
        pose = pad_or_sample_pose(_video_pose(model, video, device=device), frames=64)
        np.savez_compressed(output_root / relative, long_pose=pose, short_embedding=np.zeros(512, dtype=np.float32))
        with metadata.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=METADATA_FIELDS).writerow(row)
        existing_rows[relative] = row
        print(f"[{index}/{len(videos)}] {video.name}", flush=True)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    extract_dataset(args.dataset_root, args.output, device=args.device, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
