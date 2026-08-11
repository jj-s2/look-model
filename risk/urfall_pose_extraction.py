"""Extract all-valid 33-joint UR Fall pose windows from RGB zip archives."""
from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Protocol
from zipfile import BadZipFile, ZipFile

import numpy as np


class PoseDetector(Protocol):
    """Detect one person's 33 world-coordinate landmarks from an RGB image."""

    def detect(self, image: np.ndarray) -> np.ndarray | None: ...


class MediaPipePoseDetector:
    """MediaPipe Pose Landmarker wrapper yielding 33 world-coordinate joints."""

    def __init__(self, model_path: Path) -> None:
        try:
            import mediapipe as mp
        except ModuleNotFoundError as error:
            raise RuntimeError("mediapipe is required for UR Fall pose extraction") from error
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ValueError(f"Pose Landmarker model is unavailable: {model_path}")
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            num_poses=1,
            output_segmentation_masks=False,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an RGB HxWx3 array")
        result = self._landmarker.detect(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)),
        )
        if len(result.pose_world_landmarks) != 1:
            return None
        pose = np.asarray(
            [[landmark.x, landmark.y, landmark.z] for landmark in result.pose_world_landmarks[0]],
            dtype=np.float32,
        )
        return pose if pose.shape == (33, 3) and np.isfinite(pose).all() else None

    def close(self) -> None:
        self._landmarker.close()


def extract_urfall_pose_windows(
    manifest_path: Path,
    rgb_dir: Path,
    output_dir: Path,
    *,
    detector: PoseDetector,
    image_loader: Callable[[bytes], np.ndarray] | None = None,
) -> dict[str, int]:
    """Extract only complete finite pose windows from a guarded UR Fall manifest."""
    manifest_path, rgb_dir, output_dir = Path(manifest_path), Path(rgb_dir), Path(output_dir)
    if image_loader is None:
        image_loader = _decode_png
    records = _load_manifest(manifest_path)
    summary = {
        "input_windows": len(records),
        "extracted_windows": 0,
        "excluded_invalid_manifest": 0,
        "excluded_missing_source_archive": 0,
        "excluded_source_hash_mismatch": 0,
        "excluded_missing_rgb_frame": 0,
        "excluded_invalid_image": 0,
        "excluded_missing_pose": 0,
        "excluded_incomplete_sequence": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = output_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    pose_records: list[tuple[dict[str, Any], np.ndarray]] = []
    root = rgb_dir.resolve()
    for record in records:
        checked = _checked_record(record)
        if checked is None:
            summary["excluded_invalid_manifest"] += 1
            continue
        source_zip, start, end = checked
        archive_path = (root / source_zip).resolve()
        if root not in archive_path.parents or not archive_path.is_file():
            summary["excluded_missing_source_archive"] += 1
            continue
        if _sha256(archive_path) != record["source_sha256"]:
            summary["excluded_source_hash_mismatch"] += 1
            continue
        sequence = source_zip.removesuffix("-cam0-rgb.zip")
        try:
            with ZipFile(archive_path) as archive:
                frames = [
                    archive.read(f"{sequence}-cam0-rgb/{sequence}-cam0-rgb-{index + 1:03d}.png")
                    for index in range(start, end)
                ]
        except (BadZipFile, KeyError, OSError):
            summary["excluded_missing_rgb_frame"] += 1
            continue
        poses: list[np.ndarray] = []
        invalid_image = False
        for raw in frames:
            try:
                image = image_loader(raw)
            except Exception:
                invalid_image = True
                break
            try:
                pose = detector.detect(image)
            except Exception:
                pose = None
            if pose is None:
                summary["excluded_missing_pose"] += 1
                break
            pose = np.asarray(pose, dtype=np.float32)
            if pose.shape != (33, 3) or not np.isfinite(pose).all():
                summary["excluded_missing_pose"] += 1
                break
            poses.append(pose)
        else:
            pose_records.append((record, np.stack(poses)))
            continue
        if invalid_image:
            summary["excluded_invalid_image"] += 1
    complete_sequences = _complete_sequences(records, pose_records)
    output_rows: list[dict[str, Any]] = []
    for record, pose in pose_records:
        if record["sequence_id"] not in complete_sequences:
            continue
        relative = Path("windows") / f"{record['sample_id']}.npz"
        np.savez_compressed(output_dir / relative, pose=pose, label=np.int8(record["label"]))
        output_rows.append({**record, "pose_path": relative.as_posix(), "pose_shape": [len(pose), 33, 3]})
        summary["extracted_windows"] += 1
    summary["excluded_incomplete_sequence"] = sum(
        1 for record in records if isinstance(record.get("sequence_id"), str) and record["sequence_id"] not in complete_sequences
    )
    (output_dir / "pose_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _complete_sequences(records: list[dict[str, Any]], candidates: list[tuple[dict[str, Any], np.ndarray]]) -> set[str]:
    expected: dict[str, set[int]] = {}
    extracted: dict[str, set[int]] = {}
    for record in records:
        sequence, label = record.get("sequence_id"), record.get("label")
        if isinstance(sequence, str) and type(label) is int:
            expected.setdefault(sequence, set()).add(label)
    for record, _ in candidates:
        extracted.setdefault(record["sequence_id"], set()).add(record["label"])
    return {
        sequence for sequence, labels in expected.items()
        if labels == {0, 1} and extracted.get(sequence) == {0, 1}
    }


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read UR Fall manifest: {error}") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("UR Fall manifest rows must be objects")
    return sorted(rows, key=lambda row: str(row.get("sample_id", "")))


def _checked_record(record: dict[str, Any]) -> tuple[str, int, int] | None:
    source_zip = record.get("source_zip")
    source_hash = record.get("source_sha256")
    start, end, label = record.get("window_start_frame"), record.get("window_end_frame"), record.get("label")
    if not isinstance(source_zip, str) or Path(source_zip).name != source_zip:
        return None
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
        return None
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        return None
    if type(label) is not int or label not in {0, 1} or not isinstance(record.get("sample_id"), str):
        return None
    return source_zip, start, end


def _decode_png(raw: bytes) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError("opencv-python is required for UR Fall pose extraction") from error
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid PNG image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
