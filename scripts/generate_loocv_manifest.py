from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _build_subject_to_clips(lock: dict[str, Any]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    clips = lock.get("clips", [])
    if isinstance(clips, dict):
        iterator = clips.items()
    else:
        iterator = ((c.get("clip_id", str(i)), c) for i, c in enumerate(clips))
    for clip_id, meta in iterator:
        subject = str(meta.get("subject_id", "unknown"))
        mapping.setdefault(subject, []).append(clip_id)
    for clips in mapping.values():
        clips.sort()
    return mapping


def _split_three_subjects(
    subjects: list[str], val_subject: str, rng: random.Random
) -> dict[str, list[str]]:
    remaining = [s for s in subjects if s != val_subject]
    assert len(remaining) == 3
    rng.shuffle(remaining)
    return {
        "train": sorted(remaining[:2]),
        "validation": sorted(remaining[2:3]),
        "test": [val_subject],
    }


def generate_loocv_manifest(
    dataset_lock_path: Path,
    output_path: Path,
    seed: int = 42,
) -> None:
    lock = json.loads(dataset_lock_path.read_text(encoding="utf-8"))
    subject_to_clips = _build_subject_to_clips(lock)
    subjects = sorted(subject_to_clips.keys())
    if len(subjects) < 4:
        raise ValueError(f"Need at least 4 subjects for LOOCV, got {len(subjects)}")

    rng = random.Random(seed)
    folds: dict[str, dict[str, list[str]]] = {}
    for test_subject in subjects:
        split = _split_three_subjects(subjects, test_subject, rng)
        folds[test_subject] = {
            "train": sorted(
                clip for s in split["train"] for clip in subject_to_clips[s]
            ),
            "validation": sorted(
                clip for s in split["validation"] for clip in subject_to_clips[s]
            ),
            "test": sorted(clip for clip in subject_to_clips[test_subject]),
        }

    manifest = {
        "manifest_version": "1.0",
        "strategy": "subject_loocv",
        "seed": seed,
        "folds": folds,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote LOOCV manifest to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_loocv_manifest(args.dataset_lock, args.output, args.seed)


if __name__ == "__main__":
    main()
