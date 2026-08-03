"""Build deterministic, license-gated unified fall annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

from datasets.unified.adapters import get_adapter
from datasets.unified.registry import DatasetRegistry, load_registry
from datasets.unified.schema import UnifiedClip
from datasets.unified.splits import SplitManifest, grouped_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "datasets" / "manifest.json"


def prepare_unified_data(
    fixtures_root: Path,
    output_dir: Path,
    seed: int,
    *,
    demo: bool = False,
    manifest_path: Path = MANIFEST_PATH,
) -> Path:
    """Scan adapter directories and write deterministic lock/split files."""
    root = Path(fixtures_root)
    output = Path(output_dir)
    registry = load_registry(Path(manifest_path))
    clips = _scan_directories(root)
    if not clips:
        raise ValueError(f"no adapter-readable data found under {root}")
    output.mkdir(parents=True, exist_ok=True)
    build_dataset_lock(registry, clips, output, demo=demo)
    trainable = [
        clip for clip in clips
        if _entry_can_train(registry, clip.dataset)
    ]
    excluded = [clip for clip in clips if clip not in trainable]
    build_split_manifest(
        trainable,
        strategy="grouped",
        seed=seed,
        output_dir=output,
        all_clips=clips,
        excluded=excluded,
    )
    return output


def build_dataset_lock(
    registry: DatasetRegistry,
    clips: Sequence[UnifiedClip],
    output_dir: Path,
    *,
    demo: bool = False,
) -> Path:
    """Write a reproducible lock without timestamps or machine paths."""
    counts: dict[str, int] = {}
    for clip in clips:
        counts[clip.dataset] = counts.get(clip.dataset, 0) + 1
    datasets = [
        {
            "name": entry.name,
            "version": entry.version,
            "license_status": entry.license_status,
            "can_train": entry.can_train,
            "download_status": entry.download_status,
            "clip_count": counts.get(entry.name, 0),
        }
        for entry in sorted(registry.entries, key=lambda item: item.name)
    ]
    payload = {
        "schema_version": "1.0",
        "registry_schema_version": registry.schema_version,
        "demo": demo,
        "release_eligible": False,
        "datasets": datasets,
        "clips": [
            clip.to_dict()
            for clip in sorted(clips, key=lambda item: item.clip_id)
        ],
    }
    path = Path(output_dir) / "dataset_lock.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def build_split_manifest(
    clips: Sequence[UnifiedClip],
    strategy: str,
    seed: int,
    output_dir: Path,
    *,
    all_clips: Sequence[UnifiedClip] | None = None,
    excluded: Sequence[UnifiedClip] = (),
) -> Path:
    if strategy != "grouped":
        raise ValueError(f"unsupported split strategy: {strategy}")
    split: SplitManifest = grouped_split(
        clips,
        seed=seed,
        train_ratio=0.70,
        val_ratio=0.15,
    )
    assignments = split.assignments
    all_items = tuple(all_clips or clips)
    payload = {
        "schema_version": "1.0",
        "strategy": split.strategy,
        "seed": seed,
        "clips": [
            {
                "clip_id": clip.clip_id,
                "dataset": clip.dataset,
                "subject_id": clip.subject_id,
                "event_group_id": clip.event_group_id,
            }
            for clip in sorted(all_items, key=lambda item: item.clip_id)
        ],
        "partitions": {
            name: sorted(
                clip_id for clip_id, partition in assignments.items()
                if partition == name
            )
            for name in ("train", "validation", "test")
        },
        "excluded": sorted(clip.clip_id for clip in excluded),
        "group_assignments": dict(sorted(split.group_assignments.items())),
    }
    path = Path(output_dir) / "split_manifest.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _scan_directories(root: Path) -> tuple[UnifiedClip, ...]:
    if not root.exists() or not root.is_dir():
        raise ValueError(f"dataset root must be an existing directory: {root}")
    clips: list[UnifiedClip] = []
    for directory in sorted(item for item in root.iterdir() if item.is_dir()):
        try:
            adapter = get_adapter(directory.name)
        except ValueError:
            continue
        clips.extend(adapter.scan(directory))
    return tuple(sorted(clips, key=lambda item: item.clip_id))


def _entry_can_train(registry: DatasetRegistry, dataset: str) -> bool:
    try:
        return registry[dataset].can_train
    except KeyError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="构建统一跌倒数据集锁和划分")
    parser.add_argument("--fixtures", "--root", dest="root", required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=("grouped",), default="grouped")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_unified_data(
        args.root,
        args.output,
        args.seed,
        demo="fixture" in str(args.root).lower(),
        manifest_path=args.manifest,
    )
    print(f"[OK] unified annotations written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
