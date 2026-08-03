"""Deterministic subject- and event-grouped dataset partitions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from typing import Iterable, Literal, Mapping, Sequence

from .schema import UnifiedClip


Partition = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SplitManifest:
    strategy: str
    seed: int
    assignments: Mapping[str, Partition]
    group_assignments: Mapping[str, Partition]
    held_out_dataset: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "strategy": self.strategy,
            "seed": self.seed,
            "held_out_dataset": self.held_out_dataset,
            "assignments": dict(sorted(self.assignments.items())),
            "group_assignments": dict(sorted(self.group_assignments.items())),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def grouped_split(
    clips: Sequence[UnifiedClip],
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> SplitManifest:
    """Split clips without separating subjects or synchronized event groups."""
    _validate_ratios(train_ratio, val_ratio)
    clip_list = tuple(clips)
    if not clip_list:
        raise ValueError("clips must not be empty")
    groups = _build_groups(clip_list)
    partitions = _assign_groups(groups, seed, train_ratio, val_ratio)
    assignments = {
        item.clip_id: partitions[group_id]
        for group_id, members in groups.items()
        for item in members
    }
    manifest = SplitManifest(
        strategy="grouped",
        seed=seed,
        assignments=assignments,
        group_assignments=partitions,
    )
    assert_no_leakage(manifest, clip_list)
    return manifest


def leave_one_dataset_out(
    clips: Sequence[UnifiedClip],
    held_out_dataset: str,
    *,
    seed: int,
    train_ratio: float = 0.85,
    val_ratio: float = 0.15,
) -> SplitManifest:
    """Reserve every clip from one dataset for the test partition."""
    _validate_ratios(train_ratio, val_ratio)
    clip_list = tuple(clips)
    held_out = tuple(item for item in clip_list if item.dataset == held_out_dataset)
    remaining = tuple(item for item in clip_list if item.dataset != held_out_dataset)
    if not held_out:
        raise ValueError(f"held-out dataset not found: {held_out_dataset}")
    if not remaining:
        raise ValueError("at least one non-held-out dataset is required")
    base = grouped_split(
        remaining,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    held_groups = _build_groups(held_out)
    assignments = dict(base.assignments)
    group_assignments = dict(base.group_assignments)
    for group_id, members in held_groups.items():
        group_assignments[group_id] = "test"
        for item in members:
            assignments[item.clip_id] = "test"
    manifest = SplitManifest(
        strategy="leave_one_dataset_out",
        seed=seed,
        assignments=assignments,
        group_assignments=group_assignments,
        held_out_dataset=held_out_dataset,
    )
    assert_no_leakage(manifest, clip_list)
    return manifest


def assert_no_leakage(
    manifest: SplitManifest,
    clips: Sequence[UnifiedClip],
) -> None:
    """Raise when subject, event, or clip IDs occur in multiple partitions."""
    seen_clip_ids: set[str] = set()
    subject_partition: dict[str, Partition] = {}
    event_partition: dict[str, Partition] = {}
    for item in clips:
        if item.clip_id in seen_clip_ids:
            raise ValueError(f"duplicate clip_id: {item.clip_id}")
        seen_clip_ids.add(item.clip_id)
        try:
            partition = manifest.assignments[item.clip_id]
        except KeyError as error:
            raise ValueError(f"clip missing from split manifest: {item.clip_id}") from error
        for key, target in (
            (f"{item.dataset}:{item.subject_id}", subject_partition),
            (f"{item.dataset}:event:{item.event_group_id}", event_partition),
        ):
            previous = target.setdefault(key, partition)
            if previous != partition:
                raise ValueError(f"data leakage detected for group: {key}")


def _build_groups(clips: Sequence[UnifiedClip]) -> dict[str, tuple[UnifiedClip, ...]]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for item in clips:
        subject_key = f"{item.dataset}:{item.subject_id}"
        event_key = f"{item.dataset}:event:{item.event_group_id}"
        union(subject_key, event_key)

    grouped: dict[str, list[UnifiedClip]] = {}
    for item in clips:
        root = find(f"{item.dataset}:{item.subject_id}")
        grouped.setdefault(root, []).append(item)
    return {
        group_id: tuple(sorted(members, key=lambda item: item.clip_id))
        for group_id, members in grouped.items()
    }


def _assign_groups(
    groups: Mapping[str, tuple[UnifiedClip, ...]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, Partition]:
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    counts = _partition_counts(len(group_ids), train_ratio, val_ratio)
    partitions: list[Partition] = (
        ["train"] * counts["train"]
        + ["validation"] * counts["validation"]
        + ["test"] * counts["test"]
    )
    return {
        group_id: partitions[index]
        for index, group_id in enumerate(group_ids)
    }


def _partition_counts(
    count: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[Partition, int]:
    if count < 3:
        return {"train": count, "validation": 0, "test": 0}
    ratios = {
        "train": train_ratio,
        "validation": val_ratio,
        "test": 1.0 - train_ratio - val_ratio,
    }
    counts: dict[Partition, int] = {
        name: int(ratio * count) if ratio > 1e-12 else 0
        for name, ratio in ratios.items()
    }
    for name, ratio in ratios.items():
        if ratio > 1e-12 and counts[name] == 0:
            counts[name] = 1
    while sum(counts.values()) > count:
        largest = max(counts, key=counts.get)
        if counts[largest] > 1:
            counts[largest] -= 1
        else:
            break
    while sum(counts.values()) < count:
        largest = max(ratios, key=ratios.get)
        counts[largest] += 1
    return counts  # type: ignore[return-value]


def _validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if not 0.0 < train_ratio < 1.0 or not 0.0 <= val_ratio < 1.0:
        raise ValueError("train_ratio and val_ratio must be within [0, 1]")
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must not exceed 1")
