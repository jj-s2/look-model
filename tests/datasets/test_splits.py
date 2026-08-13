from __future__ import annotations

from dataclasses import replace

import pytest

from datasets.unified.schema import UnifiedClip, stable_clip_id
from datasets.unified.splits import (
    assert_no_leakage,
    grouped_split,
    leave_one_dataset_out,
)


def clip(
    name: str,
    *,
    dataset: str = "D",
    subject: str,
    camera: str,
    event: str,
) -> UnifiedClip:
    path = f"{subject}/{camera}/{name}.mp4"
    return UnifiedClip(
        clip_id=stable_clip_id(dataset, "1", subject, camera, path, 0.0, 1.0),
        dataset=dataset,
        dataset_version="1",
        subject_id=subject,
        camera_id=camera,
        event_group_id=event,
        media_path=path,
        start_sec=0.0,
        end_sec=1.0,
        phase="normal_adl",
        coarse_event="adl",
        hard_negative=None,
        supervision_mask=("phase", "fall_event"),
        source_label="ADL",
        provenance={"source_file": path},
    )


CLIPS = [
    clip("a", subject="S1", camera="C1", event="E1"),
    clip("b", subject="S1", camera="C2", event="E1"),
    clip("c", subject="S2", camera="C1", event="E2"),
    clip("d", subject="S3", camera="C1", event="E3"),
    clip("e", subject="S4", camera="C1", event="E4"),
]


def test_grouped_split_keeps_subject_and_camera_event_together() -> None:
    manifest = grouped_split(CLIPS, seed=42, train_ratio=0.5, val_ratio=0.25)

    assert manifest.assignments[CLIPS[0].clip_id] == manifest.assignments[CLIPS[1].clip_id]
    assert_no_leakage(manifest, CLIPS)


def test_grouped_split_is_deterministic() -> None:
    first = grouped_split(CLIPS, seed=42, train_ratio=0.5, val_ratio=0.25)
    second = grouped_split(CLIPS, seed=42, train_ratio=0.5, val_ratio=0.25)

    assert first.to_dict() == second.to_dict()


def test_leave_one_dataset_out_reserves_entire_dataset_for_test() -> None:
    clips = CLIPS + [
        replace(
            clip("f", dataset="external", subject="S1", camera="C1", event="E1"),
            media_path="external/S1/C1/f.mp4",
        )
    ]
    manifest = leave_one_dataset_out(clips, "external", seed=42)

    held_out = [item for item in clips if item.dataset == "external"]
    assert all(manifest.assignments[item.clip_id] == "test" for item in held_out)
    assert all(
        manifest.assignments[item.clip_id] != "test"
        for item in clips
        if item.dataset != "external"
    )


def test_split_rejects_invalid_ratios() -> None:
    with pytest.raises(ValueError):
        grouped_split(CLIPS, seed=42, train_ratio=0.0, val_ratio=0.5)
    with pytest.raises(ValueError):
        grouped_split(CLIPS, seed=42, train_ratio=0.8, val_ratio=0.3)
