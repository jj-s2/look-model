from __future__ import annotations

from pathlib import Path

import pytest

from datasets.unified.schema import UnifiedClip, stable_clip_id


def make_clip(**overrides: object) -> UnifiedClip:
    values: dict[str, object] = {
        "clip_id": stable_clip_id(
            "GMDCSA24", "v2.1", "S1", "C1", "Fall/01.mp4", 1.0, 3.0
        ),
        "dataset": "GMDCSA24",
        "dataset_version": "v2.1",
        "subject_id": "S1",
        "camera_id": "C1",
        "event_group_id": "E1",
        "media_path": "Fall/01.mp4",
        "start_sec": 1.0,
        "end_sec": 3.0,
        "phase": "normal_adl",
        "coarse_event": "adl",
        "hard_negative": None,
        "supervision_mask": ("phase", "fall_event"),
        "source_label": "ADL",
        "provenance": {"source_file": "Fall/01.mp4", "source_sha256": "abc"},
    }
    values.update(overrides)
    return UnifiedClip(**values)


def test_clip_id_is_stable_for_the_same_path_and_interval() -> None:
    first = stable_clip_id("GMDCSA24", "v2.1", "S1", "C1", "Fall/01.mp4", 1.0, 3.0)
    second = stable_clip_id("GMDCSA24", "v2.1", "S1", "C1", "Fall/01.mp4", 1.0, 3.0)

    assert first == second
    assert len(first) == 64


def test_clip_id_normalizes_path_separators() -> None:
    assert stable_clip_id("d", "1", "s", "c", r"Fall\01.mp4", 1, 3) == stable_clip_id(
        "d", "1", "s", "c", "Fall/01.mp4", 1, 3
    )


def test_unknown_phase_requires_supervision_mask() -> None:
    with pytest.raises(ValueError, match="supervision_mask"):
        make_clip(phase=None, coarse_event=None, supervision_mask=())


def test_absolute_media_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="relative"):
        make_clip(media_path="F:/private/video.mp4")


def test_clip_round_trip_preserves_provenance() -> None:
    clip = make_clip()

    restored = UnifiedClip.from_dict(clip.to_dict())

    assert restored == clip
    assert restored.provenance["source_sha256"] == "abc"


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_sec": -1.0},
        {"end_sec": 1.0},
        {"subject_id": ""},
        {"phase": "unknown"},
        {"coarse_event": "other"},
    ],
)
def test_clip_rejects_invalid_fields(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_clip(**overrides)
