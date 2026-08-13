from pathlib import Path

import numpy as np
import pytest

from risk.prefall_dataset import build_prefall_rows, duration_by_media_from_metadata


def _pose(frames: int = 64):
    sequence = np.zeros((frames, 17, 3), dtype=np.float32)
    for index in range(frames):
        sequence[index, :, 0] = index
        sequence[index, :, 1] = index * 2
        sequence[index, :, 2] = 1.0
    return sequence


def test_fall_row_uses_only_frames_before_guard_boundary(tmp_path: Path):
    np.savez(tmp_path / "fall.npz", long_pose=_pose(), short_embedding=np.zeros(512))
    clips = [{
        "media_path": "fall.npz", "event_group_id": "subject-1:fall:01",
        "subject_id": "subject-1", "coarse_event": "fall", "start_sec": 6.0,
    }]

    rows, audit = build_prefall_rows(
        clips, {"fall.npz": 10.0}, tmp_path, horizon_sec=3.0, guard_sec=0.5,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == 1
    assert row["window_start_frame"] == 16
    assert row["window_end_frame"] == 36
    assert row["window_end_sec"] == pytest.approx(5.5)
    assert audit["excluded_no_window"] == 0


def test_adl_row_is_negative_and_never_uses_fall_guard_logic(tmp_path: Path):
    np.savez(tmp_path / "adl.npz", long_pose=_pose(), short_embedding=np.zeros(512))
    rows, audit = build_prefall_rows(
        [{"media_path": "adl.npz", "event_group_id": "subject-1:adl:01",
          "subject_id": "subject-1", "coarse_event": "adl", "start_sec": 0.0}],
        {"adl.npz": 10.0}, tmp_path,
    )

    assert rows[0]["label"] == 0
    assert rows[0]["window_start_frame"] == 0
    assert rows[0]["window_end_frame"] == 64
    assert audit["negative_rows"] == 1


def test_missing_duration_or_too_short_prefall_window_is_excluded(tmp_path: Path):
    np.savez(tmp_path / "fall.npz", long_pose=_pose(), short_embedding=np.zeros(512))
    clips = [
        {"media_path": "missing-duration.npz", "event_group_id": "a", "subject_id": "a",
         "coarse_event": "fall", "start_sec": 5.0},
        {"media_path": "fall.npz", "event_group_id": "b", "subject_id": "b",
         "coarse_event": "fall", "start_sec": 0.4},
    ]

    rows, audit = build_prefall_rows(clips, {"fall.npz": 10.0}, tmp_path)

    assert rows == []
    assert audit["excluded_missing_duration"] == 1
    assert audit["excluded_no_window"] == 1


def test_metadata_duration_mapping_uses_original_video_duration():
    durations = duration_by_media_from_metadata([
        {"subject_id": "2", "category": "FALL", "video_id": "07", "duration_seconds": "12.5"},
        {"subject_id": "2", "category": "ADL", "video_id": "03", "duration_seconds": "8.0"},
    ])

    assert durations == {"subject-2_fall_07.npz": 12.5, "subject-2_adl_03.npz": 8.0}
