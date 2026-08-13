from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from datasets.unified.adapters import get_adapter


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "datasets"

    gmd = root / "gmdcsa24"
    gmd.mkdir(parents=True)
    with (gmd / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject_id", "camera_id", "event_group_id",
                "media_path", "start_sec", "end_sec", "label",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "subject_id": "S1", "camera_id": "C1", "event_group_id": "E1",
            "media_path": "S1/ADL/01.mp4", "start_sec": "0", "end_sec": "5",
            "label": "ADL",
        })

    (root / "prevfall" / "participant_1" / "camera_45" / "Abnormal").mkdir(
        parents=True
    )
    (root / "prevfall" / "participant_1" / "camera_45" / "Abnormal" / "frame_1.jpg").write_bytes(b"")

    caucafall = root / "caucafall" / "Subject1" / "FallForwardS1"
    caucafall.mkdir(parents=True)
    (caucafall / "action.avi").write_bytes(b"")

    upfall = root / "upfall3d" / "subject1"
    upfall.mkdir(parents=True)
    (upfall / "C1S1A1T1.csv").write_text(
        "joint_1_x,joint_1_y,LABEL\n0.1,0.2,1\n0.2,0.3,1\n",
        encoding="utf-8",
    )

    omnifall = root / "omnifall"
    omnifall.mkdir(parents=True)
    (omnifall / "annotations.jsonl").write_text(
        json.dumps({
            "path": "caucafall/FallForwardS1",
            "dataset": "caucafall",
            "subject": "S1",
            "cam": "1",
            "start": 0.0,
            "end": 2.0,
            "label": "fall",
        }) + "\n",
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize(
    "adapter_name,directory",
    [
        ("gmdcsa24", "gmdcsa24"),
        ("prevfall", "prevfall"),
        ("caucafall", "caucafall"),
        ("upfall3d", "upfall3d"),
        ("omnifall", "omnifall"),
    ],
)
def test_adapter_emits_traceable_relative_clips(
    adapter_name: str, directory: str, fixture_root: Path
) -> None:
    clips = get_adapter(adapter_name).scan(fixture_root / directory)

    assert clips
    assert all(not Path(item.media_path).is_absolute() for item in clips)
    assert all(item.subject_id for item in clips)
    assert all("source_file" in item.provenance for item in clips)


def test_gmdcsa24_uses_explicit_metadata_times(fixture_root: Path) -> None:
    clips = get_adapter("gmdcsa24").scan(fixture_root / "gmdcsa24")

    assert clips[0].start_sec == 0.0
    assert clips[0].end_sec == 5.0
    assert clips[0].phase == "normal_adl"


def test_upfall_is_metadata_only_until_license_is_confirmed(fixture_root: Path) -> None:
    clips = get_adapter("upfall3d").scan(fixture_root / "upfall3d")

    assert clips[0].phase is None
    assert clips[0].supervision_mask == ()
    assert clips[0].provenance["metadata_only"] == "true"


def test_unknown_adapter_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dataset adapter"):
        get_adapter("private_dataset")
