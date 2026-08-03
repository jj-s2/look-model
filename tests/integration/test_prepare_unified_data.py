from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.prepare_unified_fall_data import prepare_unified_data


def write_fixture_root(root: Path) -> Path:
    gmd = root / "GMDCSA24"
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

    upfall = root / "UP-Fall-3D-Skeletons" / "subject1"
    upfall.mkdir(parents=True)
    (upfall / "C1S1A1T1.csv").write_text(
        "joint_1_x,joint_1_y,LABEL\n0.1,0.2,1\n",
        encoding="utf-8",
    )
    return root


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_prepare_fixture_is_deterministic(tmp_path: Path) -> None:
    fixtures = write_fixture_root(tmp_path / "fixtures")

    first = prepare_unified_data(fixtures, tmp_path / "one", seed=42, demo=True)
    second = prepare_unified_data(fixtures, tmp_path / "two", seed=42, demo=True)

    assert read_json(first / "dataset_lock.json") == read_json(second / "dataset_lock.json")
    assert read_json(first / "split_manifest.json") == read_json(second / "split_manifest.json")


def test_restricted_dataset_is_excluded_from_training(tmp_path: Path) -> None:
    output = prepare_unified_data(
        write_fixture_root(tmp_path / "fixtures"),
        tmp_path / "output",
        seed=42,
        demo=True,
    )
    split = read_json(output / "split_manifest.json")
    restricted_ids = {
        item["clip_id"]
        for item in split["clips"]
        if item["dataset"] == "UP-Fall-3D-Skeletons"
    }

    assert not restricted_ids.intersection(split["partitions"]["train"])
    assert restricted_ids == set(split["excluded"])


def test_fixture_output_is_not_release_eligible(tmp_path: Path) -> None:
    output = prepare_unified_data(
        write_fixture_root(tmp_path / "fixtures"),
        tmp_path / "output",
        seed=42,
        demo=True,
    )
    lock = read_json(output / "dataset_lock.json")

    assert lock["demo"] is True
    assert lock["release_eligible"] is False
