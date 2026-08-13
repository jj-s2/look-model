import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from risk.upfall_prefall_dataset import build_upfall_prefall_dataset, parse_upfall_identity


def _write_upfall_csv(path: Path, *, frames: int = 18, onset: int | None = 10,
                      nonfinite_frame: int | None = None) -> None:
    header = [f"Joint{joint}_{axis}" for joint in range(1, 34) for axis in ("X", "Y", "Z")]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([*header, "LABEL"])
        for frame in range(frames):
            values = [float(frame + joint / 100) for joint in range(99)]
            if frame == nonfinite_frame:
                values[0] = float("nan")
            writer.writerow([*values, int(onset is not None and frame >= onset)])


def test_parse_upfall_identity_keeps_subject_camera_activity_and_trial():
    identity = parse_upfall_identity(Path("C2S4A5T3.csv"))

    assert identity == {"subject_id": "upfall-s4", "camera_id": "camera-2", "activity_id": 5, "trial_id": 3}


def test_builder_writes_guarded_positive_and_early_safe_windows(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _write_upfall_csv(source / "C1S2A1T1.csv")
    output = tmp_path / "output"

    summary = build_upfall_prefall_dataset(source, output, pre_frames=4, guard_frames=2)

    records = [json.loads(line) for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    positive = next(record for record in records if record["label"] == 1)
    negative = next(record for record in records if record["label"] == 0)
    pose = np.load(output / positive["window_path"])["pose"]
    assert summary["positive_windows"] == 1
    assert summary["negative_windows"] == 1
    assert summary["window_unit"] == "frames"
    assert positive["subject_id"] == "upfall-s2"
    assert positive["onset_frame"] == 10
    assert (positive["window_start_frame"], positive["window_end_frame"]) == (4, 8)
    assert (negative["window_start_frame"], negative["window_end_frame"]) == (0, 4)
    assert pose.shape == (4, 33, 3)
    assert pose[:, 0, 0].tolist() == pytest.approx([4.0, 5.0, 6.0, 7.0])


def test_builder_audits_files_without_impact_or_with_invalid_coordinates(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _write_upfall_csv(source / "C1S1A1T1.csv", onset=None)
    _write_upfall_csv(source / "C2S1A2T1.csv", nonfinite_frame=3)

    summary = build_upfall_prefall_dataset(source, tmp_path / "output", pre_frames=4, guard_frames=2)

    assert summary["input_files"] == 2
    assert summary["excluded_no_impact"] == 1
    assert summary["excluded_invalid_coordinates"] == 1
    assert summary["positive_windows"] == 0


def test_cli_records_frame_units_and_exclusion_counts(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _write_upfall_csv(source / "C1S3A1T1.csv")
    output = tmp_path / "output"
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_upfall_prefall_dataset.py"

    subprocess.run(
        [sys.executable, str(script), "--source-dir", str(source), "--output-dir", str(output),
         "--pre-frames", "4", "--guard-frames", "2"],
        check=True, capture_output=True, text=True,
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["window_unit"] == "frames"
    assert summary["positive_windows"] == 1
    assert summary["excluded_no_impact"] == 0
