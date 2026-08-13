"""Tests for leakage-safe cross-domain UR Fall fine-tuning."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from risk.urfall_crossdomain_training import train_urfall_crossdomain_experiment
from risk.urfall_multiseed import run_urfall_multiseed_experiment
from tests.risk.test_urfall_temporal_training import _dataset


def _adl(root: Path) -> Path:
    rows = []
    (root / "windows").mkdir(parents=True)
    for index in range(5):
        relative = Path("windows") / f"adl-{index}.npz"
        np.savez_compressed(root / relative, pose=np.full((3, 33, 3), -index, dtype=np.float32), label=np.int8(0))
        rows.append({"sample_id": f"adl-{index}", "sequence_id": f"urfall-adl-{index}", "label": 0, "pose_path": relative.as_posix()})
    manifest = root / "pose_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest


def test_crossdomain_folds_keep_fall_and_adl_sequences_out_of_training(tmp_path: Path) -> None:
    fall = _dataset(tmp_path / "fall")
    adl = _adl(tmp_path / "adl")
    report = train_urfall_crossdomain_experiment(fall, fall.parent, adl, adl.parent, tmp_path / "output", epochs=1, device="cpu", adl_folds=5)
    assert report["external_experiment"] is True
    assert report["promoted"] is False
    assert report["adl_holdout_sequence_count"] == 5
    assert len(report["folds"]) == 4
    assert all(fold["held_out_fall_sequence"] not in fold["train_fall_sequences"] for fold in report["folds"])
    assert all(fold["adl_holdout_sequences"] for fold in report["folds"])
    assert (tmp_path / "output" / "fall_predictions.jsonl").is_file()
    assert (tmp_path / "output" / "adl_predictions.jsonl").is_file()


def test_crossdomain_report_records_the_exact_random_seed(tmp_path: Path) -> None:
    fall = _dataset(tmp_path / "fall")
    adl = _adl(tmp_path / "adl")
    report = train_urfall_crossdomain_experiment(
        fall,
        fall.parent,
        adl,
        adl.parent,
        tmp_path / "output",
        epochs=1,
        device="cpu",
        random_seed=17,
        adl_folds=5,
    )
    assert report["random_seed"] == 17


def test_multiseed_runner_writes_stability_statistics(tmp_path: Path) -> None:
    fall = _dataset(tmp_path / "fall")
    adl = _adl(tmp_path / "adl")
    report = run_urfall_multiseed_experiment(
        fall_manifest=fall,
        fall_root=fall.parent,
        adl_manifest=adl,
        adl_root=adl.parent,
        output_dir=tmp_path / "multiseed",
        seeds=(3, 5),
        epochs=1,
        device="cpu",
        recall_floor=0.5,
    )

    assert report["seeds"] == [3, 5]
    assert report["seed_count"] == 2
    assert report["promoted"] is False
    assert set(report["input_sha256"]) == {"fall_manifest", "adl_manifest"}
    assert all(len(value) == 64 for value in report["input_sha256"].values())
    assert report["metrics"]["fall_f1"]["count"] == 2
    assert report["metrics"]["fall_f1"]["minimum"] <= report["metrics"]["fall_f1"]["maximum"]
    assert (tmp_path / "multiseed" / "seed-3" / "threshold_selection.json").is_file()
    assert (tmp_path / "multiseed" / "seed-5" / "threshold_selection.json").is_file()
    assert json.loads((tmp_path / "multiseed" / "aggregate.json").read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("seeds", [(), (1, 1), (-1, 2), (True, 2)])
def test_multiseed_runner_rejects_invalid_seed_declarations(tmp_path: Path, seeds: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="seeds"):
        run_urfall_multiseed_experiment(
            fall_manifest=tmp_path / "fall.jsonl",
            fall_root=tmp_path,
            adl_manifest=tmp_path / "adl.jsonl",
            adl_root=tmp_path,
            output_dir=tmp_path / "output",
            seeds=seeds,  # type: ignore[arg-type]
            epochs=1,
            device="cpu",
        )
