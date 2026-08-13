import json
from pathlib import Path

import pytest

from scripts.evaluate_wellbeing_shadow import evaluate_shadow_model
from scripts.train_wellbeing_shadow import train_shadow_model


def write_records(path: Path) -> None:
    rows = [
        {"subject_id": "s1", "text": "今天心情很好", "label": 0, "split": "train", "audio_present": True},
        {"subject_id": "s2", "text": "最近很开心", "label": 0, "split": "train", "audio_present": False},
        {"subject_id": "s3", "text": "最近很难受", "label": 1, "split": "train", "audio_present": True},
        {"subject_id": "s4", "text": "感觉很孤独", "label": 1, "split": "train", "audio_present": False},
        {"subject_id": "s5", "text": "今天心情很好", "label": 0, "split": "test", "audio_present": True},
        {"subject_id": "s6", "text": "最近很难受", "label": 1, "split": "test", "audio_present": False},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_train_and_evaluate_write_research_only_artifacts(tmp_path: Path) -> None:
    data = tmp_path / "checkins.jsonl"
    write_records(data)
    artifact = train_shadow_model(data, tmp_path / "model")
    assert artifact.exists()
    meta = json.loads((artifact.parent / "manifest.json").read_text(encoding="utf-8"))
    assert meta["research_only"] is True
    assert meta["promoted"] is False
    metrics = evaluate_shadow_model(artifact, data, tmp_path / "eval.json")
    assert metrics["research_only"] is True
    assert metrics["promoted"] is False
    assert 0.0 <= metrics["metrics"]["brier"] <= 1.0
    assert (tmp_path / "eval.json").exists()


def test_train_rejects_demo_records_and_subject_leakage(tmp_path: Path) -> None:
    data = tmp_path / "bad.jsonl"
    rows = [
        {"subject_id": "s1", "text": "a", "label": 0, "split": "train"},
        {"subject_id": "s1", "text": "b", "label": 1, "split": "test"},
    ]
    data.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="subject"):
        train_shadow_model(data, tmp_path / "model")
    data.write_text(json.dumps({"subject_id": "s2", "text": "demo", "label": 0, "demo": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="demo"):
        train_shadow_model(data, tmp_path / "model2")
