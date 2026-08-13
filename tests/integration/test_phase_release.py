import json

import pytest

from risk.phase_model.release import create_release_artifacts


def test_release_requires_matching_ids(tmp_path):
    with pytest.raises(ValueError, match="release_id"):
        create_release_artifacts(
            release_id="r1", metrics={"release_id": "r2"}, benchmark={"release_id": "r1"}, output_dir=tmp_path,
        )


def test_release_writes_auditable_artifacts(tmp_path):
    paths = create_release_artifacts(
        release_id="r1", metrics={"release_id": "r1", "promoted": False}, benchmark={"release_id": "r1"}, output_dir=tmp_path,
    )
    assert (tmp_path / "model_card.md").exists()
    assert (tmp_path / "checksums.sha256").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["release_id"] == "r1"
    assert "model_card" in paths
