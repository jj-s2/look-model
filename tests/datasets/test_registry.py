from __future__ import annotations

import json
from pathlib import Path

import pytest

from datasets.unified.registry import load_registry


ROOT = Path(__file__).resolve().parents[2]


def test_required_dataset_sources_and_training_gate() -> None:
    registry = load_registry(ROOT / "datasets" / "manifest.json")

    assert registry["GMDCSA24"].expected_size_bytes == 1107545615
    assert registry["CAUCAFall"].official_source.endswith("/7w7fccy7ky/4")
    assert registry["Pre-VFall"].expected_size_bytes == 21096978752
    assert registry["UP-Fall-3D-Skeletons"].license_status == "needs_confirmation"
    assert registry["UP-Fall-3D-Skeletons"].can_train is False


def test_completed_dataset_requires_sha256(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "datasets": [
                    {
                        "name": "x",
                        "version": "1",
                        "official_source": "https://example.test",
                        "license": "CC BY 4.0",
                        "license_status": "confirmed",
                        "download_status": "completed",
                        "expected_size_bytes": 1,
                        "sha256": None,
                        "subjects": 1,
                        "split_strategy": "subject_grouped",
                        "redistribution": "raw_data_not_committed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="completed dataset requires sha256"):
        load_registry(manifest)


@pytest.mark.parametrize(
    "patch",
    [
        {"name": ""},
        {"version": ""},
        {"official_source": "http://example.test"},
        {"expected_size_bytes": 0},
    ],
)
def test_registry_rejects_invalid_entry_fields(tmp_path: Path, patch: dict[str, object]) -> None:
    entry = {
        "name": "x",
        "version": "1",
        "official_source": "https://example.test",
        "license": "CC BY 4.0",
        "license_status": "confirmed",
        "download_status": "not_downloaded",
        "expected_size_bytes": 1,
        "sha256": None,
        "subjects": 1,
        "split_strategy": "subject_grouped",
        "redistribution": "raw_data_not_committed",
    }
    entry.update(patch)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": "2.0", "datasets": [entry]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_registry(manifest)
