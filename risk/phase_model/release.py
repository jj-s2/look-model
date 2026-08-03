"""Create a conservative, self-describing phase-model release directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


_JSON_NAMES = ("dataset_lock.json", "split_manifest.json", "metrics.json", "benchmark.json", "calibration.json", "ablation.json", "confusion_matrices.json")


def create_release_artifacts(*, release_id: str, metrics: Mapping[str, object], benchmark: Mapping[str, object], output_dir: str | Path, inputs: Mapping[str, object] | None = None) -> dict[str, str]:
    if not release_id.strip():
        raise ValueError("release_id must be non-empty")
    for name, data in (("metrics", metrics), ("benchmark", benchmark)):
        if data.get("release_id") != release_id:
            raise ValueError(f"{name} release_id does not match")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, Mapping[str, object]] = {
        "dataset_lock.json": {"release_id": release_id, **dict((inputs or {}).get("dataset_lock", {}))},
        "split_manifest.json": {"release_id": release_id, **dict((inputs or {}).get("split_manifest", {}))},
        "metrics.json": dict(metrics), "benchmark.json": dict(benchmark),
        "calibration.json": {"release_id": release_id, "status": "unavailable", "reason": "not calibrated"},
        "ablation.json": {"release_id": release_id, "status": "unavailable", "reason": "not run"},
        "confusion_matrices.json": {"release_id": release_id, "status": "unavailable", "reason": "not run"},
    }
    written: dict[str, str] = {}
    for name, payload in payloads.items():
        path = destination / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written[name[:-5]] = str(path)
    config_path = destination / "config.py"
    config_path.write_text(f"RELEASE_ID = {release_id!r}\n", encoding="utf-8")
    written["config"] = str(config_path)
    model_card = destination / "model_card.md"
    model_card.write_text(
        f"# PA-DTSF release {release_id}\n\n"
        "- simulated_young_subjects: true\n- clinical_validation: false\n"
        f"- promoted: {bool(metrics.get('promoted', False))}\n",
        encoding="utf-8",
    )
    written["model_card"] = str(model_card)
    checksum_lines = []
    for path in sorted(destination.iterdir()):
        if path.name == "checksums.sha256" or not path.is_file():
            continue
        checksum_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    checksum_path = destination / "checksums.sha256"
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    written["checksums"] = str(checksum_path)
    return written
