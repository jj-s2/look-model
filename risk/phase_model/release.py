"""Create a conservative, self-describing phase-model release directory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_JSON_NAMES = (
    "dataset_lock.json",
    "split_manifest.json",
    "metrics.json",
    "benchmark.json",
    "calibration.json",
    "ablation.json",
    "confusion_matrices.json",
)


def create_release_artifacts(
    *,
    release_id: str,
    metrics: Mapping[str, object],
    benchmark: Mapping[str, object],
    output_dir: str | Path,
    inputs: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Create the canonical JSON artifacts for a phase-model release."""

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
        "metrics.json": dict(metrics),
        "benchmark.json": dict(benchmark),
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


@dataclass(frozen=True)
class ReleaseBundle:
    """In-memory representation of a phase-model release."""

    release_id: str
    checkpoint: bytes
    dataset_lock: Mapping[str, object]
    split_manifest: Mapping[str, object]
    metrics: Mapping[str, object]
    benchmark: Mapping[str, object] | None = None
    calibration: Mapping[str, object] | None = None
    ablation: Mapping[str, object] | None = None
    confusion_matrices: Mapping[str, object] | None = None

    @property
    def checkpoint_sha256(self) -> str:
        return hashlib.sha256(self.checkpoint).hexdigest()

    def validate(self) -> "ReleaseBundle":
        if not self.release_id:
            raise ValueError("release_id must not be empty")
        if not self.checkpoint:
            raise ValueError("checkpoint bytes must not be empty")
        if "clips" not in self.dataset_lock:
            raise ValueError("dataset_lock must contain clips")
        if "partitions" not in self.split_manifest:
            raise ValueError("split_manifest must contain partitions")
        return self


def build_release_bundle(
    release_id: str,
    checkpoint_path: Path,
    release_dir: Path,
    inputs: Mapping[str, Mapping[str, object]] | None = None,
) -> ReleaseBundle:
    """Assemble a release bundle from files already written to ``release_dir``."""

    inputs = inputs or {}
    checkpoint_bytes = checkpoint_path.read_bytes()

    def load_json(name: str) -> Mapping[str, object]:
        data = inputs.get(name)
        if data is None:
            data = inputs.get(name.removesuffix(".json"))
        if data is not None:
            return dict(data)
        path = release_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    return ReleaseBundle(
        release_id=release_id,
        checkpoint=checkpoint_bytes,
        dataset_lock={"release_id": release_id, **dict(load_json("dataset_lock.json"))},
        split_manifest={"release_id": release_id, **dict(load_json("split_manifest.json"))},
        metrics={"release_id": release_id, **dict(load_json("metrics.json"))},
        benchmark={"release_id": release_id, **dict(load_json("benchmark.json"))}
        if (release_dir / "benchmark.json").exists() or "benchmark" in inputs or "benchmark.json" in inputs
        else None,
        calibration={"release_id": release_id, **dict(load_json("calibration.json"))}
        if (release_dir / "calibration.json").exists() or "calibration" in inputs or "calibration.json" in inputs
        else None,
        ablation={"release_id": release_id, **dict(load_json("ablation.json"))}
        if (release_dir / "ablation.json").exists() or "ablation" in inputs or "ablation.json" in inputs
        else None,
        confusion_matrices={"release_id": release_id, **dict(load_json("confusion_matrices.json"))}
        if (release_dir / "confusion_matrices.json").exists() or "confusion_matrices" in inputs or "confusion_matrices.json" in inputs
        else None,
    ).validate()


def load_release_bundle(release_dir: Path) -> ReleaseBundle:
    """Load a release bundle from disk."""

    checkpoint_path = release_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"release checkpoint not found: {checkpoint_path}")
    metrics_path = release_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"release metrics not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    release_id = str(metrics.get("release_id", release_dir.name))
    return build_release_bundle(release_id, checkpoint_path, release_dir)


def write_release_bundle(bundle: ReleaseBundle, release_dir: Path) -> None:
    """Persist a release bundle to disk."""

    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "checkpoint.pt").write_bytes(bundle.checkpoint)
    (release_dir / "dataset_lock.json").write_text(
        json.dumps(dict(bundle.dataset_lock), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (release_dir / "split_manifest.json").write_text(
        json.dumps(dict(bundle.split_manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (release_dir / "metrics.json").write_text(
        json.dumps(dict(bundle.metrics), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if bundle.benchmark is not None:
        (release_dir / "benchmark.json").write_text(
            json.dumps(dict(bundle.benchmark), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if bundle.calibration is not None:
        (release_dir / "calibration.json").write_text(
            json.dumps(dict(bundle.calibration), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if bundle.ablation is not None:
        (release_dir / "ablation.json").write_text(
            json.dumps(dict(bundle.ablation), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if bundle.confusion_matrices is not None:
        (release_dir / "confusion_matrices.json").write_text(
            json.dumps(dict(bundle.confusion_matrices), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def release_inference_config(bundle: ReleaseBundle) -> dict[str, Any]:
    """Return the calibration settings needed by an inference predictor."""

    calibration = bundle.calibration or {}
    return {
        "temperature": float(calibration.get("temperature", 1.0)),
        "threshold": float(calibration.get("threshold", 0.5)),
        "release_id": bundle.release_id,
        "checkpoint_sha256": bundle.checkpoint_sha256,
    }
