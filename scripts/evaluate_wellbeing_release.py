"""Strict, auditable PACE-WB release and promotion gate evaluation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping


_THRESHOLD_FIELDS = {
    "min_auprc": (0.0, 1.0),
    "max_brier": (0.0, 1.0),
    "max_ece": (0.0, 1.0),
    "max_false_invites_per_person_month": (0.0, 100.0),
}


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite(value: object, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")} or not lower <= number <= upper:
        raise ValueError(f"{name} must be in [{lower:g}, {upper:g}]")
    return number


def load_release_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        config = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, ValueError) and "duplicate JSON key" in str(error):
            raise
        raise ValueError("invalid PACE-WB release config") from error
    if not isinstance(config, dict) or config.get("schema_version") != "pace-wb.release.v1":
        raise ValueError("unsupported PACE-WB release schema")
    for field in ("release_id", "model_mode", "thresholds", "readiness", "prompt_budget", "delivery", "provenance"):
        if field not in config:
            raise ValueError(f"missing release config field: {field}")
    if not isinstance(config["release_id"], str) or not config["release_id"].strip():
        raise ValueError("release_id must be non-empty")
    if config.get("research_only") is not True or config.get("promoted") is not False:
        raise ValueError("release config must begin research-only and unpromoted")
    thresholds = config["thresholds"]
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be an object")
    for name, (lower, upper) in _THRESHOLD_FIELDS.items():
        if name not in thresholds:
            raise ValueError(f"missing threshold: {name}")
        _finite(thresholds[name], name, lower, upper)
    readiness = config["readiness"]
    if not isinstance(readiness, dict) or int(readiness.get("min_effective_days", 0)) < 14 or int(readiness.get("provisional_days", 0)) < 1:
        raise ValueError("invalid readiness settings")
    prompt = config["prompt_budget"]
    if not isinstance(prompt, dict) or prompt.get("short_checkin_cooldown_days") != 7 or prompt.get("gds15_cooldown_days") != 28 or prompt.get("max_short_questions") not in {2, 3}:
        raise ValueError("invalid prompt budget")
    delivery = config["delivery"]
    if not isinstance(delivery, dict) or delivery.get("wellbeing_external") is not False or delivery.get("human_review_external") is not False:
        raise ValueError("wellbeing delivery must remain external-forbidden")
    return config


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_atomic_directory(output_dir: Path, files: Mapping[str, bytes]) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    backup: Path | None = None
    try:
        for name, data in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if output_dir.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.backup.", dir=output_dir.parent))
            backup.rmdir()
            os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except Exception:
            if backup is not None and not output_dir.exists():
                os.replace(backup, output_dir)
            raise
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
        staging = None  # type: ignore[assignment]
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and backup.exists() and output_dir.exists():
            shutil.rmtree(backup, ignore_errors=True)


def evaluate_release(config_path: str | Path, evidence: Mapping[str, object], output_dir: str | Path) -> dict[str, Any]:
    config = load_release_config(config_path)
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must be a mapping")
    reasons: list[str] = []
    validation = evidence.get("elderly_external_validation", {})
    if not isinstance(validation, Mapping) or validation.get("status") != "passed" or int(validation.get("subject_count", 0)) < int(config["readiness"]["min_external_subjects"]):
        reasons.append("elderly_external_validation_missing")
    model_mode = evidence.get("model_mode", config["model_mode"])
    if config["model_mode"] == "research_shadow":
        reasons.append("config_research_only")
    if model_mode != "promoted_local_invitation":
        reasons.append("model_mode_not_promotable")
    metrics = evidence.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be an object")
    required_metrics = ("auprc", "brier", "ece", "false_invites_per_person_month")
    for name in required_metrics:
        if name not in metrics:
            reasons.append(f"missing_metric:{name}")
    if not reasons:
        thresholds = config["thresholds"]
        if _finite(metrics["auprc"], "auprc", 0.0, 1.0) < float(thresholds["min_auprc"]):
            reasons.append("auprc_below_threshold")
        if _finite(metrics["brier"], "brier", 0.0, 1.0) > float(thresholds["max_brier"]):
            reasons.append("brier_above_threshold")
        if _finite(metrics["ece"], "ece", 0.0, 1.0) > float(thresholds["max_ece"]):
            reasons.append("ece_above_threshold")
        if _finite(metrics["false_invites_per_person_month"], "false_invites_per_person_month", 0.0, 100.0) > float(thresholds["max_false_invites_per_person_month"]):
            reasons.append("false_invite_budget_exceeded")
    promoted = not reasons and config["delivery"]["wellbeing_external"] is False
    result: dict[str, Any] = {
        "schema_version": "pace-wb.release-result.v1",
        "release_id": config["release_id"],
        "model_mode": model_mode,
        "promoted": bool(promoted),
        "research_only": not bool(promoted),
        "reasons": sorted(set(reasons)),
        "delivery": dict(config["delivery"]),
        "metrics": dict(metrics),
        "elderly_external_validation": dict(validation) if isinstance(validation, Mapping) else {},
    }
    release_bytes = _canonical_bytes({key: value for key, value in config.items()})
    result_bytes = _canonical_bytes(result)
    manifest = {
        "schema_version": "pace-wb.release-manifest.v1",
        "release_id": config["release_id"],
        "promoted": bool(promoted),
        "files": {
            "release_config.json": sha256(release_bytes).hexdigest(),
            "promotion_result.json": sha256(result_bytes).hexdigest(),
        },
    }
    manifest_bytes = _canonical_bytes(manifest)
    _write_atomic_directory(Path(output_dir), {
        "release_config.json": release_bytes,
        "promotion_result.json": result_bytes,
        "manifest.json": manifest_bytes,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    print(json.dumps(evaluate_release(args.config, evidence, args.output_dir), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
