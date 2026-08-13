"""Build a subject-safe nested-LOSO RG-PCNet evaluation artifact.

The inner out-of-fold predictions are the only inputs used for calibration and
threshold selection.  Outer predictions are scored once with that frozen
decision and are never used to tune a threshold.  The release is intentionally
left non-promoted until the continuous-event evidence from Plan 3 exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from risk.phase_model.nested_selection import OOFRecord, SelectiveThreshold, select_oof_thresholds
from risk.phase_model.release_config import RGPCReleaseConfig, write_release_config


_CONTINUOUS_EVENT_REASON = "continuous event evaluation is required"
_DEFAULT_RECALL_FLOOR = 0.8
_DEFAULT_FPR_CEILING = 0.2
_DEFAULT_COVERAGE_FLOOR = 0.5
_DEFAULT_CONFIRM_SECONDS = 0.8
_DEFAULT_RECOVERY_SECONDS = 2.0
_DEFAULT_COOLDOWN_SECONDS = 10.0
_INFEASIBLE_TEMPERATURE = 5.0
_INFEASIBLE_FALL_THRESHOLD = 1.0
_INFEASIBLE_RELIABILITY_THRESHOLD = 1.0
_SHA256_HEX = frozenset("0123456789abcdef")


@dataclass
class RGPCPromotionArtifacts:
    """Inputs to the final, cross-artifact RG-PCNet promotion gate.

    The nested summary and continuous gate are intentionally kept as detached
    JSON-like mappings.  This lets the finalizer consume artifacts produced by
    separate jobs without re-running either evaluation, while the optional
    hashes provide an explicit link back to the model and release config.
    """

    nested_summary: Mapping[str, Any]
    continuous_gate: Mapping[str, Any]
    release_config: Mapping[str, Any]
    baseline_summary: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_sha256: str | None = None
    config_sha256: str | None = None
    output_dir: os.PathLike[str] | str | None = None
    source_dir: os.PathLike[str] | str | None = None
    input_hashes: Mapping[str, str] | None = None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_line(line: str, path: Path, line_number: int) -> Mapping[str, Any]:
    if not line.strip():
        raise ValueError(f"empty prediction record at {path}:{line_number}")
    try:
        value = json.loads(
            line,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid prediction JSON at {path}:{line_number}") from exc
    if type(value) is not dict:
        raise ValueError(f"prediction record must be a JSON object at {path}:{line_number}")
    return value


def _record_from_mapping(payload: Mapping[str, Any], *, path: Path, line_number: int) -> OOFRecord:
    try:
        subject = payload["subject_id"]
        label = payload["label"]
        logit = payload.get("fall_logit", payload.get("logit"))
        reliability = payload["reliability"]
        if logit is None:
            raise KeyError("fall_logit")
        record = OOFRecord(subject, label, logit, reliability)
        # Subject identity is a split boundary; normalize surrounding
        # whitespace before leakage checks, sorting, and artifact emission.
        return OOFRecord(record.subject_id.strip(), record.label, record.fall_logit, record.reliability)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid prediction record at {path}:{line_number}") from exc


def _read_jsonl(path: os.PathLike[str] | str) -> tuple[tuple[OOFRecord, Mapping[str, Any]], ...]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"unable to read prediction JSONL: {source}") from exc
    if not lines:
        raise ValueError(f"prediction JSONL must contain at least one record: {source}")
    records: list[tuple[OOFRecord, Mapping[str, Any]]] = []
    for line_number, line in enumerate(lines, 1):
        payload = _parse_json_line(line, source, line_number)
        records.append((_record_from_mapping(payload, path=source, line_number=line_number), payload))
    return tuple(records)


def _sha256_path(path: os.PathLike[str] | str) -> str:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"hash input is not a regular file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"unable to hash input: {source}") from exc
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sigmoid(value: float) -> float:
    """Numerically stable sigmoid for extreme, but finite, logits."""

    if value >= 0.0:
        exponential = math.exp(-value) if value < 745.0 else 0.0
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value) if value > -745.0 else 0.0
    return exponential / (1.0 + exponential)


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(path, _canonical_bytes(value))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    _atomic_write_bytes(path, payload)


def _selection_dict(selection: SelectiveThreshold) -> dict[str, Any]:
    return asdict(selection)


def _binary_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float | int | bool]:
    if len(labels) != len(predictions):
        raise ValueError("outer labels and predictions must have the same length")
    selected_count = len(labels)
    true_positive = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    false_positive = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    false_negative = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    selected_positive = sum(label == 1 for label in labels)
    selected_negative = sum(label == 0 for label in labels)
    recall = true_positive / selected_positive if selected_positive else 0.0
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator if denominator else 0.0
    fpr = false_positive / selected_negative if selected_negative else 1.0
    return {
        "selected_count": selected_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


def _hash_or_derived(path: os.PathLike[str] | str | None, fallback: bytes) -> tuple[str, str, bool]:
    if path is None:
        return _sha256_bytes(fallback), "derived_from_prediction_manifest", False
    return _sha256_path(path), str(Path(path)), True


def _outer_subject_value(subjects: Sequence[str]) -> str | list[str]:
    unique = sorted(set(subjects))
    if len(unique) == 1:
        return unique[0]
    return unique


def _canonical_record_payload(record: OOFRecord) -> dict[str, Any]:
    """Return the normalized, alias-independent prediction record."""

    return {
        "subject_id": record.subject_id,
        "label": record.label,
        "fall_logit": record.fall_logit,
        "reliability": record.reliability,
    }


def _record_sort_key(pair: tuple[OOFRecord, Mapping[str, Any]]) -> tuple[Any, ...]:
    record, _ = pair
    return (
        record.subject_id,
        record.label,
        record.fall_logit,
        record.reliability,
    )


def _publish_artifacts(
    destination: Path,
    writer: Any,
) -> None:
    """Publish a complete artifact directory without deleting old targets on failure."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    backup: Path | None = None
    destination_moved = False
    try:
        assert staging is not None
        writer(staging)
        if destination.exists():
            if not destination.is_dir():
                raise ValueError(f"output path must be a directory: {destination}")
            backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent))
            backup.rmdir()
            os.replace(destination, backup)
            destination_moved = True
        os.replace(staging, destination)
        staging = None
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    except Exception:
        if destination_moved and backup is not None and backup.exists():
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(backup, destination)
            backup = None
        raise
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists():
            if destination_moved:
                # A failed restore must never remove the original backup.
                # Leave it in place for manual recovery rather than deleting it.
                pass
            else:
                shutil.rmtree(backup)


def evaluate_rg_pcnet_loso(
    inner_oof_path: os.PathLike[str] | str,
    outer_predictions_path: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    *,
    checkpoint_path: os.PathLike[str] | str | None = None,
    dataset_manifest_path: os.PathLike[str] | str | None = None,
    split_manifest_path: os.PathLike[str] | str | None = None,
    release_id: str = "rgpc-loso-v1",
    recall_floor: float = _DEFAULT_RECALL_FLOOR,
    fpr_ceiling: float = _DEFAULT_FPR_CEILING,
    coverage_floor: float = _DEFAULT_COVERAGE_FLOOR,
    confirm_seconds: float = _DEFAULT_CONFIRM_SECONDS,
    recovery_seconds: float = _DEFAULT_RECOVERY_SECONDS,
    cooldown_seconds: float = _DEFAULT_COOLDOWN_SECONDS,
    outer_subject: str | None = None,
) -> dict[str, Any]:
    """Evaluate one outer fold using thresholds fit only on inner OOF records."""

    inner_pairs = tuple(sorted(_read_jsonl(inner_oof_path), key=_record_sort_key))
    outer_pairs = tuple(sorted(_read_jsonl(outer_predictions_path), key=_record_sort_key))
    inner_records = tuple(record for record, _ in inner_pairs)
    outer_records = tuple(record for record, _ in outer_pairs)
    calibration_subjects = sorted({record.subject_id for record in inner_records})
    outer_subjects = sorted({record.subject_id for record in outer_records})
    leaked = set(calibration_subjects).intersection(outer_subjects)
    if leaked:
        raise ValueError("outer subject leaked into calibration")
    if outer_subject is not None and outer_subject not in outer_subjects:
        raise ValueError("outer_subject does not match outer prediction records")

    inner_manifest = [_canonical_record_payload(record) for record, _ in inner_pairs]
    inner_bytes = _canonical_bytes(inner_manifest)
    split_fallback = _canonical_bytes({"calibration_subjects": calibration_subjects, "outer_subjects": outer_subjects})
    selection = select_oof_thresholds(
        inner_records,
        recall_floor=recall_floor,
        fpr_ceiling=fpr_ceiling,
        coverage_floor=coverage_floor,
    )

    model_hash, model_source, model_hash_available = _hash_or_derived(checkpoint_path, inner_bytes)
    dataset_hash, dataset_source, dataset_hash_available = _hash_or_derived(dataset_manifest_path, inner_bytes)
    split_hash, split_source, split_hash_available = _hash_or_derived(split_manifest_path, split_fallback)
    hash_inputs_valid = model_hash_available and dataset_hash_available and split_hash_available

    if selection.feasible:
        assert selection.fall_threshold is not None
        assert selection.reliability_threshold is not None
        release_config = RGPCReleaseConfig(
            schema_version="rgpc.release.v1",
            release_id=release_id,
            model_sha256=model_hash,
            dataset_sha256=dataset_hash,
            split_sha256=split_hash,
            temperature=selection.temperature,
            fall_threshold=selection.fall_threshold,
            reliability_threshold=selection.reliability_threshold,
            confirm_seconds=confirm_seconds,
            recovery_seconds=recovery_seconds,
            cooldown_seconds=cooldown_seconds,
            minimum_coverage=coverage_floor,
        )
        release_config_payload: dict[str, Any] = release_config.to_dict()
    else:
        # A non-feasible selection is still auditable, but cannot be loaded as a
        # runnable release because the contract requires concrete thresholds.
        release_config = None
        release_config_payload = {
            "schema_version": "rgpc.release.v1",
            "release_id": release_id.strip(),
            "model_sha256": model_hash,
            "dataset_sha256": dataset_hash,
            "split_sha256": split_hash,
            # A disabled, non-promotable sentinel that still satisfies the
            # strict release schema.  promotion_checks.selection_feasible and
            # promoted remain false, so this can never be mistaken for a
            # runnable release.
            "temperature": _INFEASIBLE_TEMPERATURE,
            "fall_threshold": _INFEASIBLE_FALL_THRESHOLD,
            "reliability_threshold": _INFEASIBLE_RELIABILITY_THRESHOLD,
            "confirm_seconds": confirm_seconds,
            "recovery_seconds": recovery_seconds,
            "cooldown_seconds": cooldown_seconds,
            "minimum_coverage": coverage_floor,
        }

    outer_rows: list[dict[str, Any]] = []
    calibrated_probabilities: list[float] = []
    decisions: list[int | None] = []
    if selection.feasible:
        assert selection.fall_threshold is not None
        assert selection.reliability_threshold is not None
        for record, payload in outer_pairs:
            probability = _sigmoid(record.fall_logit / selection.temperature)
            selected = record.reliability >= selection.reliability_threshold
            decision = int(probability >= selection.fall_threshold) if selected else None
            row = _canonical_record_payload(record)
            row.update({
                "calibrated_fall_probability": probability,
                "selected_by_reliability_gate": selected,
                "fall_decision": decision,
            })
            outer_rows.append(row)
            calibrated_probabilities.append(probability)
            decisions.append(decision)
    else:
        for record, _ in outer_pairs:
            row = _canonical_record_payload(record)
            row.update({"calibrated_fall_probability": None, "selected_by_reliability_gate": False, "fall_decision": None})
            outer_rows.append(row)
            calibrated_probabilities.append(float("nan"))
            decisions.append(None)

    selected_labels = [record.label for record, decision in zip(outer_records, decisions) if decision is not None]
    selected_predictions = [decision for decision in decisions if decision is not None]
    outer_metrics = _binary_metrics(selected_labels, selected_predictions)
    outer_metrics.update({
        "record_count": len(outer_records),
        "coverage": sum(decision is not None for decision in decisions) / len(outer_records),
        "has_both_classes": {record.label for record in outer_records} == {0, 1},
        "outer_class_counts": {
            "0": sum(record.label == 0 for record in outer_records),
            "1": sum(record.label == 1 for record in outer_records),
        },
    })

    calibration_payload = {
        "temperature": selection.temperature,
        "enabled": selection.calibration_enabled,
        "reason": selection.calibration_reason,
        "sample_count": len(inner_records),
        "class_counts": {"0": sum(record.label == 0 for record in inner_records), "1": sum(record.label == 1 for record in inner_records)},
        "subjects": calibration_subjects,
        "split_hash": selection.split_hash,
        "source": str(Path(inner_oof_path)),
    }
    selection_payload = _selection_dict(selection)
    promotion_checks = {
        "selection_feasible": selection.feasible,
        "outer_has_both_classes": bool(outer_metrics["has_both_classes"]),
        "outer_coverage_meets_floor": float(outer_metrics["coverage"]) >= coverage_floor,
        "hashes_available": hash_inputs_valid,
        "continuous_event_evaluation": False,
        "promoted": False,
    }
    result: dict[str, Any] = {
        "schema_version": "rgpc.loso.evaluation.v1",
        "outer_subject": _outer_subject_value(outer_subjects),
        "calibration_subjects": calibration_subjects,
        "outer_subjects": outer_subjects,
        "selection": selection_payload,
        "calibration": calibration_payload,
        "outer_metrics": outer_metrics,
        "aggregate_metrics": outer_metrics,
        "release_config": release_config_payload,
        "promotion_checks": promotion_checks,
        "promoted": False,
        "reason": _CONTINUOUS_EVENT_REASON,
        "provenance": {
            "model_sha256": {"value": model_hash, "source": model_source},
            "dataset_sha256": {"value": dataset_hash, "source": dataset_source},
            "split_sha256": {"value": split_hash, "source": split_source},
        },
    }

    destination = Path(output_dir)

    def write_staging(staging: Path) -> None:
        _write_jsonl(staging / "outer_predictions.jsonl", outer_rows)
        _write_json(staging / "calibration.json", calibration_payload)
        _write_json(staging / "selection.json", selection_payload)
        _write_json(staging / "release_config.json", release_config_payload)
        _write_json(staging / "evaluation.json", result)
        if release_config is not None:
            # Keep the on-disk contract exactly identical to the embedded copy.
            write_release_config(release_config, staging / "release_config.json")

    _publish_artifacts(destination, write_staging)
    return result


# --- final cross-artifact promotion -------------------------------------------------


def _object_mapping(value: object, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return converted
    converted = getattr(value, "__dict__", None)
    if isinstance(converted, Mapping):
        return converted
    raise ValueError(f"{name} must be a mapping")


def _candidate_value(candidate: object, names: Sequence[str], default: object = None) -> object:
    if isinstance(candidate, Mapping):
        for name in names:
            if name in candidate:
                return candidate[name]
    else:
        for name in names:
            if hasattr(candidate, name):
                return getattr(candidate, name)
    return default


def _first_path(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]], default: object = None) -> object:
    for path in paths:
        current: object = mapping
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            return current
    return default


def _finite_metric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _bounded_metric(value: object, lower: float, upper: float) -> float | None:
    numeric = _finite_metric(value)
    if numeric is None or numeric < lower or numeric > upper:
        return None
    return numeric


def _valid_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in _SHA256_HEX for character in value)


def _metric(mapping: Mapping[str, Any], names: Sequence[str]) -> float | None:
    return _finite_metric(_first_path(mapping, tuple((name,) for name in names)))


def _unique_seeds(value: object) -> set[str]:
    if isinstance(value, Mapping):
        value = tuple(value.values())
    if not isinstance(value, (list, tuple, set, frozenset)):
        return set()
    seeds: set[str] = set()
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("seed")
        if type(item) is int:
            seeds.add(str(item))
    return seeds


def _promotion_inputs(candidate: object) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    nested_raw = _candidate_value(candidate, ("nested_summary", "nested_loso", "loso_summary"))
    if nested_raw is None and isinstance(candidate, Mapping) and "selection" in candidate:
        nested_raw = candidate
    continuous_raw = _candidate_value(candidate, ("continuous_gate", "promotion_gate"))
    config_raw = _candidate_value(candidate, ("release_config", "config"))
    baseline_raw = _candidate_value(candidate, ("baseline_summary", "baseline", "baseline_metrics"), {})
    if nested_raw is None:
        raise ValueError("nested LOSO summary is required")
    if continuous_raw is None:
        raise ValueError("continuous promotion gate is required")
    if config_raw is None:
        nested_mapping = _object_mapping(nested_raw, "nested summary")
        config_raw = _first_path(nested_mapping, (("release_config",), ("config",)))
    if config_raw is None:
        raise ValueError("release config is required")
    return (_object_mapping(nested_raw, "nested summary"), _object_mapping(continuous_raw, "continuous gate"), _object_mapping(config_raw, "release config"), _object_mapping(baseline_raw, "baseline summary"))


def _normalized_release_config(value: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate and normalize a release config through its public schema."""

    try:
        return RGPCReleaseConfig(**dict(value)).to_dict()
    except (TypeError, ValueError, OverflowError):
        return None


def _release_ids(candidate: object, nested: Mapping[str, Any], gate: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    raw_ids = [
        _candidate_value(candidate, ("release_id",)),
        _first_path(nested, (("release_id",), ("release_config", "release_id"))),
        gate.get("release_id"), config.get("release_id"),
    ]
    ids = {str(value).strip() for value in raw_ids if value is not None and str(value).strip()}
    if len(ids) > 1:
        raise ValueError("release_id mismatch")
    if not ids:
        raise ValueError("release_id is required")
    return ids.pop()


def _hash_checks(candidate: object, gate: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    valid = all(_valid_sha256(config.get(name)) for name in ("model_sha256", "dataset_sha256", "split_sha256"))
    checkpoint_hash = _candidate_value(candidate, ("checkpoint_sha256", "checkpoint_hash", "model_sha256"))
    valid = valid and _valid_sha256(checkpoint_hash) and checkpoint_hash == config.get("model_sha256")
    gate_inputs = gate.get("input_hashes")
    declared_inputs = _candidate_value(candidate, ("input_hashes",), None)
    required_inputs = {"predictions", "truth_events", "release_config"}
    valid = valid and isinstance(gate_inputs, Mapping) and set(gate_inputs) == required_inputs
    valid = valid and all(_valid_sha256(gate_inputs.get(key)) for key in required_inputs) if isinstance(gate_inputs, Mapping) else False
    valid = valid and isinstance(declared_inputs, Mapping) and dict(declared_inputs) == dict(gate_inputs)
    config_hash = _candidate_value(candidate, ("config_sha256", "release_config_sha256"), None)
    valid = valid and _valid_sha256(config_hash)
    if isinstance(gate_inputs, Mapping):
        valid = valid and config_hash == gate_inputs.get("release_config")
    output_hashes = gate.get("output_sha256")
    required_outputs = {"continuous_metrics.json", "transitions.jsonl", "alerts.jsonl", "promotion_gate.json"}
    valid = valid and isinstance(output_hashes, Mapping) and set(output_hashes) == required_outputs
    valid = valid and all(_valid_sha256(output_hashes.get(key)) for key in required_outputs) if isinstance(output_hashes, Mapping) else False
    return bool(valid)


def finalize_rgpc_release(candidate_artifacts: RGPCPromotionArtifacts | Mapping[str, Any] | object, *, output_dir: os.PathLike[str] | str | None = None, calibration_ece_ceiling: float = 0.10) -> dict[str, Any]:
    """Apply all nested, continuous, calibration, coverage, and hash gates."""
    nested, continuous, config, baseline = _promotion_inputs(candidate_artifacts)
    release_id = _release_ids(candidate_artifacts, nested, continuous, config)
    aggregate_raw = _first_path(nested, (("aggregate_metrics",), ("outer_metrics",), ("metrics",)))
    aggregate = _object_mapping(aggregate_raw, "aggregate metrics") if aggregate_raw is not None else nested
    selection = _object_mapping(nested.get("selection", {}), "selection")
    calibration = _object_mapping(nested.get("calibration", {}), "calibration")
    nested_checks = nested.get("promotion_checks") if isinstance(nested.get("promotion_checks"), Mapping) else {}
    normalized_config = _normalized_release_config(config)
    nested_config_raw = nested.get("release_config")
    normalized_nested_config = _normalized_release_config(nested_config_raw) if isinstance(nested_config_raw, Mapping) else None
    checks: dict[str, bool] = {
        "selection": nested.get("selection_feasible", selection.get("feasible", nested_checks.get("selection_feasible"))) is True
    }
    checks["release_config"] = (
        normalized_config is not None
        and normalized_nested_config is not None
        and normalized_config == normalized_nested_config
        and normalized_config["release_id"] == release_id
    )

    candidate_f1 = _bounded_metric(_first_path(aggregate, tuple((name,) for name in ("macro_event_f1", "event_macro_f1", "macro_f1", "event_f1", "subject_macro_f1", "f1"))), 0.0, 1.0)
    baseline_f1 = _bounded_metric(_first_path(baseline, tuple((name,) for name in ("macro_event_f1", "event_macro_f1", "macro_f1", "event_f1", "subject_macro_f1", "f1"))), 0.0, 1.0)
    explicit_f1 = _candidate_value(candidate_artifacts, ("macro_event_f1_improved",), None)
    checks["macro_event_f1"] = (explicit_f1 if type(explicit_f1) is bool else False) if explicit_f1 is not None else (candidate_f1 is not None and baseline_f1 is not None and candidate_f1 > baseline_f1)

    constraints = continuous.get("constraints") if isinstance(continuous.get("constraints"), Mapping) else {}
    recall_floor = _bounded_metric(_candidate_value(candidate_artifacts, ("recall_floor",), constraints.get("minimum_event_recall")), 0.0, 1.0)
    if recall_floor is None:
        recall_floor = _bounded_metric(selection.get("recall_floor"), 0.0, 1.0)
    if recall_floor is None and _candidate_value(candidate_artifacts, ("recall_floor",), constraints.get("minimum_event_recall")) is None and selection.get("recall_floor") is None:
        recall_floor = _DEFAULT_RECALL_FLOOR
    candidate_recall = _bounded_metric(_first_path(aggregate, (("event_recall",), ("recall",), ("macro_event_recall",))), 0.0, 1.0)
    checks["recall"] = candidate_recall is not None and recall_floor is not None and candidate_recall >= recall_floor

    false_ceiling = _finite_metric(_candidate_value(candidate_artifacts, ("maximum_false_alerts_per_hour",), constraints.get("maximum_false_alerts_per_hour")))
    false_ceiling = false_ceiling if false_ceiling is not None and false_ceiling >= 0.0 else None
    candidate_false = _bounded_metric(_first_path(aggregate, (("false_alerts_per_hour",), ("false_alarm_rate",))), 0.0, float("inf"))
    baseline_false = _bounded_metric(_first_path(baseline, (("false_alerts_per_hour",), ("false_alarm_rate",))), 0.0, float("inf"))
    checks["false_alerts_per_hour"] = candidate_false is not None and ((baseline_false is not None and candidate_false <= baseline_false) or (baseline_false is None and false_ceiling is not None and candidate_false <= false_ceiling))

    ece_bound = _bounded_metric(_candidate_value(candidate_artifacts, ("calibration_ece_ceiling",), calibration_ece_ceiling), 0.0, 1.0)
    calibration_valid = calibration.get("valid") is True or calibration.get("status") in {"valid", "passed"} or calibration.get("bounded") is True
    ece = _bounded_metric(_first_path(calibration, (("ece",), ("expected_calibration_error",))), 0.0, 1.0)
    checks["calibration"] = calibration_valid and ece is not None and ece_bound is not None and ece <= ece_bound

    raw_coverage_floor = _candidate_value(candidate_artifacts, ("minimum_coverage", "coverage_floor"), constraints.get("minimum_coverage"))
    if raw_coverage_floor is None:
        raw_coverage_floor = config.get("minimum_coverage")
    coverage_floor = _bounded_metric(raw_coverage_floor, 0.0, 1.0)
    if coverage_floor is None and raw_coverage_floor is None:
        coverage_floor = _DEFAULT_COVERAGE_FLOOR
    candidate_coverage = _bounded_metric(_first_path(aggregate, (("coverage",), ("selective_coverage",))), 0.0, 1.0)
    checks["coverage"] = candidate_coverage is not None and coverage_floor is not None and candidate_coverage >= coverage_floor

    candidate_aurc = _bounded_metric(_first_path(aggregate, (("aurc",), ("risk_coverage_aurc",))), 0.0, 1.0)
    baseline_aurc = _bounded_metric(_first_path(baseline, (("aurc",), ("risk_coverage_aurc",))), 0.0, 1.0)
    explicit_aurc = _candidate_value(candidate_artifacts, ("aurc_improved",), None)
    checks["aurc"] = (explicit_aurc if type(explicit_aurc) is bool else False) if explicit_aurc is not None else (candidate_aurc is not None and baseline_aurc is not None and candidate_aurc < baseline_aurc)

    seeds = _candidate_value(candidate_artifacts, ("seeds", "seed_results"), nested.get("seeds", nested.get("seed_results")))
    checks["seeds"] = len(_unique_seeds(seeds)) >= 3
    continuous_checks = continuous.get("checks")
    checks["continuous"] = continuous.get("passed") is True and isinstance(continuous_checks, Mapping) and bool(continuous_checks) and all(value is True for value in continuous_checks.values())
    checks["hashes"] = _hash_checks(candidate_artifacts, continuous, config)
    reasons = [name for name, passed in checks.items() if not passed]
    result: dict[str, Any] = {"schema_version": "rgpc.promotion.v1", "release_id": release_id, "promoted": not reasons, "reasons": reasons, "promotion_checks": checks}
    if reasons:
        result["release_dir"] = None
        return result

    destination_raw = output_dir if output_dir is not None else _candidate_value(candidate_artifacts, ("output_dir", "release_dir"), None)
    if destination_raw is None:
        raise ValueError("output_dir is required for a passing release")
    destination = Path(destination_raw)
    if destination.name.lower() != "release":
        destination = destination / "release"
    result["release_dir"] = str(destination)

    def write_staging(staging: Path) -> None:
        source_raw = _candidate_value(candidate_artifacts, ("source_dir",), None)
        if source_raw is not None and Path(source_raw).is_dir():
            for source_file in Path(source_raw).iterdir():
                if source_file.is_file() and source_file.name not in {"promotion_result.json", "promotion_checks.json"}:
                    shutil.copy2(source_file, staging / source_file.name)
        _write_json(staging / "release_config.json", dict(config))
        _write_json(staging / "continuous_promotion_gate.json", dict(continuous))
        _write_json(staging / "nested_loso_summary.json", dict(nested))
        _write_json(staging / "promotion_checks.json", checks)
        _write_json(staging / "promotion_result.json", result)

    _publish_artifacts(destination, write_staging)
    return result


# Short alias used by callers that name the operation after its artifact.
evaluate_nested_loso = evaluate_rg_pcnet_loso


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-oof", required=True, type=Path)
    parser.add_argument("--outer-predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--release-id", default="rgpc-loso-v1")
    parser.add_argument("--recall-floor", type=float, default=_DEFAULT_RECALL_FLOOR)
    parser.add_argument("--fpr-ceiling", type=float, default=_DEFAULT_FPR_CEILING)
    parser.add_argument("--coverage-floor", type=float, default=_DEFAULT_COVERAGE_FLOOR)
    parser.add_argument("--confirm-seconds", type=float, default=_DEFAULT_CONFIRM_SECONDS)
    parser.add_argument("--recovery-seconds", type=float, default=_DEFAULT_RECOVERY_SECONDS)
    parser.add_argument("--cooldown-seconds", type=float, default=_DEFAULT_COOLDOWN_SECONDS)
    parser.add_argument("--outer-subject")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_rg_pcnet_loso(
        args.inner_oof,
        args.outer_predictions,
        args.output,
        checkpoint_path=args.checkpoint,
        dataset_manifest_path=args.dataset_manifest,
        split_manifest_path=args.split_manifest,
        release_id=args.release_id,
        recall_floor=args.recall_floor,
        fpr_ceiling=args.fpr_ceiling,
        coverage_floor=args.coverage_floor,
        confirm_seconds=args.confirm_seconds,
        recovery_seconds=args.recovery_seconds,
        cooldown_seconds=args.cooldown_seconds,
        outer_subject=args.outer_subject,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
