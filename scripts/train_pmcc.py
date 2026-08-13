"""Train the CPU PMCC calibrator from provenance-validated JSONL examples.

This command intentionally produces an auditable research artifact rather than
a deployable clinical model.  Its JSON outputs exclude local paths, device
identifiers and estimator pickles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk.pmcc.calibrator import RuleSurvivalCalibrator
from risk.pmcc.baseline import BaselineManager
from risk.pmcc.features import FeatureWindow, build_feature_window
from risk.pmcc.schema import DailyObservation, EvidenceTier
from risk.pmcc.survival import SurvivalLabel


NON_RELEASE_TIERS = {EvidenceTier.SYNTHETIC_RESEARCH, EvidenceTier.OFFLINE_FIXTURE}


def read_jsonl(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.is_file():
        raise ValueError("input must be a JSONL file")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL source line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"source line {line_number} must contain an object")
        rows.append(row)
    if not rows:
        raise ValueError("input contains no JSONL records")
    return rows


def _observation_data(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("observation", record)
    if not isinstance(value, Mapping):
        raise ValueError("record observation must be an object")
    return value


def _record_provenance(record: Mapping[str, Any], observation: Mapping[str, Any]) -> Mapping[str, Any]:
    """Merge provenance without allowing a wrapper to upgrade evidence."""
    top_level = record.get("provenance")
    embedded = observation.get("provenance")
    if top_level is not None and not isinstance(top_level, Mapping):
        raise ValueError("record provenance must be an object")
    if embedded is not None and not isinstance(embedded, Mapping):
        raise ValueError("observation provenance must be an object")
    if top_level is None and embedded is None:
        raise ValueError("record provenance is required")
    if top_level is None:
        return dict(embedded)
    if embedded is None:
        return dict(top_level)
    top_tier = _tier(top_level)
    embedded_tier = _tier(embedded)
    if top_tier is not embedded_tier:
        raise ValueError("top-level and observation provenance evidence_tier must agree")
    merged = dict(embedded)
    merged.update(top_level)
    merged["evidence_tier"] = embedded_tier.value
    merged["promoted"] = embedded.get("promoted") is True and top_level.get("promoted") is True
    return merged


def _tier(provenance: Mapping[str, Any]) -> EvidenceTier:
    try:
        return EvidenceTier(provenance.get("evidence_tier"))
    except (TypeError, ValueError) as error:
        raise ValueError("record provenance requires a valid evidence_tier") from error


def _label(record: Mapping[str, Any]) -> SurvivalLabel:
    value = record.get("label")
    if not isinstance(value, Mapping):
        raise ValueError("every training record requires a label")
    return SurvivalLabel(value.get("event_day"), value.get("censor_day"))


def _feature_bases(observations: list[DailyObservation]) -> tuple[str, ...]:
    names = sorted({name for observation in observations for name in observation.features})
    if not names:
        raise ValueError("training records must contain at least one feature")
    return tuple(names)


def _training_windows(observations: list[DailyObservation], feature_bases: tuple[str, ...]) -> tuple[FeatureWindow, ...]:
    """Build service-compatible windows with only prior-day baseline data."""
    by_subject: dict[str, list[DailyObservation]] = {}
    for observation in observations:
        by_subject.setdefault(observation.subject_id, []).append(observation)
    windows: list[FeatureWindow] = []
    for observation in observations:
        history = sorted(
            (item for item in by_subject[observation.subject_id] if item.observed_at.date() < observation.observed_at.date()),
            key=lambda item: item.observed_at,
        )
        baseline = BaselineManager()
        for prior in history:
            baseline.add(prior)
        # Include the target day in the fixed window, but never in baseline
        # updates used to compute its directional z values.
        windows.append(
            build_feature_window(
                (*history, observation), baseline, (), observation.observed_at.date(), feature_names=feature_bases
            )
        )
    return tuple(windows)


def _artifact_id(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return f"pmcc-model-{digest.hexdigest()[:16]}"


def train(records: list[dict[str, Any]], seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], RuleSurvivalCalibrator]:
    observations: list[DailyObservation] = []
    labels: list[SurvivalLabel] = []
    tiers: set[EvidenceTier] = set()
    promoted_inputs = True
    for record in records:
        observation_data = _observation_data(record)
        observation = DailyObservation.from_dict(observation_data)
        provenance = _record_provenance(record, observation_data)
        observations.append(observation)
        labels.append(_label(record))
        tiers.add(_tier(provenance))
        promoted_inputs = promoted_inputs and provenance.get("promoted") is True
    if len(tiers) != 1:
        raise ValueError("training input must contain exactly one evidence tier")
    tier = next(iter(tiers))
    feature_bases = _feature_bases(observations)
    windows = _training_windows(observations, feature_bases)
    schema = windows[0].feature_names
    model = RuleSurvivalCalibrator(random_seed=seed, evidence_tier=tier.value).fit(windows, labels)
    promoted = bool(promoted_inputs and tier not in NON_RELEASE_TIERS)
    artifact_id = _artifact_id(records)
    model_card = {
        "artifact_id": artifact_id,
        "model_type": model.metadata()["model_type"],
        "random_seed": seed,
        "feature_schema": list(schema),
        "evidence_tier": tier.value,
        "promoted": promoted,
        "clinical_use": False,
        "claim_boundary": "research_only; temporal association is not medical causality",
        "training_examples": len(records),
        "subject_level_split_required": True,
        "tcn_evaluation_claim": None,
        "artifact_loader": "RuleSurvivalCalibrator.load_artifact(path)",
    }
    release_ids = {record.get("release_id") for record in records if record.get("release_id") is not None}
    if len(release_ids) > 1:
        raise ValueError("training input must belong to one release_id")
    if release_ids:
        model_card["release_id"] = next(iter(release_ids))
    eligible_for_release_metrics = tier not in NON_RELEASE_TIERS and promoted
    metrics = {
        "artifact_id": artifact_id,
        "training_examples": len(records),
        "event_examples": sum(label.event_day is not None for label in labels),
        "non_event_examples": sum(label.event_day is None for label in labels),
        "release_metrics": {} if eligible_for_release_metrics else None,
        "release_metrics_eligible": eligible_for_release_metrics,
        "reason": None if eligible_for_release_metrics else "synthetic_or_offline_evidence_cannot_support_release_metrics",
    }
    provenance = {
        "artifact_id": artifact_id,
        "schema_version": "pmcc.model.v1",
        "evidence_tier": tier.value,
        "promoted": promoted,
        "input_record_count": len(records),
        "feature_schema": list(schema),
        "contains_absolute_paths": False,
    }
    return model_card, metrics, provenance, model


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a CPU PMCC research calibrator from JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="labeled PMCC JSONL input")
    parser.add_argument("--output", required=True, type=Path, help="model artifact directory")
    parser.add_argument("--seed", default=42, type=int, help="deterministic model seed")
    args = parser.parse_args(argv)
    card, metrics, provenance, model = train(read_jsonl(args.input), args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "model-card.json", card)
    _write_json(args.output / "metrics.json", metrics)
    _write_json(args.output / "provenance.json", provenance)
    model.save_artifact(args.output / "model.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(f"training failed: {error}")
