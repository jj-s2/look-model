"""Evaluate PMCC forecast records without mixing subjects or evidence releases.

The evaluator is intentionally artifact-oriented: it consumes already scored
JSONL records, never contacts a device, and makes no medical-performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk.pmcc.schema import EvidenceTier


HORIZONS = (("24h", 1), ("72h", 3), ("7d", 7))
ABLATIONS = ("baseline", "chains", "sleep_physiology", "gait", "mask", "quality_gate", "uncertainty", "tcn")
NON_RELEASE_TIERS = {EvidenceTier.SYNTHETIC_RESEARCH.value, EvidenceTier.OFFLINE_FIXTURE.value}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError("input must be a JSONL file")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL line {number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL line {number} must be an object")
        records.append(row)
    if not records:
        raise ValueError("input contains no records")
    return records


def _read_model_card(model: Path) -> dict[str, Any]:
    path = model / "model-card.json"
    if not path.is_file():
        raise ValueError("model must contain model-card.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("model-card.json must be JSON") from error
    if not isinstance(value, dict):
        raise ValueError("model-card.json must be an object")
    return value


def _provenance(record: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("every evaluation record requires provenance")
    tier = provenance.get("evidence_tier")
    if tier not in {item.value for item in EvidenceTier}:
        raise ValueError("record provenance requires a valid evidence_tier")
    return provenance


def _subject(record: Mapping[str, Any]) -> str:
    observation = record.get("observation", record)
    if not isinstance(observation, Mapping) or not isinstance(observation.get("subject_id"), str) or not observation["subject_id"]:
        raise ValueError("every record requires observation.subject_id")
    return observation["subject_id"]


def _observed_at(record: Mapping[str, Any]) -> datetime:
    observation = record.get("observation", record)
    if not isinstance(observation, Mapping) or not isinstance(observation.get("observed_at"), str):
        raise ValueError("every record requires an ISO-8601 observation timestamp")
    try:
        value = datetime.fromisoformat(observation["observed_at"])
    except ValueError as error:
        raise ValueError("observation.observed_at must be ISO-8601") from error
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observation.observed_at must be timezone-aware")
    return value.astimezone(timezone.utc)


def _label(record: Mapping[str, Any]) -> tuple[int | None, int]:
    label = record.get("label")
    if not isinstance(label, Mapping):
        raise ValueError("every evaluation record requires a label")
    value = label.get("event_day")
    censor = label.get("censor_day")
    if isinstance(censor, bool) or not isinstance(censor, int) or not 1 <= censor <= 7:
        raise ValueError("label.censor_day must be an integer in [1, 7]")
    if value is None:
        return None, censor
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= censor:
        raise ValueError("label.event_day must be null or an integer in [1, 7]")
    return value, censor


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return float(value)


def _predictions(record: Mapping[str, Any]) -> Mapping[str, Mapping[str, float]] | None:
    values = record.get("predictions")
    if values is None:
        return None
    if not isinstance(values, Mapping) or not values:
        raise ValueError("predictions must be a non-empty mapping when supplied")
    parsed: dict[str, Mapping[str, float]] = {}
    for name, row in values.items():
        if not isinstance(name, str) or not isinstance(row, Mapping):
            raise ValueError("predictions must map model names to horizon mappings")
        result = {horizon: _probability(row.get(horizon), f"predictions[{name}][{horizon}]") for horizon, _ in HORIZONS}
        if not result["24h"] <= result["72h"] <= result["7d"]:
            raise ValueError("prediction horizons must be monotonic")
        parsed[name] = result
    if not {"baseline", "full"}.issubset(parsed):
        raise ValueError("predictions require baseline and full model rows")
    return parsed


def _auc(scores: list[float], labels: list[int]) -> float | None:
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in (score for score, label in zip(scores, labels) if label):
        for negative in (score for score, label in zip(scores, labels) if not label):
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (positives * negatives)


def _average_precision(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    if not positives:
        return None
    ordered = sorted(zip(scores, labels), reverse=True)
    hits = 0
    total = 0.0
    for rank, (_, label) in enumerate(ordered, 1):
        if label:
            hits += 1
            total += hits / rank
    return total / positives


def _calibration(scores: list[float], labels: list[int]) -> dict[str, float | None]:
    if not scores:
        return {"brier": None, "expected": None, "observed": None, "absolute_gap": None}
    expected, observed = sum(scores) / len(scores), sum(labels) / len(labels)
    return {"brier": sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(scores),
            "expected": expected, "observed": observed, "absolute_gap": abs(expected - observed)}


def _threshold(scores: Iterable[float], budget: float = 0.10) -> float | None:
    values = sorted(scores, reverse=True)
    if not values:
        return None
    return values[min(len(values) - 1, max(0, math.ceil(len(values) * budget) - 1))]


def _unavailable_metrics(reason: str) -> dict[str, float | int | str | None]:
    return {"auroc": None, "auprc": None, "brier": None, "expected": None, "observed": None, "absolute_gap": None,
            "alert_budget": 0.10, "alert_threshold": None, "alert_precision": None, "alert_recall": None,
            "false_alerts_per_subject_day": None, "lead_time_days": None, "evaluated_records": 0,
            "censored_before_horizon": 0, "reason": reason}


def _metrics(records: list[dict[str, Any]], model_name: str, horizon_days: int, horizon: str) -> dict[str, float | int | str | None]:
    if any(record["_predictions"] is None for record in records):
        return _unavailable_metrics("scored_predictions_unavailable")
    known = [record for record in records if _label(record)[1] >= horizon_days]
    eligible = [record for record in known if not bool(record.get("abstained", False))]
    scores = [record["_predictions"][model_name][horizon] for record in eligible]
    labels = [int((_label(record)[0] or 99) <= horizon_days) for record in eligible]
    calibration = _calibration(scores, labels)
    thresholds = [record.get("_inner_thresholds", {}).get(model_name, {}).get(horizon) for record in eligible]
    alerts = [threshold is not None and score >= threshold for score, threshold in zip(scores, thresholds)]
    tp = sum(alert and label for alert, label in zip(alerts, labels))
    fp = sum(alert and not label for alert, label in zip(alerts, labels))
    positives = sum(labels)
    alert_count = sum(alerts)
    lead_times = [_label(record)[0] for record, alert, label in zip(eligible, alerts, labels) if alert and label]
    subject_days = max(1, len({(_subject(record), record.get("observation", {}).get("observed_at")) for record in eligible}))
    return {"auroc": _auc(scores, labels), "auprc": _average_precision(scores, labels), **calibration,
            "alert_budget": 0.10, "alert_threshold": "inner_fold_only",
            "alert_precision": None if not alert_count else tp / alert_count,
            "alert_recall": None if not positives else tp / positives,
            "false_alerts_per_subject_day": fp / subject_days,
            "lead_time_days": None if not lead_times else sum(lead_times) / len(lead_times),
            "evaluated_records": len(eligible), "censored_before_horizon": len(records) - len(known), "reason": None}


def _split_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[int, set[str]] = defaultdict(set)
    for record in records:
        subject = _subject(record)
        fold = int(hashlib.sha256(subject.encode("utf-8")).hexdigest()[:8], 16) % 5
        groups[fold].add(subject)
    overlapping = any(left & right for left_key, left in groups.items() for right_key, right in groups.items() if left_key < right_key)
    return {"outer_folds": {str(key): sorted(value) for key, value in sorted(groups.items())}, "passed": not overlapping,
            "rule": "a subject is assigned to exactly one deterministic outer fold; calibration remains inside training folds"}


def _outer_held_out_records(records: list[dict[str, Any]], model_names: Iterable[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign every scored window to a subject-disjoint held-out outer fold.

    The alert cutoff for an outer test window is derived exclusively from other
    subjects observed no later than that fold's held-out time boundary.
    """
    subjects = sorted({_subject(record) for record in records})
    count = min(5, len(subjects))
    assignments = {subject: int(hashlib.sha256(subject.encode("utf-8")).hexdigest()[:8], 16) % count for subject in subjects}
    held_out: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for fold in sorted(set(assignments.values())):
        test_subjects = sorted(subject for subject, assigned in assignments.items() if assigned == fold)
        test = [record for record in records if _subject(record) in test_subjects]
        training_subjects = sorted(set(subjects) - set(test_subjects))
        per_window: list[dict[str, Any]] = []
        for record in test:
            cutoff = _observed_at(record)
            training = [candidate for candidate in records if _subject(candidate) in training_subjects and _observed_at(candidate) <= cutoff]
            thresholds: dict[str, dict[str, float | None]] = {}
            for name in model_names:
                thresholds[name] = {}
                for horizon, _ in HORIZONS:
                    source = [candidate["_predictions"][name][horizon] for candidate in training if candidate["_predictions"] is not None and not bool(candidate.get("abstained", False))]
                    thresholds[name][horizon] = _threshold(source)
            record["_inner_thresholds"] = thresholds
            latest = max((_observed_at(candidate) for candidate in training), default=None)
            per_window.append({"subject_id": _subject(record), "as_of_timestamp": cutoff.isoformat(),
                               "calibration_records": len(training), "calibration_latest_timestamp": None if latest is None else latest.isoformat()})
        held_out.extend(test)
        folds.append({"fold": fold, "test_subjects": test_subjects, "training_subjects": training_subjects,
                      "held_out_records": len(test),
                      "inner_calibration": {"time_blocked": True, "per_window": per_window}})
    return held_out, {"evaluation_population": "outer_held_out_subject_windows_only", "folds": folds}


def _stratified(records: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(selector: Any) -> float | None:
        selected = [record for record in records if selector(record)]
        return None if not selected else sum(bool(record.get("abstained", False)) for record in selected) / len(selected)
    groups = sorted({record.get("subgroup") for record in records if isinstance(record.get("subgroup"), str)})
    subgroup = {group: {"records": sum(record.get("subgroup") == group for record in records), "abstention_rate": rate(lambda record, value=group: record.get("subgroup") == value)} for group in groups}
    subgroup["unavailable"] = None
    return {"coverage": {"lt_0_5": rate(lambda record: float(record.get("coverage", 0.0)) < 0.5), "gte_0_5": rate(lambda record: float(record.get("coverage", 0.0)) >= 0.5)},
            "quality": {"lt_0_5": rate(lambda record: float(record.get("quality", 0.0)) < 0.5), "gte_0_5": rate(lambda record: float(record.get("quality", 0.0)) >= 0.5)},
            "evidence_tier": {tier: sum(_provenance(record)["evidence_tier"] == tier for record in records) for tier in sorted({_provenance(record)["evidence_tier"] for record in records})},
            "subgroup": subgroup}


def evaluate(records: list[dict[str, Any]], model_card: Mapping[str, Any]) -> dict[str, Any]:
    releases = {record.get("release_id") for record in records}
    if len(releases) != 1 or not isinstance(next(iter(releases)), str) or not next(iter(releases)):
        raise ValueError("evaluation input must contain exactly one release_id")
    release_id = next(iter(releases))
    tiers = {_provenance(record)["evidence_tier"] for record in records}
    if len(tiers) != 1:
        raise ValueError("evaluation input must contain exactly one evidence tier")
    tier = next(iter(tiers))
    card_tier = model_card.get("evidence_tier")
    if card_tier != tier:
        raise ValueError("model and evaluation evidence tiers must agree")
    card_release = model_card.get("release_id")
    if tier not in NON_RELEASE_TIERS and card_release != release_id:
        raise ValueError("model and evaluation release_id must agree")
    for record in records:
        record["_predictions"] = _predictions(record)
        _label(record)
    missing_scores = any(record["_predictions"] is None for record in records)
    if missing_scores and tier not in NON_RELEASE_TIERS:
        raise ValueError("scored predictions are required for real/release evaluation")
    model_names = ["baseline", "full"] if missing_scores else sorted({name for record in records for name in record["_predictions"].keys()})
    evaluated_records, split_metadata = _outer_held_out_records(records, model_names)
    model_rows = {name: {"horizons": {horizon: _metrics(evaluated_records, name, days, horizon) for horizon, days in HORIZONS}} for name in model_names}
    non_release = tier in NON_RELEASE_TIERS or model_card.get("promoted") is not True
    return {"schema_version": "pmcc.evaluation.v1", "claim_boundary": "research_only" if non_release else "research_evaluation_not_clinical",
            "release_metrics_eligible": not non_release, "split_strategy": "subject_grouped_outer_with_inner_calibration",
            "leakage_check": _split_audit(records), "split_metadata": split_metadata, "provenance": {"release_id": release_id, "dataset_id": records[0].get("dataset_id"), "model_artifact_id": model_card.get("artifact_id"), "evidence_tier": tier, "promoted": False if non_release else True},
            "model_rows": model_rows, "ablations": {name: {"status": "unavailable", "reason": "not separately scored in this artifact"} for name in ABLATIONS},
            "coverage": {"records": len(evaluated_records), "non_abstained_records": sum(not bool(record.get("abstained", False)) for record in evaluated_records), "abstention_rate": sum(bool(record.get("abstained", False)) for record in evaluated_records) / len(evaluated_records)},
            "stratified": _stratified(records), "method": {"outer_split": "subject_grouped", "calibration": "inner_training_folds_only", "association_only": True, "clinical_use": False,
            "score_source": "artifact_predictions_or_unavailable"}}


def _write_json(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate scored PMCC forecasts with subject-grouped audit metadata.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-mode", action="store_true", help="reject non-release evidence after writing its audit artifact")
    args = parser.parse_args(argv)
    result = evaluate(_read_jsonl(args.input), _read_model_card(args.model))
    _write_json(args.output, result)
    if args.release_mode and not result["release_metrics_eligible"]:
        raise ValueError("synthetic or offline evidence is research_only and cannot support release metrics")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(f"evaluation failed: {error}")
