"""Evaluate timestamped RG-PCNet predictions and emit auditable artifacts.

The command intentionally treats predictions as already-produced model output:
truth intervals never enter the replay predictor.  Every input is parsed with
strict JSON rules and every artifact is written through a temporary staging
directory before it is moved into place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

# Keep direct ``python scripts/evaluate_continuous_rg_pcnet.py`` invocation
# equivalent to module execution when launched from outside the repository.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk.phase_model.continuous_replay import ReplayFrame, replay_stream
from risk.phase_model.event_evaluation import Alert, TruthEvent, evaluate_continuous_events
from risk.phase_model.event_state import EventDecoder, EventDecoderConfig, FrameDecision
from risk.phase_model.release_config import RGPCReleaseConfig, load_release_config


_OUTPUT_NAMES = (
    "continuous_metrics.json",
    "transitions.jsonl",
    "alerts.jsonl",
    "promotion_gate.json",
)
_MISSING = object()


class _InvalidJSONConstant(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise _InvalidJSONConstant(value)


def _pairs_without_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_bytes(raw: bytes, *, source: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source} must be valid UTF-8 JSON") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, _InvalidJSONConstant) as exc:
        raise ValueError(f"{source} contains invalid JSON") from exc


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _probability(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _required(mapping: dict[str, Any], name: str) -> Any:
    value = mapping.get(name, _MISSING)
    if value is _MISSING:
        raise ValueError(f"missing required field: {name}")
    return value


def _read_predictions(path: Path, release_id: str) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    previous: float | None = None
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"prediction JSONL line {line_number} is empty")
        value = _parse_json_bytes(line, source=f"prediction line {line_number}")
        if type(value) is not dict:
            raise ValueError(f"prediction line {line_number} must be a JSON object")
        row_release = _required(value, "release_id")
        if type(row_release) is not str or row_release != release_id:
            raise ValueError("prediction release_id mismatch")
        timestamp = _finite_number(_required(value, "timestamp"), "timestamp")
        if previous is not None and timestamp <= previous:
            raise ValueError("prediction timestamps must be strictly increasing")
        previous = timestamp
        probability = _probability(
            _required(value, "fall_probability"), "fall_probability"
        )
        phase = _required(value, "phase")
        if type(phase) is not str or not phase:
            raise ValueError("phase must be a non-empty string")
        reliable = _required(value, "reliable")
        if type(reliable) is not bool:
            raise ValueError("reliable must be a boolean")
        row = dict(value)
        row.update(
            {
                "release_id": row_release,
                "timestamp": timestamp,
                "fall_probability": probability,
                "phase": phase,
                "reliable": reliable,
            }
        )
        rows.append(row)
    if not rows:
        raise ValueError("prediction JSONL must contain at least one record")
    return rows, raw


def _read_truth(path: Path, release_id: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = _parse_json_bytes(raw, source="truth events")
    if type(value) is not dict:
        raise ValueError("truth events root must be a JSON object")
    if _required(value, "release_id") != release_id:
        raise ValueError("truth release_id mismatch")
    events = _required(value, "events")
    if type(events) is not list:
        raise ValueError("truth events must be a JSON array")
    duration = _finite_number(_required(value, "duration_seconds"), "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    tolerance = _finite_number(
        _required(value, "tolerance_seconds"), "tolerance_seconds"
    )
    if tolerance < 0:
        raise ValueError("tolerance_seconds must be non-negative")
    parsed: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if type(event) is not dict:
            raise ValueError(f"truth event {index} must be a JSON object")
        event_id = _required(event, "event_id")
        if type(event_id) is not str or not event_id:
            raise ValueError("truth event_id must be a non-empty string")
        start = _finite_number(_required(event, "start"), "truth start")
        end = _finite_number(_required(event, "end"), "truth end")
        if end < start:
            raise ValueError("truth event end must be greater than or equal to start")
        parsed.append({"event_id": event_id, "start": start, "end": end})
    return {"events": parsed, "duration_seconds": duration, "tolerance_seconds": tolerance}, raw


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _aggregate_hash(hashes: dict[str, str]) -> str:
    canonical = "".join(f"{name}:{hashes[name]}\n" for name in sorted(hashes))
    return _sha256_bytes(canonical.encode("utf-8"))


def _publish_staged_files(staged: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    backups: dict[str, Path] = {}
    installed: list[Path] = []
    try:
        for name in _OUTPUT_NAMES:
            os.replace(staged / name, temporary_output / name)
        for name in _OUTPUT_NAMES:
            destination = output / name
            if destination.exists():
                backup = temporary_output / f".{name}.old"
                os.replace(destination, backup)
                backups[name] = backup
            os.replace(temporary_output / name, destination)
            installed.append(destination)
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    except Exception:
        for destination in installed:
            destination.unlink(missing_ok=True)
        for name, backup in backups.items():
            if backup.exists():
                try:
                    os.replace(backup, output / name)
                except OSError:
                    pass
        raise
    finally:
        shutil.rmtree(temporary_output, ignore_errors=True)
        shutil.rmtree(staged, ignore_errors=True)


def evaluate_continuous(
    *,
    predictions: os.PathLike[str] | str,
    truth_events: os.PathLike[str] | str,
    release_config: os.PathLike[str] | str,
    output: os.PathLike[str] | str,
    minimum_event_recall: float,
    maximum_false_alerts_per_hour: float,
    minimum_coverage: float,
) -> dict[str, Path]:
    """Replay predictions and atomically emit four canonical evidence files."""

    prediction_path = Path(predictions)
    truth_path = Path(truth_events)
    release_path = Path(release_config)
    output_path = Path(output)
    config = load_release_config(release_path)
    rows, prediction_raw = _read_predictions(prediction_path, config.release_id)
    truth, truth_raw = _read_truth(truth_path, config.release_id)
    release_raw = release_path.read_bytes()
    recall_limit = _probability(minimum_event_recall, "minimum_event_recall")
    false_limit = _finite_number(
        maximum_false_alerts_per_hour, "maximum_false_alerts_per_hour"
    )
    if false_limit < 0:
        raise ValueError("maximum_false_alerts_per_hour must be non-negative")
    coverage_limit = _probability(minimum_coverage, "minimum_coverage")

    decoder = EventDecoder(
        EventDecoderConfig(
            config.fall_threshold,
            config.confirm_seconds,
            config.recovery_seconds,
            config.cooldown_seconds,
        )
    )
    frames = [ReplayFrame(row["timestamp"], {}) for row in rows]
    row_iter = iter(rows)

    def predictor(frame: ReplayFrame) -> FrameDecision:
        row = next(row_iter)
        if row["timestamp"] != frame.timestamp:
            raise ValueError("prediction timestamp does not match replay frame")
        return FrameDecision(
            frame.timestamp,
            row["fall_probability"],
            row["phase"],
            row["reliable"],
        )

    replay = replay_stream(frames, predictor=predictor, decoder=decoder)
    truth_objects = [TruthEvent(item["event_id"], item["start"], item["end"]) for item in truth["events"]]
    alert_objects = [
        Alert(transition.event_id or f"alert-{index:06d}", transition.timestamp)
        for index, transition in enumerate(replay.alerts)
        if transition.emitted == "fall_confirmed"
    ]
    metrics = evaluate_continuous_events(
        truth_objects,
        alert_objects,
        duration_seconds=truth["duration_seconds"],
        tolerance_seconds=truth["tolerance_seconds"],
    ).to_dict()
    metrics.update(
        {
            "release_id": config.release_id,
            "coverage": replay.coverage,
            "abstained_frames": replay.abstained_frames,
            "total_frames": replay.total_frames,
        }
    )

    transition_records = [
        {
            "event_id": transition.event_id,
            "emitted": transition.emitted,
            "reason": transition.reason,
            "state": transition.state,
            "timestamp": transition.timestamp,
        }
        for transition in replay.transitions
    ]
    alert_records = [
        {
            "event_id": transition.event_id,
            "emitted": transition.emitted,
            "reason": transition.reason,
            "state": transition.state,
            "timestamp": transition.timestamp,
        }
        for transition in replay.alerts
    ]
    checks = {
        "event_recall": metrics["event_recall"] >= recall_limit,
        "false_alerts_per_hour": metrics["false_alerts_per_hour"] <= false_limit,
        "coverage": metrics["coverage"] >= coverage_limit,
    }
    input_hashes = {
        "predictions": _sha256_bytes(prediction_raw),
        "truth_events": _sha256_bytes(truth_raw),
        "release_config": _sha256_bytes(release_raw),
    }
    source = {
        "predictions": str(prediction_path.resolve()),
        "truth_events": str(truth_path.resolve()),
        "release_config": str(release_path.resolve()),
        "truth_boundary": "evaluator_only",
        "decoder_timers_from_release": True,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.stage.", dir=output_path.parent)
    )
    try:
        metric_bytes = _canonical_json(metrics)
        transition_bytes = _jsonl(transition_records)
        alert_bytes = _jsonl(alert_records)
        non_gate_hashes = {
            "continuous_metrics.json": _sha256_bytes(metric_bytes),
            "transitions.jsonl": _sha256_bytes(transition_bytes),
            "alerts.jsonl": _sha256_bytes(alert_bytes),
        }
        gate = {
            "checks": checks,
            "constraints": {
                "minimum_event_recall": recall_limit,
                "maximum_false_alerts_per_hour": false_limit,
                "minimum_coverage": coverage_limit,
            },
            "input_hashes": input_hashes,
            "input_sha256": _aggregate_hash(input_hashes),
            "output_sha256": dict(non_gate_hashes),
            "release_id": config.release_id,
            "source_provenance": source,
            "passed": all(checks.values()),
        }
        gate["output_sha256_scope"] = (
            "promotion_gate.json hash is over canonical gate JSON with its "
            "self entry removed"
        )
        # A file cannot contain the hash of its exact final bytes without a
        # fixed-point hash.  Hash a canonical copy of the final gate with only
        # this self entry removed; the other three values hash their exact
        # published bytes.
        self_excluding_gate = dict(gate)
        self_excluding_gate["output_sha256"] = dict(gate["output_sha256"])
        gate["output_sha256"]["promotion_gate.json"] = _sha256_bytes(
            _canonical_json(self_excluding_gate)
        )
        files = {
            "continuous_metrics.json": metric_bytes,
            "transitions.jsonl": transition_bytes,
            "alerts.jsonl": alert_bytes,
            "promotion_gate.json": _canonical_json(gate),
        }
        for name, content in files.items():
            (staged / name).write_bytes(content)
        _publish_staged_files(staged, output_path)
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return {key.removesuffix(".json").removesuffix(".jsonl"): output_path / key for key in _OUTPUT_NAMES}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--truth-events", required=True, type=Path)
    parser.add_argument("--release-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-event-recall", required=True, type=float)
    parser.add_argument("--maximum-false-alerts-per-hour", required=True, type=float)
    parser.add_argument("--minimum-coverage", required=True, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate_continuous(
        predictions=args.predictions,
        truth_events=args.truth_events,
        release_config=args.release_config,
        output=args.output,
        minimum_event_recall=args.minimum_event_recall,
        maximum_false_alerts_per_hour=args.maximum_false_alerts_per_hour,
        minimum_coverage=args.minimum_coverage,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
