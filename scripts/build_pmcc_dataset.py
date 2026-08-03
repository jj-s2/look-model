"""Validate PMCC JSONL examples and add deterministic dataset provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk.pmcc.schema import DailyObservation, EvidenceTier
from risk.pmcc.survival import SurvivalLabel


SCHEMA_VERSION = "pmcc.dataset.v1"


def read_jsonl(source: Path) -> list[dict[str, Any]]:
    paths: Iterable[Path] = sorted(source.glob("*.jsonl")) if source.is_dir() else (source,)
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise ValueError("input must be a JSONL file or directory containing JSONL files")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL source line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"source line {line_number} must contain an object")
            records.append(record)
    if not records:
        raise ValueError("input contains no JSONL records")
    return records


def _nested_observation(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("observation", record)
    if not isinstance(value, Mapping):
        raise ValueError("record observation must be an object")
    return value


def _provenance(record: Mapping[str, Any], observation: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = record.get("provenance", observation.get("provenance"))
    if not isinstance(provenance, Mapping):
        raise ValueError("record provenance is required")
    return provenance


def _tier(provenance: Mapping[str, Any]) -> EvidenceTier:
    try:
        return EvidenceTier(provenance.get("evidence_tier"))
    except (TypeError, ValueError) as error:
        raise ValueError("record provenance requires a valid evidence_tier") from error


def _content_id(prefix: str, records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return f"{prefix}-{digest.hexdigest()[:16]}"


def _split_id(subject_id: str) -> str:
    bucket = int(hashlib.sha256(subject_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
    return f"subject-{split}-{hashlib.sha256(subject_id.encode('utf-8')).hexdigest()[:12]}"


def build_dataset(records: list[dict[str, Any]], expected_tier: EvidenceTier) -> tuple[dict[str, Any], ...]:
    validated: list[tuple[dict[str, Any], DailyObservation, Mapping[str, Any]]] = []
    for record in records:
        observation_data = _nested_observation(record)
        observation = DailyObservation.from_dict(observation_data)
        provenance = _provenance(record, observation_data)
        tier = _tier(provenance)
        if tier is not expected_tier:
            raise ValueError("input evidence tiers are mixed or do not match --evidence-tier")
        if "label" in record:
            label = record["label"]
            if not isinstance(label, Mapping):
                raise ValueError("label must be an object")
            SurvivalLabel(label.get("event_day"), label.get("censor_day"))
        validated.append((dict(record), observation, provenance))
    dataset_id = _content_id("pmcc-dataset", records)
    release_id = _content_id("pmcc-release", records)
    built: list[dict[str, Any]] = []
    for record, observation, provenance in validated:
        safe_provenance = dict(provenance)
        safe_provenance["evidence_tier"] = expected_tier.value
        safe_provenance["promoted"] = bool(provenance.get("promoted") is True and expected_tier not in {EvidenceTier.SYNTHETIC_RESEARCH, EvidenceTier.OFFLINE_FIXTURE})
        record["observation"] = observation.to_dict()
        record["provenance"] = safe_provenance
        record["schema_version"] = SCHEMA_VERSION
        record["dataset_id"] = dataset_id
        record["release_id"] = release_id
        record["subject_split_id"] = _split_id(observation.subject_id)
        built.append(record)
    return tuple(built)


def write_jsonl(records: tuple[dict[str, Any], ...], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a provenance-validated PMCC JSONL dataset.")
    parser.add_argument("--input", required=True, type=Path, help="source JSONL file or directory")
    parser.add_argument("--output", required=True, type=Path, help="dataset JSONL output")
    parser.add_argument("--evidence-tier", required=True, choices=[tier.value for tier in EvidenceTier])
    args = parser.parse_args(argv)
    write_jsonl(build_dataset(read_jsonl(args.input), EvidenceTier(args.evidence_tier)), args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(f"dataset build failed: {error}")
