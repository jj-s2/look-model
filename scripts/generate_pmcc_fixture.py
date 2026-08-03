"""Create deterministic, offline PMCC research fixtures as JSONL.

The generated rows are deliberately synthetic research examples.  They are
never device observations, clinical evidence, or release-ready artifacts.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk.pmcc.schema import DailyObservation


SCENARIOS = (
    ("normal", None, {"sleep_duration_minutes": 450.0, "activity_count": 1800.0, "trunk_sway": 0.25, "heart_rate": 70.0}),
    ("sleep_activity_gait_cascade", 1, {"sleep_duration_minutes": 280.0, "activity_count": 620.0, "trunk_sway": 0.72, "heart_rate": 84.0}),
    ("missing_physiology", None, {"sleep_duration_minutes": 390.0, "activity_count": 1100.0, "trunk_sway": 0.42, "heart_rate": None}),
    ("timestamp_shift", 3, {"sleep_duration_minutes": 330.0, "activity_count": 760.0, "trunk_sway": 0.61, "heart_rate": 79.0}),
)


def build_fixture_records(seed: int) -> tuple[dict[str, Any], ...]:
    """Return a deterministic set of four labeled, non-clinical examples."""
    generator = random.Random(seed)
    base = datetime(2026, 1, 15, tzinfo=timezone.utc)
    records: list[dict[str, Any]] = []
    for index, (scenario, event_day, features) in enumerate(SCENARIOS):
        jitter = generator.uniform(-0.5, 0.5)
        adjusted = {name: None if value is None else round(value + jitter, 6) for name, value in features.items()}
        availability = {name: value is not None for name, value in adjusted.items()}
        quality = {name: 0.0 if value is None else 0.9 for name, value in adjusted.items()}
        observed_at = base + timedelta(days=index, hours=2 if scenario == "timestamp_shift" else 0)
        provenance = {
            "evidence_tier": "synthetic_research",
            "promoted": False,
            "source_kind": "offline_fixture_generator",
            "non_clinical": True,
            "seed": seed,
        }
        observation = DailyObservation(
            subject_id=f"fixture-subject-{index + 1}",
            observed_at=observed_at,
            features=adjusted,
            quality=quality,
            availability=availability,
            provenance=provenance,
        )
        records.append({
            "schema_version": "pmcc.fixture.v1",
            "record_type": "pmcc_training_example",
            "scenario": scenario,
            "observation": observation.to_dict(),
            "label": {"event_day": event_day, "censor_day": 7},
            "provenance": provenance,
        })
    return tuple(records)


def write_jsonl(records: tuple[dict[str, Any], ...], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic offline PMCC research fixtures.")
    parser.add_argument("--output", required=True, type=Path, help="JSONL output path")
    parser.add_argument("--seed", default=42, type=int, help="Deterministic random seed")
    args = parser.parse_args(argv)
    write_jsonl(build_fixture_records(args.seed), args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(f"fixture generation failed: {error}")
