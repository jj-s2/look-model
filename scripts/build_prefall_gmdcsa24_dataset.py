"""Build leakage-safe GMDCSA24 pre-fall training CSV from pose caches."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk.prefall_dataset import FEATURE_COLUMNS, build_prefall_rows, duration_by_media_from_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--horizon-sec", type=float, default=3.0)
    parser.add_argument("--guard-sec", type=float, default=0.5)
    args = parser.parse_args()
    try:
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        clips = lock["clips"]
        with args.metadata_csv.open(newline="", encoding="utf-8") as stream:
            durations = duration_by_media_from_metadata(csv.DictReader(stream))
    except (OSError, KeyError, json.JSONDecodeError) as error:
        parser.error(f"cannot load dataset inputs: {error}")
    rows, audit = build_prefall_rows(
        clips, durations, args.cache_dir, horizon_sec=args.horizon_sec, guard_sec=args.guard_sec,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("subject_id", "label", *FEATURE_COLUMNS))
        writer.writeheader()
        writer.writerows({name: row[name] for name in ("subject_id", "label", *FEATURE_COLUMNS)} for row in rows)
    audit = {**audit, "output_rows": len(rows), "output_csv": str(args.output_csv)}
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
