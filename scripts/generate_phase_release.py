"""Package metrics and frozen provenance into an auditable release directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk.phase_model.release import create_release_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    create_release_artifacts(
        release_id=args.release_id,
        metrics=json.loads(args.metrics.read_text(encoding="utf-8")),
        benchmark=json.loads(args.benchmark.read_text(encoding="utf-8")),
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
