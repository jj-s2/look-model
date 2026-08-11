"""Build audited frame-window prefall samples from UP-Fall 3D skeleton CSV files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk.upfall_prefall_dataset import build_upfall_prefall_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="directory containing UP-Fall CSV files")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for NPZ windows and audit files")
    parser.add_argument("--pre-frames", type=int, default=10, help="number of source frames per sample")
    parser.add_argument("--guard-frames", type=int, default=5, help="frames excluded before first impact")
    args = parser.parse_args()
    if not args.source_dir.is_dir():
        parser.error(f"--source-dir is not a directory: {args.source_dir}")
    try:
        summary = build_upfall_prefall_dataset(
            args.source_dir, args.output_dir, pre_frames=args.pre_frames, guard_frames=args.guard_frames,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
