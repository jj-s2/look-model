"""Build audited UR Fall RGB pre-impact sample manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk.urfall_prefall_dataset import build_urfall_prefall_manifest

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pre-frames", type=int, default=10)
    parser.add_argument("--guard-frames", type=int, default=5)
    args = parser.parse_args()
    try:
        summary = build_urfall_prefall_manifest(args.source_dir, args.rgb_dir, args.output_dir, pre_frames=args.pre_frames, guard_frames=args.guard_frames)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

