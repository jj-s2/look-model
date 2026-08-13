"""Build audited normal-activity UR Fall windows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from risk.urfall_adl_dataset import build_urfall_adl_manifest

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=10)
    args = parser.parse_args()
    try:
        result = build_urfall_adl_manifest(args.source_dir, args.rgb_dir, args.output_dir, window_frames=args.window_frames)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
