"""Extract MediaPipe pose windows from an audited UR Fall manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk.urfall_pose_extraction import MediaPipePoseDetector, extract_urfall_pose_windows

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--allow-single-class-sequences", action="store_true")
    args = parser.parse_args()
    detector = MediaPipePoseDetector(args.model_path)
    try:
        summary = extract_urfall_pose_windows(
            args.manifest, args.rgb_dir, args.output_dir, detector=detector,
            require_complete_pairs=not args.allow_single_class_sequences,
        )
    except ValueError as error:
        parser.error(str(error))
    finally:
        detector.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
