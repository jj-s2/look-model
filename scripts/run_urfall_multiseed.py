"""Run deterministic multi-seed UR Fall cross-domain validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk.urfall_multiseed import run_urfall_multiseed_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fall-manifest", "fall-root", "adl-manifest", "adl-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--recall-floor", type=float, default=0.8)
    args = parser.parse_args()
    report = run_urfall_multiseed_experiment(
        fall_manifest=args.fall_manifest,
        fall_root=args.fall_root,
        adl_manifest=args.adl_manifest,
        adl_root=args.adl_root,
        output_dir=args.output_dir,
        seeds=args.seeds,
        epochs=args.epochs,
        device=args.device,
        recall_floor=args.recall_floor,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
