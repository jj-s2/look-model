"""Evaluate fall-trained UR Fall checkpoints on held-out ADL windows."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from risk.urfall_adl_evaluation import evaluate_urfall_adl_false_alerts
def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True); parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--checkpoints-dir",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    args=parser.parse_args()
    print(json.dumps(evaluate_urfall_adl_false_alerts(args.manifest,args.data_root,args.checkpoints_dir,args.output_dir,device=args.device),ensure_ascii=False,indent=2,sort_keys=True))
    return 0
if __name__=="__main__": raise SystemExit(main())
