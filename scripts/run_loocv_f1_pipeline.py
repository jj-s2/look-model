"""Train and evaluate PA-DTSF across a LOOCV fold manifest.

For each fold the script trains a model on the fold's train split, runs the
inner threshold search on the fold's validation split, evaluates on the fold's
test split, and checks the F1 promotion gate.  Results are written per-fold
and a JSON summary aggregates performance across folds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run(
    cmd: list[str], cwd: Path, description: str, check: bool = True
) -> subprocess.CompletedProcess:
    print(f"[{description}] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"FAILED: {description}", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        if check:
            raise RuntimeError(f"{description} failed")
    return result


def run_loocv_pipeline(
    *,
    config: Path,
    dataset_lock: Path,
    split_manifest: Path,
    data_root: Path,
    baseline: Path,
    output_dir: Path,
    release_id: str,
    epochs: int | None = None,
    lr_scheduler: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    folds = sorted(manifest.get("folds", {}).keys())
    if not folds:
        raise ValueError("split_manifest contains no folds")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_fold_results: dict[str, dict[str, Any]] = {}

    for fold in folds:
        fold_train_dir = output_dir / f"fold-{fold}" / "train"
        fold_exp_dir = output_dir / f"fold-{fold}" / "experiment"

        train_cmd = [
            sys.executable,
            "scripts/train_phase_model.py",
            "--config", str(config),
            "--dataset-lock", str(dataset_lock),
            "--split-manifest", str(split_manifest),
            "--data-root", str(data_root),
            "--output", str(fold_train_dir),
            "--release-id", f"{release_id}-{fold}",
            "--fold", fold,
        ]
        if epochs is not None:
            train_cmd.extend(["--epochs", str(epochs)])
        if lr_scheduler is not None:
            train_cmd.extend(["--lr-scheduler", lr_scheduler])
        if not dry_run:
            _run(train_cmd, PROJECT_ROOT, f"train-{fold}")

        exp_cmd = [
            sys.executable,
            "scripts/run_phase_f1_experiment.py",
            "--checkpoint", str(fold_train_dir / "checkpoint.pt"),
            "--dataset-lock", str(dataset_lock),
            "--split-manifest", str(split_manifest),
            "--data-root", str(data_root),
            "--baseline", str(baseline),
            "--output", str(fold_exp_dir),
            "--release-id", f"{release_id}-{fold}",
            "--fold", fold,
        ]
        if not dry_run:
            _run(exp_cmd, PROJECT_ROOT, f"experiment-{fold}")

        experiment_path = fold_exp_dir / "experiment.json"
        if experiment_path.exists():
            per_fold_results[fold] = json.loads(experiment_path.read_text(encoding="utf-8"))

    summary = {
        "release_id": release_id,
        "folds": folds,
        "per_fold": per_fold_results,
        "aggregate": {
            "mean_confirmation_f1": float(
                sum(r["candidate"]["confirmation_f1"] for r in per_fold_results.values()) / len(per_fold_results)
            ) if per_fold_results else None,
            "mean_inner_f1": float(
                sum(r["candidate"]["inner_mean_f1"] for r in per_fold_results.values()) / len(per_fold_results)
            ) if per_fold_results else None,
            "all_promoted": all(r.get("promoted", False) for r in per_fold_results.values()) if per_fold_results else False,
        },
    }
    (output_dir / "loocv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-lock", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr-scheduler", type=str)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = run_loocv_pipeline(
        config=args.config,
        dataset_lock=args.dataset_lock,
        split_manifest=args.split_manifest,
        data_root=args.data_root,
        baseline=args.baseline,
        output_dir=args.output,
        release_id=args.release_id,
        epochs=args.epochs,
        lr_scheduler=args.lr_scheduler,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
