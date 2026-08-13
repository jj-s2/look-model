"""External false-alert evaluation of fall-trained TCN checkpoints on UR Fall ADL."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from risk.upfall_temporal_training import UpFallTemporalTCN, _resolve_device


def evaluate_urfall_adl_false_alerts(manifest_path: Path, data_root: Path, checkpoints_dir: Path, output_dir: Path, *, device: str = "auto") -> dict[str, object]:
    """Score ADL-only pose windows with every saved fall-sequence checkpoint."""
    torch = _torch()
    poses, sample_ids = _load_adl_samples(manifest_path, data_root)
    checkpoints = sorted(Path(checkpoints_dir).glob("*.pt"))
    if not checkpoints:
        raise ValueError("no UR Fall checkpoints found")
    device_name = _resolve_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for checkpoint_path in checkpoints:
        artifact = torch.load(checkpoint_path, map_location=device_name, weights_only=False)
        config = artifact.get("model_config") if isinstance(artifact, dict) else None
        state = artifact.get("state_dict") if isinstance(artifact, dict) else None
        held_out = artifact.get("held_out_sequence") if isinstance(artifact, dict) else None
        if not isinstance(config, dict) or not isinstance(state, dict) or not isinstance(held_out, str):
            raise ValueError(f"invalid checkpoint: {checkpoint_path.name}")
        model = UpFallTemporalTCN(hidden_dim=int(config["hidden_dim"]), dropout=float(config["dropout"])).to(device_name)
        model.network.load_state_dict(state["network"])
        model.head.load_state_dict(state["head"])
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(torch.as_tensor(poses, dtype=torch.float32, device=device_name))).cpu().numpy()
        flags = probabilities >= 0.5
        folds.append({"held_out_fall_sequence": held_out, "sample_count": int(len(poses)), "false_positive_rate": float(flags.mean())})
        predictions.extend({"sample_id": sample_id, "held_out_fall_sequence": held_out, "probability": float(probability), "false_alert": bool(probability >= 0.5)} for sample_id, probability in zip(sample_ids, probabilities.tolist(), strict=True))
    report: dict[str, object] = {
        "external_adl_evaluation": True, "promoted": False, "sample_count": int(len(poses)),
        "checkpoint_count": len(checkpoints), "false_positive_rate": float(sum(fold["false_positive_rate"] for fold in folds) / len(folds)),
        "folds": folds,
    }
    (output_dir / "adl_false_alert_metrics.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output_dir / "adl_predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions), encoding="utf-8")
    return report


def _load_adl_samples(manifest_path: Path, data_root: Path) -> tuple[np.ndarray, list[str]]:
    try:
        rows = [json.loads(line) for line in Path(manifest_path).read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read ADL pose manifest: {error}") from error
    root = Path(data_root).resolve()
    poses: list[np.ndarray] = []
    sample_ids: list[str] = []
    for row in sorted(rows, key=lambda item: str(item.get("sample_id", ""))):
        if not isinstance(row, dict) or type(row.get("label")) is not int or row["label"] != 0:
            raise ValueError("ADL manifest must contain only label-0 objects")
        sample_id, relative = row.get("sample_id"), row.get("pose_path")
        if not isinstance(sample_id, str) or not isinstance(relative, str):
            raise ValueError("ADL manifest row has invalid sample metadata")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"ADL pose sample unavailable: {relative}")
        with np.load(path) as artifact:
            pose, label = np.asarray(artifact["pose"], dtype=np.float32), int(artifact["label"])
        if label != 0 or pose.ndim != 3 or pose.shape[1:] != (33, 3) or not np.isfinite(pose).all():
            raise ValueError(f"invalid ADL pose sample: {relative}")
        poses.append(pose); sample_ids.append(sample_id)
    if not poses or len({pose.shape for pose in poses}) != 1:
        raise ValueError("ADL pose samples must be non-empty with one shape")
    return np.stack(poses), sample_ids


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for UR Fall ADL evaluation") from error
    return torch
