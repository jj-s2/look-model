"""Leakage-safe cross-domain fine-tuning on fall windows and held-out ADL sequences."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from risk.upfall_temporal_training import UpFallTemporalTCN, _binary_metrics, _resolve_device, _seed_torch
from risk.urfall_temporal_training import _load_samples


def train_urfall_crossdomain_experiment(
    fall_manifest: Path, fall_root: Path, adl_manifest: Path, adl_root: Path, output_dir: Path,
    *, epochs: int = 80, learning_rate: float = 1e-3, hidden_dim: int = 32, dropout: float = 0.2,
    device: str = "auto", random_seed: int = 42, adl_folds: int = 5,
) -> dict[str, object]:
    """Train fall LOSO folds while holding one deterministic ADL sequence group out."""
    if type(epochs) is not int or epochs <= 0 or type(adl_folds) is not int or adl_folds < 2:
        raise ValueError("epochs must be positive and adl_folds must be at least two")
    fall_poses, fall_labels, fall_sequences, _ = _load_samples(fall_manifest, fall_root)
    adl_poses, adl_sequences = _load_adl(adl_manifest, adl_root)
    values = sorted(set(fall_sequences))
    if len(values) < 2:
        raise ValueError("at least two fall sequences are required")
    adl_values = sorted(set(adl_sequences))
    groups = {sequence: index % adl_folds for index, sequence in enumerate(adl_values)}
    torch, nn = _torch()
    device_name = _resolve_device(device)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fall_array, adl_array = np.asarray(fall_sequences), np.asarray(adl_sequences)
    folds: list[dict[str, object]] = []
    for index, held_out in enumerate(values):
        holdout_group = index % adl_folds
        fall_train = np.flatnonzero(fall_array != held_out); fall_valid = np.flatnonzero(fall_array == held_out)
        adl_train = np.asarray([groups[sequence] != holdout_group for sequence in adl_sequences])
        adl_valid = ~adl_train
        if not adl_train.any() or not adl_valid.any() or set(fall_labels[fall_train].tolist()) != {0, 1}:
            raise ValueError("each cross-domain fold requires fall labels and ADL train/holdout sequences")
        _seed_torch(random_seed + index)
        model = UpFallTemporalTCN(hidden_dim=hidden_dim, dropout=dropout).to(device_name)
        train_pose = np.concatenate((fall_poses[fall_train], adl_poses[adl_train]))
        train_labels = np.concatenate((fall_labels[fall_train], np.zeros(int(adl_train.sum()), dtype=np.int64)))
        x = torch.as_tensor(train_pose, dtype=torch.float32, device=device_name)
        y = torch.as_tensor(train_labels, dtype=torch.float32, device=device_name)
        positive, negative = float(train_labels.sum()), float(len(train_labels) - train_labels.sum())
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device_name))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
        for _ in range(epochs):
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            fall_probability = torch.sigmoid(model(torch.as_tensor(fall_poses[fall_valid], dtype=torch.float32, device=device_name))).cpu().numpy()
            adl_probability = torch.sigmoid(model(torch.as_tensor(adl_poses[adl_valid], dtype=torch.float32, device=device_name))).cpu().numpy()
        metrics = _binary_metrics(fall_labels[fall_valid], fall_probability >= 0.5)
        folds.append({
            "held_out_fall_sequence": held_out, "train_fall_sequences": sorted(set(fall_array[fall_train].tolist())),
            "adl_holdout_sequences": sorted({sequence for sequence in adl_values if groups[sequence] == holdout_group}),
            "fall_metrics": metrics, "adl_false_positive_rate": float((adl_probability >= 0.5).mean()),
        })
    report: dict[str, object] = {
        "external_experiment": True, "promoted": False, "validation": "FallLOSO+ADLGroupedHoldout",
        "fall_sequence_count": len(values), "adl_holdout_sequence_count": len(adl_values),
        "metrics": {key: float(sum(fold["fall_metrics"][key] for fold in folds) / len(folds)) for key in ("precision", "recall", "f1")},
        "adl_false_positive_rate": float(sum(fold["adl_false_positive_rate"] for fold in folds) / len(folds)),
        "folds": folds,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def _load_adl(manifest_path: Path, data_root: Path) -> tuple[np.ndarray, list[str]]:
    try:
        rows = [json.loads(line) for line in Path(manifest_path).read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read ADL manifest: {error}") from error
    root = Path(data_root).resolve(); poses: list[np.ndarray] = []; sequences: list[str] = []
    for row in sorted(rows, key=lambda item: str(item.get("sample_id", ""))):
        if not isinstance(row, dict) or row.get("label") != 0 or not isinstance(row.get("sequence_id"), str) or not isinstance(row.get("pose_path"), str):
            raise ValueError("ADL manifest requires label-0 sequence rows")
        path = (root / row["pose_path"]).resolve()
        if root not in path.parents or not path.is_file(): raise ValueError("ADL pose sample unavailable")
        with np.load(path) as artifact: pose, label = np.asarray(artifact["pose"], dtype=np.float32), int(artifact["label"])
        if label != 0 or pose.ndim != 3 or pose.shape[1:] != (33, 3) or not np.isfinite(pose).all(): raise ValueError("invalid ADL pose")
        poses.append(pose); sequences.append(row["sequence_id"])
    if not poses or len({pose.shape for pose in poses}) != 1: raise ValueError("ADL samples need one non-empty shape")
    return np.stack(poses), sequences


def _torch() -> tuple[Any, Any]:
    try:
        import torch, torch.nn as nn
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required") from error
    return torch, nn
