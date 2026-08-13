"""Sequence-grouped external UR Fall experiment using the existing pose TCN."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from risk.upfall_temporal_training import UpFallTemporalTCN, _binary_metrics, _resolve_device, _seed_torch


def train_urfall_external_experiment(
    manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    *,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    hidden_dim: int = 32,
    dropout: float = 0.2,
    device: str = "auto",
    random_seed: int = 42,
) -> dict[str, object]:
    """Train leave-one-sequence-out folds; output is always external-only."""
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    poses, labels, sequences, sample_ids = _load_samples(manifest_path, data_root)
    values = sorted(set(sequences))
    if len(values) < 2:
        raise ValueError("UR Fall experiment requires at least two sequences")
    torch, nn = _torch()
    device_name = _resolve_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    sequence_array = np.asarray(sequences)
    folds: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for index, held_out in enumerate(values):
        train_index = np.flatnonzero(sequence_array != held_out)
        valid_index = np.flatnonzero(sequence_array == held_out)
        if set(labels[train_index].tolist()) != {0, 1} or set(labels[valid_index].tolist()) != {0, 1}:
            raise ValueError("each sequence fold requires a positive-negative pair")
        _seed_torch(random_seed + index)
        model = UpFallTemporalTCN(hidden_dim=hidden_dim, dropout=dropout).to(device_name)
        train_pose = torch.as_tensor(poses[train_index], dtype=torch.float32, device=device_name)
        train_labels = torch.as_tensor(labels[train_index], dtype=torch.float32, device=device_name)
        positive = float(labels[train_index].sum())
        negative = float(len(train_index) - positive)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device_name))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(train_pose), train_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(torch.as_tensor(poses[valid_index], dtype=torch.float32, device=device_name))).cpu().numpy()
        metrics = _binary_metrics(labels[valid_index], probabilities >= 0.5)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": {"joints": 33, "hidden_dim": hidden_dim, "dropout": dropout},
                "held_out_sequence": held_out,
            },
            checkpoints_dir / f"{held_out}.pt",
        )
        folds.append({"held_out_sequence": held_out, "train_sequences": sorted(set(sequence_array[train_index].tolist())), "sample_count": int(len(valid_index)), **metrics})
        for sample_index, probability in zip(valid_index.tolist(), probabilities.tolist(), strict=True):
            predictions.append({"sample_id": sample_ids[sample_index], "sequence_id": sequences[sample_index], "label": int(labels[sample_index]), "probability": float(probability), "prediction": int(probability >= 0.5), "held_out_sequence": held_out})
    metrics = {name: float(sum(float(fold[name]) for fold in folds) / len(folds)) for name in ("precision", "recall", "f1")}
    report: dict[str, object] = {
        "external_experiment": True, "promoted": False, "not_for_clinical_performance": True,
        "dataset": "UR Fall Detection Dataset", "split_unit": "sequence_id", "validation": "LeaveOneSequenceOut",
        "sequence_count": len(values), "sample_count": int(len(labels)), "threshold": 0.5, "metrics": metrics, "folds": folds,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output_dir / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions), encoding="utf-8")
    (output_dir / "model_card.json").write_text(json.dumps({**report, "model_type": "UpFallTemporalTCN", "model_config": {"joints": 33, "hidden_dim": hidden_dim, "dropout": dropout}, "epochs": epochs, "learning_rate": float(learning_rate), "source_manifest": str(manifest_path)}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def _load_samples(manifest_path: Path, data_root: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    try:
        rows = [json.loads(line) for line in Path(manifest_path).read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read UR Fall pose manifest: {error}") from error
    poses: list[np.ndarray] = []
    labels: list[int] = []
    sequences: list[str] = []
    sample_ids: list[str] = []
    root = Path(data_root).resolve()
    for row in sorted(rows, key=lambda item: str(item.get("sample_id", ""))):
        if not isinstance(row, dict):
            raise ValueError("UR Fall pose manifest rows must be objects")
        sample_id, sequence, label, relative = row.get("sample_id"), row.get("sequence_id"), row.get("label"), row.get("pose_path")
        if not isinstance(sample_id, str) or not isinstance(sequence, str) or type(label) is not int or label not in {0, 1} or not isinstance(relative, str):
            raise ValueError("UR Fall pose manifest row has invalid metadata")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"UR Fall pose sample unavailable: {relative}")
        with np.load(path) as artifact:
            pose, stored_label = np.asarray(artifact["pose"], dtype=np.float32), int(artifact["label"])
        if stored_label != label or pose.ndim != 3 or pose.shape[1:] != (33, 3) or not np.isfinite(pose).all():
            raise ValueError(f"UR Fall pose sample invalid: {relative}")
        poses.append(pose); labels.append(label); sequences.append(sequence); sample_ids.append(sample_id)
    if not poses or len({pose.shape for pose in poses}) != 1:
        raise ValueError("UR Fall pose samples must be non-empty and share one shape")
    for sequence in set(sequences):
        if {label for label, value in zip(labels, sequences, strict=True) if value == sequence} != {0, 1}:
            raise ValueError("each sequence must retain a positive-negative pair")
    return np.stack(poses), np.asarray(labels, dtype=np.int64), sequences, sample_ids


def _torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for UR Fall training") from error
    return torch, nn
