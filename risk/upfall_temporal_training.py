"""Small temporal classifier and sample loader for UP-Fall preimpact windows."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def _torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for UP-Fall temporal training") from error
    return torch, nn


def load_upfall_samples(manifest_path: Path, data_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load manifest-referenced finite 33-joint pose windows in stable sample order."""
    try:
        records = [json.loads(line) for line in Path(manifest_path).read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read UP-Fall manifest: {error}") from error
    if not records:
        raise ValueError("UP-Fall manifest contains no samples")
    poses: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[str] = []
    for record in sorted(records, key=lambda item: str(item.get("sample_id", ""))):
        if not isinstance(record, dict):
            raise ValueError("UP-Fall manifest rows must be objects")
        sample_id = record.get("sample_id")
        relative = record.get("window_path")
        label = record.get("label")
        subject = record.get("subject_id")
        if not isinstance(sample_id, str) or not isinstance(relative, str) or type(label) is not int or label not in {0, 1}:
            raise ValueError("UP-Fall manifest row has invalid sample metadata")
        if not isinstance(subject, str) or not subject:
            raise ValueError("UP-Fall manifest row has invalid subject_id")
        path = (Path(data_root) / relative).resolve()
        root = Path(data_root).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"UP-Fall sample path is unavailable: {relative}")
        try:
            with np.load(path) as artifact:
                pose = np.asarray(artifact["pose"], dtype=np.float32)
                stored_label = int(artifact["label"])
        except (KeyError, OSError, ValueError) as error:
            raise ValueError(f"cannot load UP-Fall sample {relative}: {error}") from error
        if stored_label != label:
            raise ValueError(f"UP-Fall sample label mismatch: {relative}")
        if pose.ndim != 3 or pose.shape[1:] != (33, 3) or len(pose) == 0 or not np.isfinite(pose).all():
            raise ValueError(f"UP-Fall sample has invalid pose shape or coordinates: {relative}")
        poses.append(pose)
        labels.append(label)
        subjects.append(subject)
    if len({pose.shape for pose in poses}) != 1:
        raise ValueError("UP-Fall samples must share one pose shape")
    return np.stack(poses), np.asarray(labels, dtype=np.int64), subjects


def normalize_upfall_pose(pose: object) -> object:
    """Return relative position, root translation, and velocity features per joint."""
    torch, _ = _torch()
    if not isinstance(pose, torch.Tensor) or pose.ndim != 4 or pose.shape[2:] != (33, 3):
        raise ValueError("pose must be a torch tensor with shape (N, T, 33, 3)")
    if pose.shape[0] == 0 or pose.shape[1] == 0 or not torch.isfinite(pose).all():
        raise ValueError("pose must be non-empty and finite")
    pose = pose.float()
    centroid = pose.mean(dim=2, keepdim=True)
    relative = pose - centroid
    scale = relative.square().sum(dim=-1).sqrt().mean(dim=(1, 2), keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    relative = relative / scale
    translation = (centroid - centroid[:, :1]) / scale
    translation = translation.expand_as(relative)
    velocity = torch.diff(relative, dim=1, prepend=relative[:, :1])
    return torch.cat((relative, translation, velocity), dim=-1)


class UpFallTemporalTCN:
    """Two-block temporal convolutional network for 33-joint pose windows."""

    def __init__(self, *, joints: int = 33, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        _, nn = _torch()
        if joints != 33:
            raise ValueError("UP-Fall temporal model requires 33 joints")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        groups = 4 if hidden_dim % 4 == 0 else 1
        channels = joints * 9
        self.network = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_dim, 1)

    def __call__(self, pose: object) -> object:
        features = normalize_upfall_pose(pose)
        sequence = features.reshape(features.shape[0], features.shape[1], -1).transpose(1, 2)
        return self.head(self.pool(self.network(sequence)).squeeze(-1)).squeeze(-1)

    def parameters(self):
        return (parameter for module in (self.network, self.pool, self.head) for parameter in module.parameters())

    def to(self, device: str):
        for module in (self.network, self.pool, self.head):
            module.to(device)
        return self

    def train(self, mode: bool = True):
        for module in (self.network, self.pool, self.head):
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self) -> dict[str, object]:
        return {"network": self.network.state_dict(), "head": self.head.state_dict()}


def train_upfall_loso(
    manifest_path: Path, data_root: Path, output_dir: Path, *, epochs: int = 80,
    learning_rate: float = 1e-3, hidden_dim: int = 32, dropout: float = 0.2,
    device: str = "auto", random_seed: int = 42,
) -> dict[str, object]:
    """Train a fresh temporal model for each held-out UP-Fall subject."""
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    torch, nn = _torch()
    poses, labels, subjects = load_upfall_samples(manifest_path, data_root)
    sample_ids = _manifest_sample_ids(manifest_path)
    if len(sample_ids) != len(poses):
        raise ValueError("UP-Fall manifest sample count changed while loading")
    device_name = _resolve_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    subject_values = sorted(set(subjects))
    subject_array = np.asarray(subjects)
    for fold_index, held_out_subject in enumerate(subject_values):
        train_index = np.flatnonzero(subject_array != held_out_subject)
        valid_index = np.flatnonzero(subject_array == held_out_subject)
        if set(labels[train_index].tolist()) != {0, 1}:
            raise ValueError(f"training fold for {held_out_subject} requires both labels")
        _seed_torch(random_seed + fold_index)
        model = UpFallTemporalTCN(hidden_dim=hidden_dim, dropout=dropout).to(device_name)
        train_pose = torch.as_tensor(poses[train_index], dtype=torch.float32, device=device_name)
        train_labels = torch.as_tensor(labels[train_index], dtype=torch.float32, device=device_name)
        valid_pose = torch.as_tensor(poses[valid_index], dtype=torch.float32, device=device_name)
        positive = float(labels[train_index].sum())
        negative = float(len(train_index) - positive)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device_name))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(train_pose), train_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(valid_pose)).detach().cpu().numpy()
        valid_labels = labels[valid_index]
        metrics = _binary_metrics(valid_labels, probabilities >= 0.5)
        fold = {
            "held_out_subject": held_out_subject,
            "train_subjects": sorted(set(subject_array[train_index].tolist())),
            "sample_count": int(len(valid_index)),
            **metrics,
        }
        folds.append(fold)
        for sample_index, probability in zip(valid_index.tolist(), probabilities.tolist(), strict=True):
            predictions.append({
                "sample_id": sample_ids[sample_index],
                "subject_id": subjects[sample_index],
                "label": int(labels[sample_index]),
                "probability": float(probability),
                "prediction": int(probability >= 0.5),
                "held_out_subject": held_out_subject,
            })
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": {"joints": 33, "hidden_dim": hidden_dim, "dropout": dropout},
                "held_out_subject": held_out_subject,
            },
            output_dir / f"{held_out_subject}.pt",
        )
    aggregate = {
        metric: float(sum(float(fold[metric]) for fold in folds) / len(folds))
        for metric in ("precision", "recall", "f1")
    }
    report: dict[str, object] = {
        "validation": "LeaveOneSubjectOut",
        "folds": folds,
        "metrics": aggregate,
        "sample_count": int(len(labels)),
        "subject_count": len(subject_values),
        "threshold": 0.5,
        "promoted": False,
        "experimental_only": True,
        "not_for_clinical_performance": True,
        "window_unit": "frames",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    model_card = {
        **report,
        "model_type": "UpFallTemporalTCN",
        "model_config": {"joints": 33, "hidden_dim": hidden_dim, "dropout": dropout},
        "epochs": epochs,
        "learning_rate": float(learning_rate),
        "source_manifest": str(manifest_path),
    }
    (output_dir / "model_card.json").write_text(
        json.dumps(model_card, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    return report


def _manifest_sample_ids(manifest_path: Path) -> list[str]:
    try:
        records = [json.loads(line) for line in Path(manifest_path).read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read UP-Fall manifest: {error}") from error
    sample_ids = []
    for record in sorted(records, key=lambda item: str(item.get("sample_id", ""))):
        if not isinstance(record, dict) or not isinstance(record.get("sample_id"), str):
            raise ValueError("UP-Fall manifest row has invalid sample_id")
        sample_ids.append(record["sample_id"])
    return sample_ids


def _resolve_device(request: str) -> str:
    torch, _ = _torch()
    if request == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if request not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if request == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return request


def _seed_torch(seed: int) -> None:
    torch, _ = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
