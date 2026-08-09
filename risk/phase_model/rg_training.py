"""Deterministic, subject-safe RG-PCNet training from frozen pose caches."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from typing import Mapping

import numpy as np

from .rg_losses import RGPCLossTargets, compute_rgpc_loss
from .rg_pcnet import RGPCNet
from .training_data import RGPCDataset, collate_rgpc_samples


def seed_everything(seed: int) -> None:
    """Seed every training RNG and select deterministic torch kernels."""
    import torch

    # Must be present before the first CUDA GEMM for deterministic cuBLAS kernels.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, workers: int):
    """Build a reproducible RG-PCNet loader with per-worker NumPy/Python seeds."""
    import torch
    from torch.utils.data import DataLoader

    def worker_init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=generator, worker_init_fn=worker_init, collate_fn=collate_rgpc_samples)


class EarlyStopping:
    """Track a deep-copied best model state and stop after non-improvements."""

    def __init__(self, patience: int, mode: str = "max") -> None:
        if patience < 0 or mode not in {"max", "min"}:
            raise ValueError("patience must be non-negative and mode must be 'max' or 'min'")
        self.patience, self.mode = patience, mode
        self.best_score = None
        self.best_epoch = None
        self.bad_epochs = 0
        self.should_stop = False
        self.best_model_state = None

    def update(self, score: float, *, epoch: int, model=None) -> bool:
        improved = self.best_score is None or (score > self.best_score if self.mode == "max" else score < self.best_score)
        if improved:
            self.best_score, self.best_epoch, self.bad_epochs = float(score), int(epoch), 0
            self.should_stop = False
            if model is not None:
                self.best_model_state = copy.deepcopy(model.state_dict())
            return True
        self.bad_epochs += 1
        self.should_stop = self.bad_epochs >= self.patience
        return False

    def restore(self, model) -> None:
        if self.best_model_state is None:
            raise RuntimeError("no best model state has been recorded")
        model.load_state_dict(self.best_model_state)


def _load_config(path: Path | None):
    if path is None:
        from configs.skeleton import rg_pcnet_v1 as config
        return config
    spec = importlib.util.spec_from_file_location("rg_pcnet_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load config: {path}")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def _load_provenance(dataset_lock: Path, split_manifest: Path, release_id: str) -> tuple[dict, dict]:
    lock = json.loads(dataset_lock.read_text(encoding="utf-8"))
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or not isinstance(split, dict):
        raise ValueError("dataset lock and split manifest must be JSON objects")
    if split.get("release_id") not in (None, release_id):
        raise ValueError("split_manifest release_id does not match")
    return lock, split


def _partition_ids(split: Mapping[str, object], name: str) -> list[str]:
    partitions = split.get("partitions", split)
    values = partitions.get(name, []) if isinstance(partitions, Mapping) else []
    return [str(value) for value in values]


def _select_clips(lock: Mapping[str, object], ids: list[str]) -> list[dict]:
    by_id = {str(clip.get("clip_id")): dict(clip) for clip in lock.get("clips", []) if isinstance(clip, Mapping)}
    missing = [clip_id for clip_id in ids if clip_id not in by_id]
    if missing:
        raise ValueError(f"split references clips missing from dataset lock: {missing}")
    return [by_id[clip_id] for clip_id in ids]


def _preflight(dataset: RGPCDataset) -> None:
    for index in range(len(dataset)):
        sample = dataset[index]
        if not bool(np.asarray(sample.valid_mask).any()):
            raise ValueError(f"clip_id {sample.clip_id} has no valid frames")


def _to_device(batch, device):
    return type(batch)(*(value.to(device) if hasattr(value, "to") else value for value in batch.__dict__.values()))


def _targets(clean, corrupt):
    import torch
    return RGPCLossTargets(clean.fall_target, clean.phase_target, clean.phase_mask,
                           corrupt.reliability_target, clean.valid_mask,
                           torch.zeros(clean.valid_mask.shape, dtype=torch.float32, device=clean.valid_mask.device))


def _macro_f1(labels: list[int], predictions: list[int]) -> float:
    scores = []
    for label in (0, 1):
        tp = sum(actual == label and predicted == label for actual, predicted in zip(labels, predictions))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(labels, predictions))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(labels, predictions))
        scores.append(0.0 if 2 * tp + fp + fn == 0 else (2 * tp) / (2 * tp + fp + fn))
    return float(sum(scores) / len(scores))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parents[2]).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_rg_pcnet(*, dataset_lock: Path, split_manifest: Path, data_root: Path, output_dir: Path, release_id: str, config_path: Path | None = None, device: str = "auto") -> dict[str, object]:
    """Train RG-PCNet using clean supervision and clip-aligned corrupt consistency views."""
    import torch

    dataset_lock, split_manifest, data_root, output_dir = map(Path, (dataset_lock, split_manifest, data_root, output_dir))
    if not dataset_lock.exists() or not split_manifest.exists():
        raise FileNotFoundError("dataset lock and split manifest must exist")
    config = _load_config(config_path)
    seed = int(config.SEED)
    seed_everything(seed)
    if device == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device in {"cpu", "cuda"}:
        resolved_device = torch.device(device)
    else:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    lock, split = _load_provenance(dataset_lock, split_manifest, release_id)
    train_clips, validation_clips = _select_clips(lock, _partition_ids(split, "train")), _select_clips(lock, _partition_ids(split, "validation"))
    if not train_clips or not validation_clips:
        raise ValueError("train and validation partitions must both contain clips")
    train_subjects = {str(clip["subject_id"]) for clip in train_clips}
    validation_subjects = {str(clip["subject_id"]) for clip in validation_clips}
    overlap = train_subjects & validation_subjects
    if overlap:
        raise ValueError(f"train and validation subjects overlap: {sorted(overlap)}")

    clean_train = RGPCDataset(train_clips, data_root, seed=seed)
    corrupt_train = RGPCDataset(train_clips, data_root, seed=seed, corruption_probability=float(config.CORRUPTION_PROBABILITY), corruption_severity=float(getattr(config, "CORRUPTION_SEVERITY", 0.5)))
    validation = RGPCDataset(validation_clips, data_root, seed=seed)
    _preflight(clean_train)
    _preflight(validation)
    model = RGPCNet(input_dim=int(config.INPUT_DIM), hidden_dim=int(config.HIDDEN_DIM), dropout=float(config.DROPOUT)).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.LEARNING_RATE), weight_decay=float(config.WEIGHT_DECAY))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
    stopping = EarlyStopping(patience=int(config.PATIENCE), mode="max")

    for epoch in range(int(config.EPOCHS)):
        clean_train.set_epoch(epoch)
        corrupt_train.set_epoch(epoch)
        _preflight(corrupt_train)
        clean_loader = make_loader(clean_train, int(config.BATCH_SIZE), True, seed + epoch, int(config.NUM_WORKERS))
        corrupt_loader = make_loader(corrupt_train, int(config.BATCH_SIZE), True, seed + epoch, int(config.NUM_WORKERS))
        model.train()
        for clean_batch, corrupt_batch in zip(clean_loader, corrupt_loader):
            if clean_batch.clip_ids != corrupt_batch.clip_ids:
                raise RuntimeError("clean and corrupted loader clip IDs are not aligned")
            clean_batch, corrupt_batch = _to_device(clean_batch, resolved_device), _to_device(corrupt_batch, resolved_device)
            optimizer.zero_grad(set_to_none=True)
            clean_output = model(clean_batch.features, clean_batch.valid_mask)
            corrupted_output = model(corrupt_batch.features, corrupt_batch.valid_mask)
            loss = compute_rgpc_loss(clean_output, _targets(clean_batch, corrupt_batch), corrupted_output=corrupted_output)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.GRAD_CLIP_NORM))
            optimizer.step()

        model.eval()
        labels, predictions = [], []
        with torch.no_grad():
            for batch in make_loader(validation, int(config.BATCH_SIZE), False, seed, int(config.NUM_WORKERS)):
                batch = _to_device(batch, resolved_device)
                output = model(batch.features, batch.valid_mask)
                labels.extend(batch.fall_target.long().cpu().tolist())
                predictions.extend((output.window_fall_logit.sigmoid() >= 0.5).long().cpu().tolist())
        score = _macro_f1(labels, predictions)
        scheduler.step(score)
        stopping.update(score, epoch=epoch, model=model)
        if stopping.should_stop:
            break

    stopping.restore(model)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        checkpoint = {"release_id": release_id, "best_epoch": stopping.best_epoch, "best_score": stopping.best_score, "model_state_dict": model.state_dict(), "model_config": {"input_dim": int(config.INPUT_DIM), "hidden_dim": int(config.HIDDEN_DIM), "dropout": float(config.DROPOUT)}}
        checkpoint_path = temporary / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        shutil.copyfile(dataset_lock, temporary / "dataset_lock.json")
        shutil.copyfile(split_manifest, temporary / "split_manifest.json")
        checkpoint_hash = _sha256(checkpoint_path)
        metrics = {"batch_size": int(config.BATCH_SIZE), "best_epoch": stopping.best_epoch, "best_validation_macro_f1": stopping.best_score, "checkpoint_sha256": checkpoint_hash, "device": str(resolved_device), "epochs_completed": epoch + 1, "promoted": False, "release_id": release_id, "seed": seed, "validation_subjects": sorted(validation_subjects)}
        manifest = {"cuda": torch.version.cuda, "device": str(resolved_device), "git_commit": _git_commit(), "input_manifest_hashes": {"dataset_lock": _sha256(dataset_lock), "split_manifest": _sha256(split_manifest)}, "release_id": release_id, "seed": seed, "torch": torch.__version__}
        (temporary / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metrics


__all__ = ["EarlyStopping", "make_loader", "seed_everything", "train_rg_pcnet"]
