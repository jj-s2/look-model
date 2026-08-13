"""Deterministic, subject-safe RG-PCNet training from frozen pose caches."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from dataclasses import replace
from types import SimpleNamespace
from typing import Mapping

import numpy as np

from .rg_losses import RGPCLossTargets, compute_rgpc_loss
from .rg_pcnet import RGPCNet
from .training_data import PhasePoseDataset, RGPCDataset, collate_rgpc_samples
from .teacher_distillation import TeacherLogits, load_teacher_logits


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


def _worker_init(_worker_id: int) -> None:
    """Pickle-safe Windows spawn worker initializer."""
    import torch
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, workers: int):
    """Build a reproducible RG-PCNet loader with per-worker NumPy/Python seeds."""
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=generator, worker_init_fn=_worker_init, collate_fn=collate_rgpc_samples)


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


def _load_config_snapshot(path: Path | None) -> tuple[SimpleNamespace, bytes, str]:
    source_path = path or Path(__file__).parents[2] / "configs" / "skeleton" / "rg_pcnet_v1.py"
    source = Path(source_path).read_bytes()
    namespace: dict[str, object] = {"__file__": str(source_path)}
    exec(compile(source, str(source_path), "exec"), namespace)
    values = {key: value for key, value in namespace.items() if key.isupper() and isinstance(value, (str, int, float, bool))}
    return SimpleNamespace(**values), source, hashlib.sha256(source).hexdigest()


def _load_provenance(lock_bytes: bytes, split_bytes: bytes, release_id: str) -> tuple[dict, dict]:
    lock, split = json.loads(lock_bytes), json.loads(split_bytes)
    if not isinstance(lock, dict) or lock.get("schema_version") != "1.0":
        raise ValueError("dataset lock schema_version must be '1.0'")
    clips = lock.get("clips")
    if not isinstance(clips, list):
        raise ValueError("dataset lock clips must be a list")
    ids: set[str] = set()
    for clip in clips:
        if not isinstance(clip, Mapping):
            raise ValueError("dataset lock clips must contain mappings")
        for field in ("clip_id", "subject_id", "media_path"):
            if not isinstance(clip.get(field), str) or not clip[field].strip():
                raise ValueError(f"clip {field} must be a non-empty string")
        if clip["clip_id"] in ids:
            raise ValueError(f"duplicate clip_id: {clip['clip_id']}")
        ids.add(clip["clip_id"])
    if not isinstance(split, dict) or split.get("schema_version") != "1.0":
        raise ValueError("split manifest schema_version must be '1.0'")
    split_release = split.get("release_id")
    if split_release is not None and (not isinstance(split_release, str) or split_release != release_id):
        raise ValueError("split_manifest release_id does not match")
    partitions = split.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("split manifest partitions must be a mapping")
    for name in ("train", "validation"):
        members = partitions.get(name)
        if not isinstance(members, list) or not members:
            raise ValueError(f"split partition {name} must be a non-empty list")
        if any(not isinstance(item, str) or not item.strip() for item in members) or len(set(members)) != len(members):
            raise ValueError(f"split partition {name} contains invalid or duplicate clip IDs")
        unknown = set(members) - ids
        if unknown:
            raise ValueError(f"split references clips missing from dataset lock: {sorted(unknown)}")
    return lock, split


def _partition_ids(split: Mapping[str, object], name: str) -> list[str]:
    return list(split["partitions"][name])


def _select_clips(lock: Mapping[str, object], ids: list[str]) -> list[dict]:
    by_id = {clip["clip_id"]: dict(clip) for clip in lock["clips"]}
    return [by_id[clip_id] for clip_id in ids]


def _preflight(dataset: RGPCDataset) -> None:
    for index in range(len(dataset)):
        sample = dataset[index]
        if not bool(np.asarray(sample.valid_mask).any()):
            raise ValueError(f"clip_id {sample.clip_id} has no valid frames")


def _to_device(batch, device):
    return type(batch)(*(value.to(device) if hasattr(value, "to") else value for value in batch.__dict__.values()))


def _targets(primary):
    return RGPCLossTargets(primary.fall_target, primary.phase_target, primary.phase_mask,
                           primary.reliability_target, primary.valid_mask, primary.dt,
                           primary.teacher_fall_logit, primary.teacher_mask)


def _inject_teacher(batch, teacher: TeacherLogits | None):
    """Attach teacher values to a collated training batch without touching samples."""
    import torch

    values = [teacher.get(clip_id, 0.0) if teacher is not None else 0.0 for clip_id in batch.clip_ids]
    mask = [teacher is not None and clip_id in teacher for clip_id in batch.clip_ids]
    return replace(
        batch,
        teacher_fall_logit=torch.tensor(values, dtype=torch.float32, device=batch.features.device),
        teacher_mask=torch.tensor(mask, dtype=torch.bool, device=batch.features.device),
    )


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


def _cache_hashes(clips: list[dict], data_root: Path) -> dict[str, str]:
    resolver = PhasePoseDataset(clips, data_root)
    return {clip["clip_id"]: _sha256(resolver._resolve_feature_path(clip)) for clip in sorted(clips, key=lambda item: item["clip_id"])}


def train_rg_pcnet(*, dataset_lock: Path, split_manifest: Path, data_root: Path, output_dir: Path, release_id: str, config_path: Path | None = None, device: str = "auto", teacher_manifest: Path | None = None) -> dict[str, object]:
    """Train RG-PCNet using clean supervision and clip-aligned corrupt consistency views."""
    import torch

    dataset_lock, split_manifest, data_root, output_dir = map(Path, (dataset_lock, split_manifest, data_root, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not dataset_lock.exists() or not split_manifest.exists():
        raise FileNotFoundError("dataset lock and split manifest must exist")
    lock_bytes, split_bytes = dataset_lock.read_bytes(), split_manifest.read_bytes()
    teacher_bytes = Path(teacher_manifest).read_bytes() if teacher_manifest is not None else None
    config, config_bytes, config_sha = _load_config_snapshot(config_path)
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

    lock, split = _load_provenance(lock_bytes, split_bytes, release_id)
    train_clips, validation_clips = _select_clips(lock, _partition_ids(split, "train")), _select_clips(lock, _partition_ids(split, "validation"))
    if not train_clips or not validation_clips:
        raise ValueError("train and validation partitions must both contain clips")
    train_subjects = {str(clip["subject_id"]) for clip in train_clips}
    validation_subjects = {str(clip["subject_id"]) for clip in validation_clips}
    overlap = train_subjects & validation_subjects
    if overlap:
        raise ValueError(f"train and validation subjects overlap: {sorted(overlap)}")
    teacher = None
    if teacher_bytes is not None:
        if "outer_fold" in split:
            candidate_fold = split["outer_fold"]
            if not isinstance(candidate_fold, str) or not candidate_fold.strip():
                raise ValueError("split outer_fold must be a non-empty string")
            outer_fold = candidate_fold
        elif len(validation_subjects) == 1:
            outer_fold = next(iter(validation_subjects))
        else:
            raise ValueError("outer_fold is required when validation subjects are not exactly one")
        teacher = load_teacher_logits(
            teacher_bytes,
            train_clip_ids={str(clip["clip_id"]) for clip in train_clips},
            outer_test_clip_ids={str(clip["clip_id"]) for clip in validation_clips},
            outer_fold=outer_fold,
        )

    cache_hashes = _cache_hashes(train_clips + validation_clips, data_root)
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
            clean_batch, corrupt_batch = _inject_teacher(clean_batch, teacher), _inject_teacher(corrupt_batch, teacher)
            optimizer.zero_grad(set_to_none=True)
            clean_output = model(clean_batch.features, clean_batch.valid_mask)
            corrupted_output = model(corrupt_batch.features, corrupt_batch.valid_mask)
            loss = compute_rgpc_loss(corrupted_output, _targets(corrupt_batch), corrupted_output=clean_output)
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
    if cache_hashes != _cache_hashes(train_clips + validation_clips, data_root):
        raise RuntimeError("pose-cache inputs changed during training")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        checkpoint = {"release_id": release_id, "best_epoch": stopping.best_epoch, "best_score": stopping.best_score, "model_state_dict": model.state_dict(), "model_config": {"input_dim": int(config.INPUT_DIM), "hidden_dim": int(config.HIDDEN_DIM), "dropout": float(config.DROPOUT)}, "resolved_config": vars(config)}
        checkpoint_path = temporary / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        (temporary / "dataset_lock.json").write_bytes(lock_bytes)
        (temporary / "split_manifest.json").write_bytes(split_bytes)
        checkpoint_hash = _sha256(checkpoint_path)
        metrics = {"batch_size": int(config.BATCH_SIZE), "best_epoch": stopping.best_epoch, "best_validation_macro_f1": stopping.best_score, "checkpoint_sha256": checkpoint_hash, "device": str(resolved_device), "epochs_completed": epoch + 1, "promoted": False, "release_id": release_id, "seed": seed, "validation_subjects": sorted(validation_subjects)}
        manifest = {"cache_hashes": cache_hashes, "config_source_sha256": config_sha, "cuda": torch.version.cuda, "device": str(resolved_device), "git_commit": _git_commit(), "input_manifest_hashes": {"dataset_lock": hashlib.sha256(lock_bytes).hexdigest(), "split_manifest": hashlib.sha256(split_bytes).hexdigest()}, "release_id": release_id, "resolved_config": vars(config), "seed": seed, "torch": torch.__version__}
        if teacher is not None:
            manifest["teacher_manifest_sha256"] = teacher.manifest_sha256
            manifest["teacher_checkpoint_sha256"] = teacher.checkpoint_sha256
        (temporary / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return metrics


__all__ = ["EarlyStopping", "make_loader", "seed_everything", "train_rg_pcnet"]
