"""Fold-safe teacher-logit loading and masked binary distillation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from types import MappingProxyType


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class TeacherLogits(Mapping[str, float]):
    """Immutable, sorted lookup of train-only PoseC3D fall logits."""

    _values: Mapping[str, float]
    checkpoint_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        """Defensively normalize public construction as well as loader output."""
        if not isinstance(self._values, Mapping):
            raise TypeError("teacher values must be a mapping")
        normalized: dict[str, float] = {}
        for clip_id, value in self._values.items():
            if not isinstance(clip_id, str) or not clip_id.strip():
                raise ValueError("teacher clip_id must be a non-empty string")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("teacher fall_logit must be a finite number")
            normalized[clip_id] = float(value)
        for name in ("checkpoint_sha256", "manifest_sha256"):
            digest = getattr(self, name)
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ValueError(f"{name} must be a SHA-256 hex string")
            object.__setattr__(self, name, digest.lower())
        object.__setattr__(self, "_values", MappingProxyType(dict(sorted(normalized.items()))))

    def __getitem__(self, clip_id: str) -> float:
        return self._values[clip_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def load_teacher_logits(
    manifest: Path | str | bytes,
    *,
    train_clip_ids: set[str],
    outer_test_clip_ids: set[str],
    outer_fold: str,
) -> TeacherLogits:
    """Load one snapshot of a teacher JSONL manifest after fold-safety checks."""
    raw = bytes(manifest) if isinstance(manifest, bytes) else Path(manifest).read_bytes()
    if not isinstance(outer_fold, str) or not outer_fold.strip():
        raise ValueError("outer_fold must be a non-empty string")
    values: dict[str, float] = {}
    checkpoint: str | None = None
    try:
        lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
    except UnicodeDecodeError as error:
        raise ValueError("teacher manifest must be UTF-8 JSONL") from error
    if not lines:
        raise ValueError("teacher manifest is empty")
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"teacher manifest line {line_number} is invalid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"teacher manifest line {line_number} must be an object")
        clip_id = record.get("clip_id")
        record_fold = record.get("outer_fold")
        sha = record.get("checkpoint_sha256")
        logit = record.get("fall_logit")
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError(f"teacher manifest line {line_number} clip_id must be a non-empty string")
        # Leak detection must win over the generic train-membership error.
        if clip_id in outer_test_clip_ids:
            raise ValueError(f"teacher manifest contains outer test clip: {clip_id}")
        if not isinstance(record_fold, str) or not record_fold.strip():
            raise ValueError(f"teacher manifest line {line_number} outer_fold must be a non-empty string")
        if record_fold != outer_fold:
            raise ValueError(f"teacher manifest outer_fold mismatch for clip_id {clip_id}")
        if clip_id not in train_clip_ids:
            raise ValueError(f"teacher manifest clip_id {clip_id} is not present in train clips")
        if clip_id in values:
            raise ValueError(f"duplicate teacher manifest clip_id: {clip_id}")
        if isinstance(logit, bool) or not isinstance(logit, (int, float)) or not math.isfinite(float(logit)):
            raise ValueError(f"teacher manifest fall_logit must be a finite number for clip_id {clip_id}")
        if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
            raise ValueError(f"teacher manifest checkpoint_sha256 must be a SHA-256 hex string for clip_id {clip_id}")
        sha = sha.lower()
        if checkpoint is not None and sha != checkpoint:
            raise ValueError("teacher manifest must contain one checkpoint SHA-256")
        checkpoint = sha
        values[clip_id] = float(logit)
    if not values:
        raise ValueError("teacher manifest is empty")
    return TeacherLogits(MappingProxyType(dict(sorted(values.items()))), checkpoint or "", hashlib.sha256(raw).hexdigest())


def masked_binary_distillation(student, teacher, mask, *, temperature: float = 1.0):
    """Temperature-scaled KL from teacher's `[fall_logit, 0]` distribution."""
    import torch
    import torch.nn.functional as F

    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not isinstance(student, torch.Tensor) or not isinstance(teacher, torch.Tensor):
        raise ValueError("student and teacher must be tensors")
    if student.shape != teacher.shape or student.ndim != 1 or not student.is_floating_point() or not teacher.is_floating_point() or student.device != teacher.device:
        raise ValueError("student and teacher must be floating [B] tensors on the same device")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("student and teacher logits must be finite")
    if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool or mask.shape != student.shape or mask.device != student.device:
        raise ValueError("mask must be a bool [B] tensor on the student device")
    if not torch.any(mask):
        return student.sum() * 0.0
    scale = float(temperature)
    student_pairs = torch.stack((student, torch.zeros_like(student)), dim=-1) / scale
    teacher_pairs = torch.stack((teacher, torch.zeros_like(teacher)), dim=-1) / scale
    per_item = F.kl_div(student_pairs.log_softmax(dim=-1), teacher_pairs.softmax(dim=-1), reduction="none", log_target=False).sum(dim=-1)
    result = per_item[mask].mean() * scale**2
    if not torch.isfinite(result):
        raise ValueError("distillation must be finite")
    return result


__all__ = ["TeacherLogits", "load_teacher_logits", "masked_binary_distillation"]
