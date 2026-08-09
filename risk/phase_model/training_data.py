"""Subject-aware pose-cache samples for phase-model training.

The module keeps pose-cache loading and preprocessing in one place so that
cross-subject validation can use the same normalization contract as inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .normalization import normalize_pose_array
from .temporal_features import build_temporal_features


_COCO_LEFT_RIGHT_PAIRS = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)


@dataclass(frozen=True)
class AugmentationConfig:
    flip_probability: float = 0.5
    noise_std: float = 0.01
    keypoint_dropout: float = 0.05
    frame_mask_probability: float = 0.20
    time_scale_min: float = 0.90
    time_scale_max: float = 1.10

    def __post_init__(self) -> None:
        for name in ("flip_probability", "keypoint_dropout", "frame_mask_probability"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if float(self.noise_std) < 0.0:
            raise ValueError("noise_std must be non-negative")
        if float(self.time_scale_min) <= 0.0 or float(self.time_scale_max) < float(self.time_scale_min):
            raise ValueError("time scale bounds must satisfy 0 < minimum <= maximum")


@dataclass(frozen=True)
class SubjectFold:
    train_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]


@dataclass(frozen=True)
class PhaseSample:
    """One normalized pose-cache example and its immutable provenance fields."""

    clip_id: str
    subject_id: str
    short_embedding: object | None
    long_pose: object
    record: Mapping[str, object]
    short_quality: float = 0.0


def subject_folds(
    clips: Sequence[Mapping[str, object]], subjects: Sequence[str]
) -> tuple[SubjectFold, ...]:
    """Create leave-one-subject-out folds limited to subjects present in clips."""
    present = {str(item["subject_id"]) for item in clips}
    selected = tuple(str(subject) for subject in subjects if str(subject) in present)
    return tuple(
        SubjectFold(
            tuple(subject for subject in selected if subject != held_out),
            (held_out,),
        )
        for held_out in selected
    )


def augment_pose(pose: object, rng: object, config: AugmentationConfig) -> object:
    """Apply deterministic, label-preserving perturbations to a COCO-17 pose."""
    import numpy as np

    array = np.asarray(pose, dtype=np.float32).copy()
    if array.ndim != 3 or array.shape[1:] != (17, 3):
        raise ValueError("pose must have shape (time, 17, 3)")
    if not isinstance(config, AugmentationConfig):
        raise TypeError("config must be an AugmentationConfig")
    if not all(hasattr(rng, name) for name in ("random", "uniform", "normal", "integers")):
        raise TypeError("rng must provide NumPy Generator-style random methods")

    array = _time_scale(array, rng, config)
    if rng.random() < config.flip_probability:
        array[..., 0] *= -1.0
        for left, right in _COCO_LEFT_RIGHT_PAIRS:
            array[:, [left, right], :] = array[:, [right, left], :]
    if config.noise_std:
        visible = array[..., 2] > 0.0
        noise = rng.normal(0.0, config.noise_std, size=array[..., :2].shape).astype(np.float32)
        scale = _pose_scales(array)
        array[..., :2] += noise * scale[:, None, None] * visible[..., None]
    if config.keypoint_dropout:
        dropped = rng.random(array.shape[:2]) < config.keypoint_dropout
        array[dropped] = 0.0
    if config.frame_mask_probability:
        if rng.random() < config.frame_mask_probability:
            length = min(array.shape[0], int(rng.integers(2, 7)))
            start = int(rng.integers(0, array.shape[0] - length + 1))
            array[start : start + length] = 0.0
    return array


def _time_scale(array: object, rng: object, config: AugmentationConfig) -> object:
    import numpy as np

    pose = np.asarray(array, dtype=np.float32)
    frames = pose.shape[0]
    if frames <= 1 or config.time_scale_min == config.time_scale_max:
        return pose
    scale = float(rng.uniform(config.time_scale_min, config.time_scale_max))
    source = np.linspace(0.0, frames - 1, num=frames, dtype=np.float32)
    scaled = np.clip(source / scale, 0.0, frames - 1)
    lower = np.floor(scaled).astype(np.intp)
    upper = np.minimum(lower + 1, frames - 1)
    fraction = (scaled - lower).astype(np.float32)[:, None, None]
    return pose[lower] * (1.0 - fraction) + pose[upper] * fraction


def _pose_scales(pose: object) -> object:
    """Return the same per-frame body scale used by pose normalization."""
    import numpy as np

    array = np.asarray(pose, dtype=np.float32)
    scales = np.zeros(array.shape[0], dtype=np.float32)
    for index, frame in enumerate(array):
        visible = frame[:, 2] >= 0.25
        hips_visible = bool(visible[11] and visible[12])
        shoulders_visible = bool(visible[5] and visible[6])
        if hips_visible:
            origin = (frame[11, :2] + frame[12, :2]) / 2.0
        elif np.any(visible):
            origin = frame[visible, :2].mean(axis=0)
        else:
            continue
        if hips_visible and shoulders_visible:
            shoulder = (frame[5, :2] + frame[6, :2]) / 2.0
            scale = float(np.linalg.norm(shoulder - origin))
        else:
            spread = frame[visible, :2] - origin
            scale = float(np.sqrt(np.mean(spread * spread))) if spread.size else 0.0
        if np.isfinite(scale) and scale > 1e-6:
            scales[index] = scale
    return scales


class PhasePoseDataset:
    """Load frozen pose caches and normalize them after train-only augmentation."""

    def __init__(
        self,
        clips: Sequence[Mapping[str, object]],
        data_root: Path | str,
        *,
        train: bool = False,
        augmentation: AugmentationConfig | None = None,
        seed: int = 42,
    ) -> None:
        self._clips = tuple(dict(clip) for clip in clips)
        self._root = Path(data_root)
        self._train = train
        self._augmentation = augmentation or AugmentationConfig()
        self._seed = int(seed)
        self._epoch = 0
        for clip in self._clips:
            if not str(clip.get("clip_id", "")).strip():
                raise ValueError("each clip must have a non-empty clip_id")
            if not str(clip.get("subject_id", "")).strip():
                raise ValueError("each clip must have a non-empty subject_id")
            if not str(clip.get("media_path", "")).strip():
                raise ValueError("each clip must have a non-empty media_path")

    def __len__(self) -> int:
        return len(self._clips)

    def set_epoch(self, epoch: int) -> None:
        """Change the deterministic augmentation stream for a new training epoch."""
        self._epoch = int(epoch)

    def __getitem__(self, index: int) -> PhaseSample:
        clip = self._clips[index]
        raw_pose, _short_embedding = self._load_cache(clip)
        pose = raw_pose
        if self._train:
            import numpy as np

            pose = augment_pose(pose, np.random.default_rng(self._seed + self._epoch * len(self) + index), self._augmentation)
        normalized = normalize_pose_array(pose)
        return PhaseSample(
            clip_id=str(clip["clip_id"]),
            subject_id=str(clip["subject_id"]),
            short_embedding=None,
            long_pose=normalized,
            record=clip,
        )

    def _load_cache(self, clip: Mapping[str, object]) -> tuple[object, object | None]:
        import numpy as np

        path = self._resolve_feature_path(clip)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as cache:
                if "long_pose" not in cache:
                    raise ValueError(f"pose cache is missing long_pose: {path}")
                pose = np.asarray(cache["long_pose"], dtype=np.float32)
                short = np.asarray(cache["short_embedding"], dtype=np.float32) if "short_embedding" in cache else None
        elif path.suffix.lower() == ".npy":
            pose = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
            short = None
        else:
            raise ValueError(f"pose cache must be .npz or .npy: {path}")
        if pose.shape != (64, 17, 3):
            raise ValueError(f"long_pose must have shape (64, 17, 3): {path}")
        return pose, short

    def _resolve_feature_path(self, clip: Mapping[str, object]) -> Path:
        media_path = Path(str(clip["media_path"]))
        candidates = [self._root / media_path]
        dataset = str(clip.get("dataset") or "")
        if dataset:
            candidates.append(self._root / dataset / media_path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1]


@dataclass(frozen=True)
class RGPCSample:
    clip_id: str
    subject_id: str
    features: object
    valid_mask: object
    fall_target: float
    phase_target: object
    phase_mask: object
    reliability_target: object
    corruption: str = "clean"
    corruption_severity: float = 0.0


@dataclass(frozen=True)
class RGPCBatch:
    features: object
    valid_mask: object
    fall_target: object
    phase_target: object
    phase_mask: object
    reliability_target: object
    clip_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    corruptions: tuple[str, ...]
    corruption_severities: object


def phase_targets_for_record(record: Mapping[str, object], frames: int):
    """Create phase targets only for records with trusted frame-level labels."""
    import numpy as np

    if frames <= 0:
        raise ValueError("frames must be positive")
    sequence = record.get("coarse_phase_sequence")
    mapping = {"normal": 0, "descent_or_impact": 1, "postfall_or_recovery": 2}
    if sequence is not None:
        if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes)) or len(sequence) != frames:
            raise ValueError("coarse_phase_sequence must match the frame count")
        target = np.asarray([mapping.get(str(value), -1) for value in sequence], dtype=np.int64)
        source = str(record.get("label_source", ""))
        reviewed = str(record.get("review_status", "")) == "approved"
        trusted = reviewed and source in {"human", "official", "external_sensor"}
        mask = (target >= 0) & trusted
        target[~mask] = -1
        return target, mask
    if str(record.get("coarse_event", "")).lower() in {"adl", "normal", "nonfall"}:
        return np.zeros(frames, dtype=np.int64), np.ones(frames, dtype=bool)
    return np.full(frames, -1, dtype=np.int64), np.zeros(frames, dtype=bool)


def collate_rgpc_samples(samples: Sequence[RGPCSample]) -> RGPCBatch:
    """Pad variable-length RG-PCNet samples without converting padding to labels."""
    import numpy as np
    import torch

    if not samples:
        raise ValueError("samples must not be empty")
    batch = len(samples)
    max_time = max(len(sample.features) for sample in samples)
    feature_dim = samples[0].features.shape[1]
    features = np.zeros((batch, max_time, feature_dim), dtype=np.float32)
    valid = np.zeros((batch, max_time), dtype=bool)
    phase = np.full((batch, max_time), -1, dtype=np.int64)
    phase_mask = np.zeros((batch, max_time), dtype=bool)
    reliability = np.zeros((batch, max_time), dtype=np.float32)
    for row, sample in enumerate(samples):
        length = len(sample.features)
        features[row, :length] = sample.features
        valid[row, :length] = sample.valid_mask
        phase[row, :length] = sample.phase_target
        phase_mask[row, :length] = sample.phase_mask
        reliability[row, :length] = sample.reliability_target
    return RGPCBatch(
        torch.from_numpy(features),
        torch.from_numpy(valid),
        torch.tensor([sample.fall_target for sample in samples], dtype=torch.float32),
        torch.from_numpy(phase),
        torch.from_numpy(phase_mask),
        torch.from_numpy(reliability),
        tuple(sample.clip_id for sample in samples),
        tuple(sample.subject_id for sample in samples),
        tuple(sample.corruption for sample in samples),
        torch.tensor([sample.corruption_severity for sample in samples], dtype=torch.float32),
    )


class RGPCDataset:
    """Deterministically expose normalized pose caches as RG-PCNet samples."""

    def __init__(self, clips: Sequence[Mapping[str, object]], data_root: Path | str) -> None:
        self._pose_dataset = PhasePoseDataset(clips, data_root, train=False)

    def __len__(self) -> int:
        return len(self._pose_dataset)

    def __getitem__(self, index: int) -> RGPCSample:
        sample = self._pose_dataset[index]
        temporal = build_temporal_features(sample.long_pose)
        phase_target, phase_mask = phase_targets_for_record(sample.record, len(temporal.values))
        return RGPCSample(
            clip_id=sample.clip_id,
            subject_id=sample.subject_id,
            features=temporal.values,
            valid_mask=temporal.valid_mask,
            fall_target=float(str(sample.record.get("coarse_event", "")).lower() == "fall"),
            phase_target=phase_target,
            phase_mask=phase_mask,
            reliability_target=temporal.valid_mask.astype("float32"),
        )


__all__ = [
    "AugmentationConfig",
    "PhasePoseDataset",
    "PhaseSample",
    "RGPCBatch",
    "RGPCDataset",
    "RGPCSample",
    "SubjectFold",
    "augment_pose",
    "collate_rgpc_samples",
    "phase_targets_for_record",
    "subject_folds",
]
