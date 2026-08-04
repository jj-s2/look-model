"""Subject-aware pose-cache samples for phase-model training.

The module keeps pose-cache loading and preprocessing in one place so that
cross-subject validation can use the same normalization contract as inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .normalization import normalize_pose_array


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
    if not hasattr(rng, "random") or not hasattr(rng, "uniform") or not hasattr(rng, "normal"):
        raise TypeError("rng must provide NumPy Generator-style random methods")

    array = _time_scale(array, rng, config)
    if rng.random() < config.flip_probability:
        array[..., 0] *= -1.0
        for left, right in _COCO_LEFT_RIGHT_PAIRS:
            array[:, [left, right], :] = array[:, [right, left], :]
    if config.noise_std:
        visible = array[..., 2] > 0.0
        noise = rng.normal(0.0, config.noise_std, size=array[..., :2].shape).astype(np.float32)
        array[..., :2] += noise * visible[..., None]
    if config.keypoint_dropout:
        dropped = rng.random(array.shape[:2]) < config.keypoint_dropout
        array[dropped] = 0.0
    if config.frame_mask_probability:
        masked_frames = rng.random(array.shape[0]) < config.frame_mask_probability
        array[masked_frames] = 0.0
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
        raw_pose, short_embedding = self._load_cache(clip)
        pose = raw_pose
        if self._train:
            import numpy as np

            pose = augment_pose(pose, np.random.default_rng(self._seed + self._epoch * len(self) + index), self._augmentation)
        normalized = normalize_pose_array(pose)
        return PhaseSample(
            clip_id=str(clip["clip_id"]),
            subject_id=str(clip["subject_id"]),
            short_embedding=short_embedding,
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
        if pose.ndim != 3 or pose.shape[1:] != (17, 3):
            raise ValueError(f"long_pose must have shape (time, 17, 3): {path}")
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


__all__ = [
    "AugmentationConfig",
    "PhasePoseDataset",
    "PhaseSample",
    "SubjectFold",
    "augment_pose",
    "subject_folds",
]
