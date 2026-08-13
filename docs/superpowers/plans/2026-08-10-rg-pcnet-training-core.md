# RG-PCNet Training Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, batched, causal RG-PCNet training path that learns fall, coarse phase, and reliability outputs without inventing unavailable phase labels.

**Architecture:** Keep the existing PA-DTSF/PoseC3D code as baselines and add a separate `rg_pcnet` implementation. A shared NumPy feature contract feeds a proper PyTorch `nn.Module`; a masked dataset/collator and explicit corruption targets feed a multi-task loss. A new training script performs subject-safe train/validation loading, early stopping, and reproducible checkpoint generation.

**Tech Stack:** Python 3.10, NumPy >=1.24, PyTorch >=2.4 with CUDA 12.1, pytest >=7.4.

## Global Constraints

- Keep COCO-17 joint order; hips are indices `11, 12` and shoulders are `5, 6` in offline and online code.
- RG-PCNet consumes only current and historical frames; no convolution may read future frames.
- Coarse phases are exactly `normal`, `descent_or_impact`, and `postfall_or_recovery`.
- Missing phase labels use `-1` with `phase_mask=False`; never derive primary targets from the same pose rules used as model features.
- Default hidden dimension is `128`; causal TCN dilations are `1, 2, 4, 8`.
- Default real-training batch size is `8`; validation order is stable and training order is seeded.
- Save dataset hash, split hash, git commit, config, seed, environment, and checkpoint SHA-256.
- Do not modify or delete existing PA-DTSF checkpoints, PoseC3D weights, experimental outputs, or unrelated dirty-worktree files.
- Before modifying any existing symbol, run GitNexus `impact`; stop for HIGH or CRITICAL risk and review direct callers.
- Every production-code step follows RED → GREEN → REFACTOR and commits only the files listed in that task.

---

## File Structure

- Create `risk/phase_model/temporal_features.py`: one authoritative COCO-17 temporal feature contract.
- Modify `risk/phase_model/normalization.py`: align online hip indices with COCO-17.
- Modify `risk/phase_model/training_data.py`: add RG-PCNet sample construction and padded collation while retaining current APIs.
- Create `risk/phase_model/corruptions.py`: deterministic pose corruption and reliability targets.
- Create `risk/phase_model/rg_pcnet.py`: causal residual TCN and three output heads.
- Create `risk/phase_model/rg_losses.py`: masked classification, transition, reliability, consistency, and selective-risk losses.
- Create `risk/phase_model/rg_training.py`: deterministic loaders, epoch loops, early stopping, and checkpoint metadata.
- Create `configs/skeleton/rg_pcnet_v1.py`: exact initial hyperparameters.
- Create `scripts/train_rg_pcnet.py`: CLI and provenance validation for outer/inner subject folds.
- Create focused tests under `tests/risk/phase_model/` and `tests/integration/`.

### Task 1: Unify COCO-17 normalization and temporal features

**Files:**
- Create: `risk/phase_model/temporal_features.py`
- Modify: `risk/phase_model/normalization.py:66-114`
- Test: `tests/risk/phase_model/test_temporal_features.py`
- Test: `tests/risk/phase_model/test_normalization.py`

**Interfaces:**
- Consumes: normalized pose arrays shaped `[T, 17, 3]`; optional timestamps shaped `[T]`.
- Produces: `TemporalPoseFeatures(values, valid_mask, missing_mask, dt)` and `build_temporal_features(pose, timestamps=None)`.

- [ ] **Step 1: Write failing normalization and feature-contract tests**

Before writing, name the break: using nose indices as hips or changing the 112-feature ordering must fail these tests.

```python
# tests/risk/phase_model/test_temporal_features.py
import numpy as np
import pytest

from risk.phase_model.temporal_features import FEATURE_DIM, build_temporal_features


def _pose(frames: int = 3) -> np.ndarray:
    pose = np.zeros((frames, 17, 3), dtype=np.float32)
    pose[:, :, 2] = 1.0
    pose[:, 11, :2] = (10.0, 20.0)
    pose[:, 12, :2] = (12.0, 20.0)
    pose[:, 5, :2] = (10.0, 0.0)
    pose[:, 6, :2] = (12.0, 0.0)
    pose[1:, 11:13, 1] += np.arange(1, frames, dtype=np.float32)[:, None]
    return pose


def test_temporal_features_have_stable_shape_masks_and_real_dt():
    result = build_temporal_features(_pose(), np.array([0.0, 0.1, 0.35], dtype=np.float32))
    assert FEATURE_DIM == 112
    assert result.values.shape == (3, 112)
    assert result.valid_mask.tolist() == [True, True, True]
    assert result.missing_mask.shape == (3, 17)
    assert result.dt.tolist() == pytest.approx([0.0, 0.1, 0.25])


def test_missing_joint_is_explicit_and_does_not_create_velocity_spike():
    pose = _pose()
    pose[1, 0] = 0.0
    result = build_temporal_features(pose)
    assert result.missing_mask[1, 0]
    velocity_offset = 17 * 4
    assert result.values[1, velocity_offset : velocity_offset + 2].tolist() == [0.0, 0.0]
```

Append this test to `tests/risk/phase_model/test_normalization.py`:

```python
def test_online_normalization_uses_coco_hip_indices_11_and_12():
    item = _observation()
    points = list(item.keypoints)
    points[0], points[1] = (900.0, 800.0), (1000.0, 800.0)
    points[11], points[12] = (10.0, 20.0), (12.0, 20.0)
    corrected = PoseObservation(
        timestamp=item.timestamp, tracking_id=item.tracking_id,
        keypoints=tuple(points), scores=item.scores, visible_mask=item.visible_mask,
        bbox=item.bbox, frame_size=item.frame_size, stream_fresh=item.stream_fresh,
    )
    result = normalize_pose_window(DualWindow(short=(corrected,), long=()))
    assert result.coordinates[0][11][0] == pytest.approx(-0.05)
    assert result.coordinates[0][12][0] == pytest.approx(0.05)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/risk/phase_model/test_temporal_features.py tests/risk/phase_model/test_normalization.py -q
```

Expected: import failure for `temporal_features` and failure showing online normalization still uses `(0, 1)` as hips.

- [ ] **Step 3: Implement the feature contract and hip-index fix**

Create `risk/phase_model/temporal_features.py` with these public declarations and ordering:

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

JOINTS = 17
JOINT_BASE_DIM = JOINTS * 4       # x, y, confidence, missing
JOINT_VELOCITY_DIM = JOINTS * 2  # dx/dt, dy/dt
GLOBAL_DIM = 10
FEATURE_DIM = JOINT_BASE_DIM + JOINT_VELOCITY_DIM + GLOBAL_DIM


@dataclass(frozen=True)
class TemporalPoseFeatures:
    values: np.ndarray
    valid_mask: np.ndarray
    missing_mask: np.ndarray
    dt: np.ndarray


def build_temporal_features(pose: np.ndarray, timestamps: np.ndarray | None = None) -> TemporalPoseFeatures:
    array = np.asarray(pose, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (JOINTS, 3):
        raise ValueError("pose must have shape (time, 17, 3)")
    frames = array.shape[0]
    if frames == 0:
        raise ValueError("pose must contain at least one frame")
    if timestamps is None:
        times = np.arange(frames, dtype=np.float32)
    else:
        times = np.asarray(timestamps, dtype=np.float32)
        if times.shape != (frames,) or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("timestamps must be finite, strictly increasing, and match time")
    dt = np.zeros(frames, dtype=np.float32)
    dt[1:] = np.diff(times)
    visible = array[..., 2] >= 0.25
    missing = ~visible
    base = np.concatenate((array[..., :2], array[..., 2:3], missing[..., None].astype(np.float32)), axis=-1)
    velocity = np.zeros((frames, JOINTS, 2), dtype=np.float32)
    valid_pair = visible[1:] & visible[:-1]
    raw_velocity = (array[1:, :, :2] - array[:-1, :, :2]) / dt[1:, None, None]
    velocity[1:] = np.where(valid_pair[..., None], raw_velocity, 0.0)
    hips = array[:, (11, 12), :2].mean(axis=1)
    shoulders = array[:, (5, 6), :2].mean(axis=1)
    root_delta = np.zeros((frames, 2), dtype=np.float32)
    root_delta[1:] = (hips[1:] - hips[:-1]) / dt[1:, None]
    torso = shoulders - hips
    torso_norm = np.linalg.norm(torso, axis=1).clip(min=1e-6)
    mins = np.where(visible[..., None], array[..., :2], np.inf).min(axis=1)
    maxs = np.where(visible[..., None], array[..., :2], -np.inf).max(axis=1)
    extent = np.where(np.isfinite(maxs - mins), maxs - mins, 0.0)
    valid_ratio = visible.mean(axis=1).astype(np.float32)
    mean_conf = np.where(visible, array[..., 2], 0.0).sum(axis=1) / visible.sum(axis=1).clip(min=1)
    global_features = np.column_stack((
        dt, root_delta[:, 0], root_delta[:, 1], np.linalg.norm(root_delta, axis=1),
        torso[:, 0] / torso_norm, torso[:, 1] / torso_norm,
        extent[:, 0], extent[:, 1], valid_ratio, mean_conf,
    )).astype(np.float32)
    values = np.concatenate((base.reshape(frames, -1), velocity.reshape(frames, -1), global_features), axis=1)
    return TemporalPoseFeatures(values, visible.any(axis=1), missing, dt)
```

In `normalize_pose_window`, replace `hip_indices = (0, 1)` with `hip_indices = (11, 12)`.

- [ ] **Step 4: Run tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add risk/phase_model/temporal_features.py risk/phase_model/normalization.py tests/risk/phase_model/test_temporal_features.py tests/risk/phase_model/test_normalization.py
git commit -m "feat: unify temporal pose feature contract"
```

### Task 2: Add deterministic padded batches without fake labels

**Files:**
- Modify: `risk/phase_model/training_data.py:54-245`
- Test: `tests/risk/phase_model/test_rg_training_data.py`

**Interfaces:**
- Consumes: `PhasePoseDataset` records and `TemporalPoseFeatures`.
- Produces: `RGPCSample`, `RGPCBatch`, `RGPCDataset`, and `collate_rgpc_samples(samples)`.

- [ ] **Step 1: Write failing dataset and collation tests**

Name the break: assigning fall-phase labels to unlabeled fall clips or dropping padding masks must fail.

```python
import numpy as np
import pytest

from risk.phase_model.training_data import RGPCSample, collate_rgpc_samples, phase_targets_for_record


def test_unlabeled_fall_clip_has_no_primary_phase_supervision():
    target, mask = phase_targets_for_record({"coarse_event": "fall"}, frames=4)
    assert target.tolist() == [-1, -1, -1, -1]
    assert mask.tolist() == [False, False, False, False]


def test_adl_clip_is_supervised_as_normal_only():
    target, mask = phase_targets_for_record({"coarse_event": "adl"}, frames=3)
    assert target.tolist() == [0, 0, 0]
    assert mask.tolist() == [True, True, True]


def test_collate_pads_time_and_preserves_valid_masks():
    a = RGPCSample("a", "s1", np.ones((2, 112), np.float32), np.array([True, True]), 0.0,
                   np.array([0, 0]), np.array([True, True]), np.ones(2, np.float32))
    b = RGPCSample("b", "s2", np.ones((3, 112), np.float32), np.array([True, True, True]), 1.0,
                   np.array([-1, -1, -1]), np.array([False, False, False]), np.ones(3, np.float32))
    batch = collate_rgpc_samples([a, b])
    assert batch.features.shape == (2, 3, 112)
    assert batch.valid_mask.tolist() == [[True, True, False], [True, True, True]]
    assert batch.phase_mask.tolist() == [[True, True, False], [False, False, False]]
    assert batch.subject_ids == ("s1", "s2")
```

- [ ] **Step 2: Run the new tests and verify RED**

Run `python -m pytest tests/risk/phase_model/test_rg_training_data.py -q`.

Expected: imports fail because the RG-PCNet batch contract does not exist.

- [ ] **Step 3: Add exact immutable sample and batch contracts**

Append to `training_data.py`:

```python
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
```

Implement `collate_rgpc_samples` with NumPy padding, then convert to torch tensors:

```python
def collate_rgpc_samples(samples: Sequence[RGPCSample]) -> RGPCBatch:
    import numpy as np
    import torch
    if not samples:
        raise ValueError("samples must not be empty")
    batch, max_time, feature_dim = len(samples), max(len(s.features) for s in samples), samples[0].features.shape[1]
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
        torch.from_numpy(features), torch.from_numpy(valid),
        torch.tensor([s.fall_target for s in samples], dtype=torch.float32),
        torch.from_numpy(phase), torch.from_numpy(phase_mask), torch.from_numpy(reliability),
        tuple(s.clip_id for s in samples), tuple(s.subject_id for s in samples),
        tuple(s.corruption for s in samples),
        torch.tensor([s.corruption_severity for s in samples], dtype=torch.float32),
    )
```

Add `RGPCDataset` as a thin wrapper over pose caches: load normalized pose, call `build_temporal_features`, create fall target from `coarse_event`, and create phase targets through `phase_targets_for_record`. Training augmentation is added only in Task 3.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
python -m pytest tests/risk/phase_model/test_rg_training_data.py tests/risk/phase_model/test_training_data.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add risk/phase_model/training_data.py tests/risk/phase_model/test_rg_training_data.py
git commit -m "feat: add masked rgpc training batches"
```

### Task 3: Add auditable corruption and reliability targets

**Files:**
- Create: `risk/phase_model/corruptions.py`
- Modify: `risk/phase_model/training_data.py`
- Test: `tests/risk/phase_model/test_corruptions.py`

**Interfaces:**
- Consumes: normalized pose `[T,17,3]`, seeded NumPy generator, severity `[0,1]`.
- Produces: `CorruptedPose(pose, reliability_target, corruption, severity)` and `corrupt_pose(pose, rng, *, severity, corruption)`.

- [ ] **Step 1: Write failing behavior tests**

```python
import numpy as np
import pytest

from risk.phase_model.corruptions import corrupt_pose


def test_corruption_is_seeded_and_severity_reduces_reliability():
    pose = np.ones((16, 17, 3), dtype=np.float32)
    first = corrupt_pose(pose, np.random.default_rng(7), severity=0.8, corruption="joint_dropout")
    second = corrupt_pose(pose, np.random.default_rng(7), severity=0.8, corruption="joint_dropout")
    assert first.pose == pytest.approx(second.pose)
    assert first.reliability_target.shape == (16,)
    assert float(first.reliability_target.mean()) < 0.5


def test_clean_corruption_preserves_pose_and_full_reliability():
    pose = np.ones((4, 17, 3), dtype=np.float32)
    result = corrupt_pose(pose, np.random.default_rng(3), severity=0.0, corruption="frame_drop")
    assert result.pose == pytest.approx(pose)
    assert result.reliability_target.tolist() == [1.0] * 4
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/risk/phase_model/test_corruptions.py -q`.

Expected: module import fails.

- [ ] **Step 3: Implement bounded corruption families**

Create a frozen `CorruptedPose` dataclass and implement `corrupt_pose` for exactly these names: `joint_dropout`, `frame_drop`, `freeze`, `confidence_collapse`, and `time_jitter`. Validate shape and severity, copy input, and set per-frame reliability to `clip(1 - severity * affected_fraction, 0, 1)`. `time_jitter` changes timestamps only in a separate returned `timestamp_scale`; it must not reorder frames. Do not add camera rotation to primary reliability corruption because rotation can change the semantic direction of a fall.

Use this public signature:

```python
@dataclass(frozen=True)
class CorruptedPose:
    pose: np.ndarray
    reliability_target: np.ndarray
    corruption: str
    severity: float
    timestamp_scale: np.ndarray


def corrupt_pose(
    pose: np.ndarray,
    rng: np.random.Generator,
    *,
    severity: float,
    corruption: str,
) -> CorruptedPose:
    array = np.asarray(pose, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (17, 3) or array.shape[0] == 0:
        raise ValueError("pose must have shape (time, 17, 3) and contain frames")
    severity = float(severity)
    if not np.isfinite(severity) or not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be finite and in [0, 1]")
    allowed = {"joint_dropout", "frame_drop", "freeze", "confidence_collapse", "time_jitter"}
    if corruption not in allowed:
        raise ValueError(f"unsupported corruption: {corruption}")
    changed = array.copy()
    frames = len(changed)
    affected = np.zeros(frames, dtype=np.float32)
    timestamp_scale = np.ones(frames, dtype=np.float32)
    if severity == 0.0:
        return CorruptedPose(changed, np.ones(frames, np.float32), corruption, severity, timestamp_scale)
    if corruption == "joint_dropout":
        mask = rng.random((frames, 17)) < severity
        changed[mask] = 0.0
        affected = mask.mean(axis=1).astype(np.float32)
    elif corruption == "frame_drop":
        mask = rng.random(frames) < severity
        changed[mask] = 0.0
        affected = mask.astype(np.float32)
    elif corruption == "freeze":
        length = min(frames, max(1, round(frames * severity)))
        start = int(rng.integers(0, frames - length + 1))
        changed[start : start + length] = changed[max(0, start - 1)]
        affected[start : start + length] = 1.0
    elif corruption == "confidence_collapse":
        changed[..., 2] *= 1.0 - severity
        affected.fill(severity)
    else:
        timestamp_scale = np.clip(rng.normal(1.0, 0.1 * severity, frames), 0.5, 1.5).astype(np.float32)
        affected = np.clip(np.abs(timestamp_scale - 1.0) / 0.1, 0.0, 1.0)
    reliability = np.clip(1.0 - severity * affected, 0.0, 1.0).astype(np.float32)
    return CorruptedPose(changed, reliability, corruption, severity, timestamp_scale)
```

In `RGPCDataset.__getitem__`, derive a generator from `seed + epoch * len(dataset) + index`, return one clean or corrupted sample according to a configured probability, and build temporal features after corruption. Keep the corruption name and severity in the sample metadata used by experiment artifacts.

- [ ] **Step 4: Verify GREEN and no regressions**

Run:

```powershell
python -m pytest tests/risk/phase_model/test_corruptions.py tests/risk/phase_model/test_rg_training_data.py tests/risk/phase_model/test_training_data.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add risk/phase_model/corruptions.py risk/phase_model/training_data.py tests/risk/phase_model/test_corruptions.py
git commit -m "feat: supervise pose reliability with corruptions"
```

### Task 4: Implement the causal RG-PCNet model

**Files:**
- Create: `risk/phase_model/rg_pcnet.py`
- Test: `tests/risk/phase_model/test_rg_pcnet.py`

**Interfaces:**
- Consumes: `features: Tensor[B,T,112]`, `valid_mask: BoolTensor[B,T]`.
- Produces: `RGPCNetOutput(fall_logits, window_fall_logit, phase_logits, reliability_logits, window_embedding, valid_mask)`.

- [ ] **Step 1: Write failing shape and causality tests**

```python
import pytest

torch = pytest.importorskip("torch")

from risk.phase_model.rg_pcnet import RGPCNet


def test_rgpcnet_outputs_all_three_tasks():
    model = RGPCNet(input_dim=112, hidden_dim=16, dropout=0.0)
    output = model(torch.zeros(2, 8, 112), torch.tensor([[1] * 8, [1] * 5 + [0] * 3], dtype=torch.bool))
    assert output.fall_logits.shape == (2, 8)
    assert output.window_fall_logit.shape == (2,)
    assert output.phase_logits.shape == (2, 8, 3)
    assert output.reliability_logits.shape == (2, 8)
    assert output.window_embedding.shape == (2, 16)


def test_rgpcnet_is_causal_for_prefix_outputs():
    torch.manual_seed(4)
    model = RGPCNet(input_dim=112, hidden_dim=16, dropout=0.0).eval()
    first = torch.randn(1, 12, 112)
    changed = first.clone()
    changed[:, 8:] = torch.randn_like(changed[:, 8:]) * 20
    mask = torch.ones(1, 12, dtype=torch.bool)
    with torch.no_grad():
        a = model(first, mask).fall_logits[:, :8]
        b = model(changed, mask).fall_logits[:, :8]
    assert a == pytest.approx(b, abs=1e-6)
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/risk/phase_model/test_rg_pcnet.py -q`.

Expected: module import fails.

- [ ] **Step 3: Implement a real `nn.Module` with left padding only**

The file must expose:

```python
@dataclass(frozen=True)
class RGPCNetOutput:
    fall_logits: torch.Tensor
    window_fall_logit: torch.Tensor
    phase_logits: torch.Tensor
    reliability_logits: torch.Tensor
    window_embedding: torch.Tensor
    valid_mask: torch.Tensor


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_pad = 2 * dilation
        self.depthwise = nn.Conv1d(channels, channels, 3, dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.nn.functional.pad(x, (self.left_pad, 0))
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.dropout(torch.nn.functional.silu(self.norm(x)))
        return x + residual
```

`RGPCNet` uses `nn.Linear(input_dim, hidden_dim)`, four `CausalDepthwiseBlock`s with dilations `(1,2,4,8)`, and three frame heads. Mask invalid logits before masked max pooling; masked mean produces the embedding. Reject empty masks and mismatched shapes with `ValueError`.

- [ ] **Step 4: Verify GREEN**

Run the command from Step 2. Expected: both tests pass.

- [ ] **Step 5: Commit**

```powershell
git add risk/phase_model/rg_pcnet.py tests/risk/phase_model/test_rg_pcnet.py
git commit -m "feat: add causal reliability gated tcn"
```

### Task 5: Implement masked multi-task and transition losses

**Files:**
- Create: `risk/phase_model/rg_losses.py`
- Test: `tests/risk/phase_model/test_rg_losses.py`

**Interfaces:**
- Consumes: `RGPCNetOutput`, `RGPCLossTargets`, optional corrupted output.
- Produces: `RGPCLoss(total, components)` and `transition_consistency_loss(phase_logits, valid_mask, dt)`.

- [ ] **Step 1: Write failing hand-derived loss tests**

```python
import pytest

torch = pytest.importorskip("torch")

from risk.phase_model.rg_losses import transition_consistency_loss


def test_allowed_normal_to_descent_transition_has_negligible_penalty():
    logits = torch.tensor([[[12.0, -12.0, -12.0], [-12.0, 12.0, -12.0]]])
    loss = transition_consistency_loss(logits, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0.0, 0.1]]))
    assert loss.item() < 1e-6


def test_forbidden_normal_to_postfall_transition_is_penalized():
    logits = torch.tensor([[[12.0, -12.0, -12.0], [-12.0, -12.0, 12.0]]])
    loss = transition_consistency_loss(logits, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0.0, 0.1]]))
    assert loss.item() > 0.99


def test_large_timestamp_gap_is_not_treated_as_adjacent():
    logits = torch.tensor([[[12.0, -12.0, -12.0], [-12.0, -12.0, 12.0]]])
    loss = transition_consistency_loss(logits, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0.0, 3.0]]), max_dt=0.5)
    assert loss.item() == 0.0
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/risk/phase_model/test_rg_losses.py -q`.

- [ ] **Step 3: Implement exact masks and forbidden-edge matrix**

Define forbidden edges as `normal -> postfall_or_recovery` and `postfall_or_recovery -> descent_or_impact`:

```python
FORBIDDEN = ((0, 2), (2, 1))


def transition_consistency_loss(phase_logits, valid_mask, dt, *, max_dt: float = 0.5):
    probabilities = phase_logits.softmax(dim=-1)
    adjacent = valid_mask[:, :-1] & valid_mask[:, 1:] & (dt[:, 1:] <= max_dt)
    penalty = sum(probabilities[:, :-1, i] * probabilities[:, 1:, j] for i, j in FORBIDDEN)
    if not torch.any(adjacent):
        return phase_logits.sum() * 0.0
    return penalty[adjacent].mean()
```

Add `RGPCLossTargets` with `fall_target`, `phase_target`, `phase_mask`, `reliability_target`, `valid_mask`, and `dt`. `compute_rgpc_loss` must calculate:

- weighted BCE on `window_fall_logit`;
- masked three-class CE only where `phase_mask & valid_mask`;
- transition loss above;
- BCE on reliability logits only at valid frames;
- clean/corrupted symmetric KL only where both are valid and target reliability is at least `0.5`;
- selective loss `mean(selection * sample_bce) + relu(coverage_target - mean(selection))`, where selection is the valid-frame mean sigmoid reliability per sample.

Use weights `fall=1.0`, `phase=0.5`, `transition=0.1`, `reliability=0.3`, `consistency=0.1`, `selective=0.1`. Every component must remain a differentiable zero tensor when its supervision is absent.

- [ ] **Step 4: Verify GREEN and model integration**

Run:

```powershell
python -m pytest tests/risk/phase_model/test_rg_losses.py tests/risk/phase_model/test_rg_pcnet.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add risk/phase_model/rg_losses.py tests/risk/phase_model/test_rg_losses.py
git commit -m "feat: add phase consistent selective losses"
```

### Task 6: Add deterministic DataLoaders and early-stopped training

**Files:**
- Create: `risk/phase_model/rg_training.py`
- Create: `configs/skeleton/rg_pcnet_v1.py`
- Create: `scripts/train_rg_pcnet.py`
- Test: `tests/risk/phase_model/test_rg_training.py`
- Test: `tests/integration/test_train_rg_pcnet.py`

**Interfaces:**
- Consumes: frozen dataset lock, one subject-isolated split, pose-cache root, config.
- Produces: `checkpoint.pt`, `metrics.json`, `run_manifest.json`, copied dataset/split manifests.

- [ ] **Step 1: Write failing deterministic and early-stop tests**

```python
# tests/risk/phase_model/test_rg_training.py
import pytest

torch = pytest.importorskip("torch")

from risk.phase_model.rg_training import EarlyStopping, seed_everything


def test_seed_everything_repeats_torch_values():
    seed_everything(23)
    first = torch.rand(4)
    seed_everything(23)
    assert first == pytest.approx(torch.rand(4))


def test_early_stopping_restores_best_epoch_not_last_epoch():
    state = EarlyStopping(patience=2, mode="max")
    assert state.update(0.4, epoch=0)
    assert state.update(0.6, epoch=1)
    assert not state.update(0.5, epoch=2)
    assert not state.update(0.4, epoch=3)
    assert state.should_stop
    assert state.best_epoch == 1
```

The integration test creates two ADL and two fall `.npz` fixtures across two training subjects plus one validation subject, trains one epoch on CPU, and asserts:

```python
assert metrics["batch_size"] == 2
assert metrics["best_epoch"] == 0
assert metrics["promoted"] is False
assert metrics["validation_subjects"] == ["s3"]
assert len(metrics["checkpoint_sha256"]) == 64
assert json.loads((output / "run_manifest.json").read_text())["git_commit"]
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/risk/phase_model/test_rg_training.py tests/integration/test_train_rg_pcnet.py -q
```

- [ ] **Step 3: Implement deterministic training primitives**

`seed_everything(seed)` seeds Python, NumPy, torch CPU and all CUDA devices, then sets deterministic CuDNN flags. `make_loader(dataset, batch_size, shuffle, seed, workers)` creates a seeded generator and worker initializer and uses `collate_rgpc_samples`. `EarlyStopping` stores `best_score`, `best_epoch`, `bad_epochs`, and `should_stop`.

Use the stable entry point `train_rg_pcnet(*, dataset_lock: Path, split_manifest: Path, data_root: Path, output_dir: Path, release_id: str, config_path: Path | None = None, device: str = "auto") -> dict[str, object]`.

The function must build separate train and validation datasets, reject overlapping subject IDs, train with AdamW, clip gradient norm at `1.0`, use `ReduceLROnPlateau`, evaluate validation macro F1 without changing thresholds, retain a deep copy of the best state dict, restore it before saving, and never mark the training-only artifact promoted.

Create `configs/skeleton/rg_pcnet_v1.py` with exact values:

```python
SEED = 42
INPUT_DIM = 112
HIDDEN_DIM = 128
DROPOUT = 0.20
BATCH_SIZE = 8
EPOCHS = 40
PATIENCE = 6
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 0
CORRUPTION_PROBABILITY = 0.50
```

The CLI accepts `--dataset-lock`, `--split-manifest`, `--data-root`, `--output`, `--release-id`, `--config`, and `--device` and prints sorted JSON metrics.

- [ ] **Step 4: Verify GREEN on CPU and CUDA availability**

Run:

```powershell
python -m pytest tests/risk/phase_model/test_rg_training.py tests/integration/test_train_rg_pcnet.py -q
python -c "import torch; print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})"
```

Expected: tests pass; environment command reports the actual training backend without changing tests based on its value.

- [ ] **Step 5: Run the focused regression suite**

```powershell
python -m pytest tests/risk/phase_model tests/integration/test_train_phase_model.py tests/integration/test_train_rg_pcnet.py -q
```

Expected: zero failures. Existing PA-DTSF tests remain green.

- [ ] **Step 6: Commit**

```powershell
git add risk/phase_model/rg_training.py configs/skeleton/rg_pcnet_v1.py scripts/train_rg_pcnet.py tests/risk/phase_model/test_rg_training.py tests/integration/test_train_rg_pcnet.py
git commit -m "feat: train rgpcnet with subject safe early stopping"
```

### Task 7: Add fold-safe optional PoseC3D distillation

**Files:**
- Create: `risk/phase_model/teacher_distillation.py`
- Modify: `risk/phase_model/rg_losses.py`
- Modify: `risk/phase_model/rg_training.py`
- Test: `tests/risk/phase_model/test_teacher_distillation.py`

**Interfaces:**
- Consumes: fold-specific JSONL records containing `clip_id`, `outer_fold`, `fall_logit`, and `checkpoint_sha256`.
- Produces: validated `TeacherLogits` indexed only by training clip ID and an optional masked KL component named `distill`.

- [ ] **Step 1: Write failing leakage and KL tests**

```python
import pytest

torch = pytest.importorskip("torch")

from risk.phase_model.teacher_distillation import load_teacher_logits, masked_binary_distillation


def test_teacher_manifest_rejects_outer_test_clip(tmp_path):
    path = tmp_path / "teacher.jsonl"
    path.write_text('{"clip_id":"test-a","outer_fold":"s4","fall_logit":2.0,"checkpoint_sha256":"' + "a" * 64 + '"}\n')
    with pytest.raises(ValueError, match="outer test clip"):
        load_teacher_logits(path, train_clip_ids={"train-a"}, outer_test_clip_ids={"test-a"}, outer_fold="s4")


def test_matching_teacher_and_student_logits_have_near_zero_distillation():
    student = torch.tensor([2.0, -1.0])
    teacher = torch.tensor([2.0, -1.0])
    loss = masked_binary_distillation(student, teacher, torch.tensor([True, True]), temperature=2.0)
    assert loss.item() < 1e-7
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/risk/phase_model/test_teacher_distillation.py -q`.

- [ ] **Step 3: Implement manifest validation and masked binary KL**

Reject duplicate clip IDs, fold mismatch, non-finite logits, invalid SHA-256, any outer-test clip, and any record not present in `train_clip_ids`. Return an immutable mapping ordered by clip ID. Implement distillation by dividing both logits by temperature, applying `log_softmax` to the student pair `[logit, 0]`, applying `softmax` to the teacher pair, computing `kl_div(student_log_probs, teacher_probs, reduction="none", log_target=False)`, summing the class dimension, selecting the explicit mask, averaging, and multiplying by `temperature ** 2`. An empty mask returns a differentiable zero.

Add `teacher_fall_logit` and `teacher_mask` to `RGPCBatch`. When no teacher manifest is supplied, fill zeros and a false mask. Add `0.2 * distill` to `compute_rgpc_loss`. Save teacher checkpoint hash and manifest hash in the training run manifest.

- [ ] **Step 4: Verify GREEN and no-teacher parity**

```powershell
python -m pytest tests/risk/phase_model/test_teacher_distillation.py tests/risk/phase_model/test_rg_losses.py tests/risk/phase_model/test_rg_training.py tests/integration/test_train_rg_pcnet.py -q
```

Add an integration assertion that identical seeds and no teacher manifest produce the same checkpoint hash before and after enabling the optional code path.

- [ ] **Step 5: Commit**

```powershell
git add risk/phase_model/teacher_distillation.py risk/phase_model/rg_losses.py risk/phase_model/rg_training.py tests/risk/phase_model/test_teacher_distillation.py tests/risk/phase_model/test_rg_losses.py tests/risk/phase_model/test_rg_training.py tests/integration/test_train_rg_pcnet.py
git commit -m "feat: distill fold safe posec3d logits"
```

## Plan 1 Completion Gate

Before moving to calibration/release:

1. Run `python -m pytest tests/risk/phase_model tests/integration/test_train_phase_model.py tests/integration/test_train_rg_pcnet.py -q` with zero failures.
2. Run one CPU smoke training and, if CUDA is available, one CUDA smoke training using the same fixture and compare output schema.
3. Run GitNexus `detect_changes(scope="all", worktree="F:\\邵吉锦\\look model\\.worktrees\\elderly-monitoring")`; inspect every affected process.
4. Confirm no tracked model weights or output artifacts were committed.
5. Push the task commits to `origin/codex/elderly-monitoring` only after local and remote hashes match.
