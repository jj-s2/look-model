"""Masked, phase-consistent losses for RG-PCNet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .rg_pcnet import RGPCNetOutput
from .teacher_distillation import masked_binary_distillation


FORBIDDEN = ((0, 2), (2, 1))


@dataclass(frozen=True)
class RGPCLossTargets:
    fall_target: torch.Tensor
    phase_target: torch.Tensor
    phase_mask: torch.Tensor
    reliability_target: torch.Tensor
    valid_mask: torch.Tensor
    dt: torch.Tensor
    teacher_fall_logit: torch.Tensor | None = None
    teacher_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class RGPCLoss:
    total: torch.Tensor
    components: Mapping[str, torch.Tensor]


def transition_consistency_loss(
    phase_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    dt: torch.Tensor,
    *,
    max_dt: float = 0.5,
) -> torch.Tensor:
    """Penalize the two impossible adjacent phase transitions."""
    _validate_phase_logits(phase_logits)
    _validate_frame_mask(valid_mask, phase_logits.shape[:2], "valid_mask", phase_logits.device)
    _validate_dt(dt, phase_logits.shape[:2], phase_logits.device)
    if not isinstance(max_dt, (int, float)) or not torch.isfinite(torch.tensor(float(max_dt))) or max_dt < 0:
        raise ValueError("max_dt must be a finite non-negative number")

    probabilities = phase_logits.softmax(dim=-1)
    adjacent = valid_mask[:, :-1] & valid_mask[:, 1:] & (dt[:, 1:] <= max_dt)
    penalty = sum(
        probabilities[:, :-1, source] * probabilities[:, 1:, destination]
        for source, destination in FORBIDDEN
    )
    if not torch.any(adjacent):
        return phase_logits.sum() * 0.0
    return penalty[adjacent].mean()


def compute_rgpc_loss(
    output: RGPCNetOutput,
    targets: RGPCLossTargets,
    corrupted_output: RGPCNetOutput | None = None,
    *,
    fall_pos_weight: float | torch.Tensor | None = None,
    coverage_target: float = 0.8,
    max_dt: float = 0.5,
) -> RGPCLoss:
    """Return all RG-PCNet losses, preserving gradients for empty supervision."""
    _validate_output(output, "output")
    _validate_targets(targets, output)
    if not isinstance(coverage_target, (int, float)) or not 0.0 <= float(coverage_target) <= 1.0:
        raise ValueError("coverage_target must be in [0, 1]")
    if corrupted_output is not None:
        _validate_output(corrupted_output, "corrupted_output")
        if corrupted_output.phase_logits.device != output.phase_logits.device:
            raise ValueError("corrupted_output tensors must share the output device")
        if corrupted_output.phase_logits.shape != output.phase_logits.shape:
            raise ValueError("corrupted_output phase_logits must match output")
        if corrupted_output.valid_mask.shape != output.valid_mask.shape:
            raise ValueError("corrupted_output valid_mask must match output")

    valid = targets.valid_mask
    fall_known = targets.fall_target >= 0
    if torch.any(fall_known):
        pos_weight = _pos_weight(fall_pos_weight, output.window_fall_logit)
        fall = F.binary_cross_entropy_with_logits(
            output.window_fall_logit[fall_known],
            targets.fall_target[fall_known],
            pos_weight=pos_weight,
        )
    else:
        fall = _zero(output.window_fall_logit)

    phase_valid = targets.phase_mask & valid
    if torch.any(phase_valid):
        phase = F.cross_entropy(output.phase_logits[phase_valid], targets.phase_target[phase_valid])
    else:
        phase = _zero(output.phase_logits)

    transition = transition_consistency_loss(output.phase_logits, valid, targets.dt, max_dt=max_dt)

    reliability_valid = valid & (targets.reliability_target >= 0)
    if torch.any(reliability_valid):
        reliability = F.binary_cross_entropy_with_logits(
            output.reliability_logits[reliability_valid],
            targets.reliability_target[reliability_valid],
        )
    else:
        reliability = _zero(output.reliability_logits)

    consistency = _consistency_loss(output, corrupted_output, targets.reliability_target, valid)
    if targets.teacher_fall_logit is None and targets.teacher_mask is None:
        distill = _zero(output.window_fall_logit)
    else:
        assert targets.teacher_fall_logit is not None and targets.teacher_mask is not None
        distill = masked_binary_distillation(output.window_fall_logit, targets.teacher_fall_logit, targets.teacher_mask)

    if torch.any(fall_known):
        sample_bce = F.binary_cross_entropy_with_logits(
            output.window_fall_logit[fall_known], targets.fall_target[fall_known], reduction="none"
        )
        selection = _selection(output.reliability_logits[fall_known], valid[fall_known])
        selective = (selection * sample_bce).mean() + F.relu(
            output.window_fall_logit.new_tensor(float(coverage_target)) - selection.mean()
        )
    else:
        selective = _zero(output.window_fall_logit, output.reliability_logits)

    components = {
        "fall": fall,
        "phase": phase,
        "transition": transition,
        "reliability": reliability,
        "consistency": consistency,
        "selective": selective,
        "distill": distill,
    }
    total = (
        components["fall"]
        + 0.5 * components["phase"]
        + 0.1 * components["transition"]
        + 0.3 * components["reliability"]
        + 0.1 * components["consistency"]
        + 0.1 * components["selective"]
        + 0.2 * components["distill"]
    )
    if not torch.isfinite(total) or not all(torch.isfinite(value) for value in components.values()):
        raise ValueError("loss inputs must produce finite components")
    return RGPCLoss(total=total, components=components)


def _consistency_loss(
    output: RGPCNetOutput,
    corrupted_output: RGPCNetOutput | None,
    reliability_target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if corrupted_output is None:
        return _zero(output.phase_logits)
    shared = valid & output.valid_mask & corrupted_output.valid_mask & (reliability_target >= 0.5)
    if not torch.any(shared):
        return _zero(output.phase_logits, corrupted_output.phase_logits)
    clean_log_probs = output.phase_logits[shared].log_softmax(dim=-1)
    corrupt_log_probs = corrupted_output.phase_logits[shared].log_softmax(dim=-1)
    clean_probs = clean_log_probs.exp()
    corrupt_probs = corrupt_log_probs.exp()
    return 0.5 * (
        F.kl_div(clean_log_probs, corrupt_probs, reduction="batchmean")
        + F.kl_div(corrupt_log_probs, clean_probs, reduction="batchmean")
    )


def _selection(reliability_logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weights = valid_mask.to(reliability_logits.dtype)
    counts = weights.sum(dim=1)
    if torch.any(counts <= 0):
        raise ValueError("each selected sample must contain a valid frame")
    return (reliability_logits.sigmoid() * weights).sum(dim=1) / counts


def _zero(*tensors: torch.Tensor) -> torch.Tensor:
    return sum((tensor.sum() * 0.0 for tensor in tensors), tensors[0].new_zeros(()))


def _pos_weight(value: float | torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor | None:
    if value is None:
        return None
    weight = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if weight.numel() != 1 or not torch.isfinite(weight).all() or weight.item() <= 0:
        raise ValueError("fall_pos_weight must be one finite positive value")
    return weight


def _validate_output(output: RGPCNetOutput, name: str) -> None:
    if not isinstance(output, RGPCNetOutput):
        raise TypeError(f"{name} must be an RGPCNetOutput")
    _validate_phase_logits(output.phase_logits)
    batch, time = output.phase_logits.shape[:2]
    _validate_frame_mask(output.valid_mask, (batch, time), f"{name}.valid_mask", output.phase_logits.device)
    if not output.valid_mask.any(dim=1).all():
        raise ValueError(f"{name}.valid_mask requires at least one valid frame per sample")
    for field, shape in (("reliability_logits", (batch, time)), ("fall_logits", (batch, time)), ("window_fall_logit", (batch,))):
        tensor = getattr(output, field)
        if not isinstance(tensor, torch.Tensor) or tensor.shape != shape or not tensor.is_floating_point():
            raise ValueError(f"{name}.{field} has an invalid shape or dtype")
        if tensor.device != output.phase_logits.device:
            raise ValueError(f"{name}.{field} must share the phase_logits device")
    if (
        not isinstance(output.window_embedding, torch.Tensor)
        or output.window_embedding.ndim != 2
        or output.window_embedding.shape[0] != batch
        or not output.window_embedding.is_floating_point()
        or output.window_embedding.device != output.phase_logits.device
    ):
        raise ValueError(f"{name}.window_embedding has an invalid shape, dtype, or device")


def _validate_targets(targets: RGPCLossTargets, output: RGPCNetOutput) -> None:
    if not isinstance(targets, RGPCLossTargets):
        raise TypeError("targets must be an RGPCLossTargets")
    batch, time = output.phase_logits.shape[:2]
    _validate_frame_mask(targets.valid_mask, (batch, time), "valid_mask", output.phase_logits.device)
    if not torch.equal(targets.valid_mask, output.valid_mask):
        raise ValueError("valid_mask must match output.valid_mask")
    _validate_dt(targets.dt, (batch, time), output.phase_logits.device)
    _validate_tensor(targets.fall_target, (batch,), "fall_target", output.phase_logits.device, floating=True)
    _validate_tensor(targets.phase_target, (batch, time), "phase_target", output.phase_logits.device, integral=True)
    _validate_frame_mask(targets.phase_mask, (batch, time), "phase_mask", output.phase_logits.device)
    _validate_tensor(targets.reliability_target, (batch, time), "reliability_target", output.phase_logits.device, floating=True)
    if (targets.teacher_fall_logit is None) != (targets.teacher_mask is None):
        raise ValueError("teacher_fall_logit and teacher_mask must be provided together")
    if targets.teacher_fall_logit is not None:
        _validate_tensor(targets.teacher_fall_logit, (batch,), "teacher_fall_logit", output.phase_logits.device, floating=True)
        _validate_teacher_mask(targets.teacher_mask, (batch,), output.phase_logits.device)
        if not torch.isfinite(targets.teacher_fall_logit).all():
            raise ValueError("teacher_fall_logit must be finite")
    if not torch.isfinite(targets.fall_target).all() or not torch.all(
        (targets.fall_target == -1) | (targets.fall_target == 0) | (targets.fall_target == 1)
    ):
        raise ValueError("fall_target must contain only finite {-1, 0, 1} labels")
    phase_valid = targets.phase_mask & targets.valid_mask
    if torch.any((targets.phase_target[phase_valid] < 0) | (targets.phase_target[phase_valid] > 2)):
        raise ValueError("phase_target labels must be in [0, 2] where phase_mask is true")
    if not torch.isfinite(targets.reliability_target).all() or torch.any(
        (targets.reliability_target < 0) | (targets.reliability_target > 1)
    ):
        raise ValueError("reliability_target labels must be finite and in [0, 1]")


def _validate_phase_logits(phase_logits: torch.Tensor) -> None:
    if not isinstance(phase_logits, torch.Tensor) or phase_logits.ndim != 3 or phase_logits.shape[-1] != 3 or not phase_logits.is_floating_point():
        raise ValueError("phase_logits must have shape [B, T, 3] and floating dtype")


def _validate_frame_mask(mask: torch.Tensor, shape: tuple[int, int], name: str, device: torch.device | None = None) -> None:
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype is not torch.bool
        or tuple(mask.shape) != shape
        or (device is not None and mask.device != device)
    ):
        raise ValueError(f"{name} must have shape {shape}, dtype torch.bool, and the expected device")


def _validate_teacher_mask(mask: torch.Tensor | None, shape: tuple[int], device: torch.device) -> None:
    if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool or tuple(mask.shape) != shape or mask.device != device:
        raise ValueError(f"teacher_mask must have shape {shape}, dtype torch.bool, and the expected device")


def _validate_dt(dt: torch.Tensor, shape: tuple[int, int], device: torch.device) -> None:
    _validate_tensor(dt, shape, "dt", device, floating=True)
    if not torch.isfinite(dt).all() or torch.any(dt < 0):
        raise ValueError("dt must be finite and non-negative")


def _validate_tensor(
    tensor: torch.Tensor, shape: tuple[int, ...], name: str, device: torch.device, *, floating: bool = False, integral: bool = False
) -> None:
    valid_dtype = (
        isinstance(tensor, torch.Tensor)
        and ((floating and tensor.is_floating_point()) or (integral and tensor.dtype is torch.long) or (not floating and not integral))
    )
    if not valid_dtype or tuple(tensor.shape) != shape or tensor.device != device:
        raise ValueError(f"{name} has an invalid shape or dtype")


__all__ = ["FORBIDDEN", "RGPCLoss", "RGPCLossTargets", "compute_rgpc_loss", "transition_consistency_loss"]
