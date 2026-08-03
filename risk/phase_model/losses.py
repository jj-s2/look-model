"""Masked multitask losses for partially supervised public fall datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .model import PhaseModelOutputTensor


@dataclass(frozen=True)
class LossTargets:
    phase: object
    fall_event: object
    prefall: object
    recovery: object


@dataclass(frozen=True)
class MultiTaskLoss:
    total: object
    components: Mapping[str, object]


def compute_multitask_loss(outputs: PhaseModelOutputTensor, targets: LossTargets, supervision_mask: set[str]) -> MultiTaskLoss:
    if not supervision_mask:
        raise ValueError("supervision mask cannot be empty")
    try:
        import torch
        import torch.nn.functional as F
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for multitask loss; install torch to train the neural model") from error
    components = {name: torch.zeros((), device=outputs.phase_logits.device) for name in ("phase", "prefall", "fall_event", "recovery", "order")}
    if "phase" in supervision_mask:
        valid = targets.phase >= 0
        if torch.any(valid):
            components["phase"] = F.cross_entropy(outputs.phase_logits[valid], targets.phase[valid].long())
    if "fall_event" in supervision_mask:
        components["fall_event"] = F.binary_cross_entropy_with_logits(outputs.fall_event_logit, targets.fall_event.float())
    if "prefall" in supervision_mask:
        valid = targets.prefall >= 0
        if torch.any(valid):
            logits, labels = outputs.prefall_logit[valid], targets.prefall[valid].float()
            base = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            p_t = torch.exp(-base)
            components["prefall"] = (((1.0 - p_t) ** 2) * base).mean()
    if "recovery" in supervision_mask:
        valid = targets.recovery >= 0
        if torch.any(valid):
            components["recovery"] = F.binary_cross_entropy_with_logits(outputs.recovery_logit[valid], targets.recovery[valid].float())
    total = components["phase"] + 0.8 * components["prefall"] + 0.7 * components["fall_event"] + 0.3 * components["recovery"] + 0.2 * components["order"]
    return MultiTaskLoss(total=total, components=components)
