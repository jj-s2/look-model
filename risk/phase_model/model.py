"""Optional PyTorch implementation of the quality-gated PA-DTSF heads."""

from __future__ import annotations

from dataclasses import dataclass


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for risk.phase_model.model; install torch to train or run the neural model") from error
    return torch, nn


@dataclass(frozen=True)
class PhaseModelOutputTensor:
    phase_logits: object
    fall_event_logit: object
    prefall_logit: object
    recovery_logit: object
    abstain_logit: object | None = None


class LongBranchTCN:
    def __init__(self, *, joints: int = 17, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        _, nn = _torch()
        if joints <= 0 or hidden_dim <= 0:
            raise ValueError("joints and hidden_dim must be positive")
        layers = []
        channels = joints * 3
        for dilation in (1, 2, 4):
            layers.extend([
                nn.Conv1d(channels, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation, groups=1),
                nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            ])
            channels = hidden_dim
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def __call__(self, long_pose):
        x = long_pose.reshape(long_pose.shape[0], long_pose.shape[1], -1).transpose(1, 2)
        return self.pool(self.network(x)).squeeze(-1)


class PhaseAwareFusionModel:
    def __init__(self, *, short_dim: int = 512, joints: int = 17, hidden_dim: int = 128) -> None:
        _, nn = _torch()
        if min(short_dim, joints, hidden_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        self.long_branch = LongBranchTCN(joints=joints, hidden_dim=hidden_dim)
        self.short_projection = nn.Sequential(nn.Linear(short_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.long_projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.phase_head = nn.Linear(hidden_dim, 6)
        self.fall_head = nn.Linear(hidden_dim, 1)
        self.prefall_head = nn.Linear(hidden_dim, 1)
        self.recovery_head = nn.Linear(hidden_dim, 1)
        self.abstain_head = nn.Linear(hidden_dim, 1)

    def __call__(self, short_embedding, long_pose, short_quality, long_quality) -> PhaseModelOutputTensor:
        torch, _ = _torch()
        if short_embedding.shape[0] != long_pose.shape[0]:
            raise ValueError("short and long branches must have the same batch size")
        weights = torch.stack((short_quality, long_quality), dim=1).to(short_embedding.dtype)
        if torch.any(weights.sum(dim=1) <= 0):
            raise ValueError("at least one branch must have positive quality")
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        short = self.short_projection(short_embedding)
        long = self.long_projection(self.long_branch(long_pose))
        fused = weights[:, :1] * short + weights[:, 1:] * long
        return PhaseModelOutputTensor(
            phase_logits=self.phase_head(fused),
            fall_event_logit=self.fall_head(fused).squeeze(-1),
            prefall_logit=self.prefall_head(fused).squeeze(-1),
            recovery_logit=self.recovery_head(fused).squeeze(-1),
            abstain_logit=self.abstain_head(fused).squeeze(-1),
        )

    def parameters(self):
        modules = (self.long_branch.network, self.long_branch.pool, self.short_projection, self.long_projection, self.phase_head, self.fall_head, self.prefall_head, self.recovery_head, self.abstain_head)
        return (parameter for module in modules for parameter in module.parameters())
