"""Causal temporal model for fall, phase, and reliability predictions."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RGPCNetOutput:
    fall_logits: torch.Tensor
    window_fall_logit: torch.Tensor
    phase_logits: torch.Tensor
    reliability_logits: torch.Tensor
    window_embedding: torch.Tensor
    valid_mask: torch.Tensor


class CausalDepthwiseBlock(nn.Module):
    """A residual depthwise temporal block that only reads the past."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_pad = 2 * dilation
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation, groups=channels
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        # LayerNorm runs on [batch, time, channels], so it never mixes time steps.
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, (self.left_pad, 0))
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(F.silu(x))
        return x + residual


class RGPCNet(nn.Module):
    """Four-scale causal TCN with frame-level and window-level outputs."""

    def __init__(self, input_dim: int = 112, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            CausalDepthwiseBlock(hidden_dim, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        )
        self.fall_head = nn.Linear(hidden_dim, 1)
        self.phase_head = nn.Linear(hidden_dim, 3)
        self.reliability_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> RGPCNetOutput:
        self._validate_inputs(features, valid_mask)

        encoded = self.input_projection(features).transpose(1, 2)
        for block in self.blocks:
            encoded = block(encoded)
        frame_embeddings = encoded.transpose(1, 2)

        fall_logits = self.fall_head(frame_embeddings).squeeze(-1)
        phase_logits = self.phase_head(frame_embeddings)
        reliability_logits = self.reliability_head(frame_embeddings).squeeze(-1)

        masked_fall_logits = fall_logits.masked_fill(~valid_mask, float("-inf"))
        window_fall_logit = masked_fall_logits.max(dim=1).values
        mask_as_weights = valid_mask.unsqueeze(-1).to(frame_embeddings.dtype)
        window_embedding = (frame_embeddings * mask_as_weights).sum(dim=1)
        window_embedding = window_embedding / valid_mask.sum(dim=1, keepdim=True)

        return RGPCNetOutput(
            fall_logits=fall_logits,
            window_fall_logit=window_fall_logit,
            phase_logits=phase_logits,
            reliability_logits=reliability_logits,
            window_embedding=window_embedding,
            valid_mask=valid_mask,
        )

    def _validate_inputs(self, features: torch.Tensor, valid_mask: torch.Tensor) -> None:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features must have shape [B, T, {self.input_dim}], got {tuple(features.shape)}"
            )
        if valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must have dtype torch.bool")
        if valid_mask.shape != features.shape[:2]:
            raise ValueError(
                "valid_mask shape must match the batch and time dimensions of features"
            )
        if not valid_mask.any(dim=1).all():
            raise ValueError("each sequence must contain at least one valid frame")
