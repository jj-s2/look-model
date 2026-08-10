"""RG-PCNet release adapter with calibrated reliability abstention."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from .release_config import RGPCReleaseConfig, load_release_config
from .rg_pcnet import RGPCNet
from .schema import Phase, PhaseModelOutput
from .temporal_features import FEATURE_DIM, build_temporal_features
from .windows import DualWindow


class RGPredictor:
    """Load one immutable RG-PCNet release and expose phase-model output."""

    def __init__(self, model: RGPCNet, config: RGPCReleaseConfig, *, device: str) -> None:
        self._torch = torch
        self._model = model
        self._config = config
        self.device = torch.device(device)
        self.model_version = config.release_id
        self.embedding_version = "rgpc-temporal-v1"
        self._model.eval()

    @classmethod
    def from_release(cls, release_dir: str | Path, *, device: str = "auto") -> "RGPredictor":
        directory = Path(release_dir)
        config = load_release_config(directory / "release_config.json")
        checkpoint = directory / "checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RG-PCNet checkpoint not found: {checkpoint}")

        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        actual_hash = digest.hexdigest()
        if actual_hash != config.model_sha256:
            raise ValueError("checkpoint SHA-256 does not match release config")

        resolved = cls._resolve_device(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("RG-PCNet checkpoint must contain a mapping")
        raw_state = payload.get("model", payload.get("state_dict"))
        if not isinstance(raw_state, Mapping):
            raise ValueError("RG-PCNet checkpoint is missing model state dict")
        model_config = payload.get("model_config", {})
        if model_config is None:
            model_config = {}
        if not isinstance(model_config, Mapping):
            raise ValueError("RG-PCNet model_config must be a mapping")
        allowed = {"input_dim", "hidden_dim", "dropout"}
        unknown = set(model_config) - allowed
        if unknown:
            raise ValueError(f"unsupported RG-PCNet model config keys: {sorted(unknown)}")
        kwargs = dict(model_config)
        if int(kwargs.get("input_dim", FEATURE_DIM)) != FEATURE_DIM:
            raise ValueError(f"RG-PCNet input_dim must be {FEATURE_DIM}")
        try:
            model = RGPCNet(**kwargs).to(resolved)
            model.load_state_dict(raw_state)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("invalid RG-PCNet checkpoint or model configuration") from error
        return cls(model, config, device=str(resolved))

    @staticmethod
    def _resolve_device(requested: str):
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        try:
            return torch.device(requested)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"invalid inference device: {requested}") from error

    @staticmethod
    def _pose_arrays(window: DualWindow) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(window, DualWindow) or len(window.long) != 64:
            raise ValueError("RG-PCNet requires a DualWindow with exactly 64 long observations")
        pose = np.zeros((64, 17, 3), dtype=np.float32)
        timestamps = np.empty(64, dtype=np.float32)
        origin = window.long[0].timestamp.timestamp()
        for frame_index, observation in enumerate(window.long):
            timestamps[frame_index] = float(observation.timestamp.timestamp() - origin)
            for joint_index in range(min(17, len(observation.keypoints))):
                x, y = observation.keypoints[joint_index]
                confidence = observation.scores[joint_index] if joint_index < len(observation.scores) else 0.0
                pose[frame_index, joint_index] = (float(x), float(y), float(confidence))
        return pose, timestamps

    def _predict_tensors(self, window: DualWindow) -> tuple[float, tuple[float, float, float], float]:
        pose, timestamps = self._pose_arrays(window)
        features = build_temporal_features(pose, timestamps)
        feature_tensor = torch.as_tensor(features.values, dtype=torch.float32, device=self.device).unsqueeze(0)
        valid_tensor = torch.as_tensor(features.valid_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            output = self._model(feature_tensor, valid_tensor)
            fall_logit = output.window_fall_logit[0]
            fall_probability = float(torch.sigmoid(fall_logit / self._config.temperature).cpu())
            mask = output.valid_mask[0]
            phase_logits = output.phase_logits[0][mask]
            reliability_logits = output.reliability_logits[0][mask]
            coarse = torch.softmax(phase_logits.mean(dim=0), dim=-1).cpu().tolist()
            reliability = float(torch.sigmoid(reliability_logits.mean()).cpu())
        return fall_probability, tuple(float(value) for value in coarse), reliability

    def predict(self, window: DualWindow) -> PhaseModelOutput:
        fall_probability, coarse, reliability = self._predict_tensors(window)
        if len(coarse) != 3 or any(not np.isfinite(value) or value < 0.0 for value in coarse):
            raise ValueError("RG-PCNet phase probabilities must be finite and non-negative")
        total = sum(coarse)
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("RG-PCNet phase probabilities must have positive sum")
        coarse = tuple(value / total for value in coarse)
        phase_probs = (coarse[0], 0.0, coarse[1], 0.0, coarse[2], 0.0)
        decision = None if reliability < self._config.reliability_threshold else int(
            fall_probability >= self._config.fall_threshold
        )
        phase_index = max(range(6), key=phase_probs.__getitem__)
        return PhaseModelOutput(
            phase_probs=phase_probs,
            fall_event_prob=fall_probability,
            prefall_prob=coarse[1],
            recovery_prob=coarse[2],
            quality_score=reliability,
            embedding_version=self.embedding_version,
            model_version=self.model_version,
            phase=tuple(Phase)[phase_index],
            fall_decision=decision,
        )


__all__ = ["RGPredictor"]
