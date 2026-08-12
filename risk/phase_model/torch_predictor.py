"""Checkpoint-backed PyTorch predictor for the phase-risk service."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .calibration import Calibrator
from .model import PhaseAwareFusionModel
from .normalization import normalize_pose_array
from .release import ReleaseBundle, load_release_bundle, release_inference_config
from .schema import Phase, PhaseModelOutput
from .windows import DualWindow


class TorchPhasePredictor:
    """Adapt the released long-pose checkpoint to ``PhaseModelOutput``.

    Optionally applies validation-only temperature scaling and a decision
    threshold loaded from the release's ``calibration.json``.
    """

    _REQUIRED_STATE_KEYS = {
        "long_branch",
        "short_projection",
        "long_projection",
        "phase_head",
        "fall_head",
        "prefall_head",
        "recovery_head",
        "abstain_head",
    }

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str = "auto",
        model_version: str | None = None,
        embedding_version: str = "short_embedding_unavailable",
        temperature: float = 1.0,
        threshold: float | None = None,
        release_id: str | None = None,
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as error:  # pragma: no cover - dependency guard
            raise RuntimeError("PyTorch is required for TorchPhasePredictor") from error
        self._torch = torch
        self.device = self._resolve_device(device)
        path = Path(checkpoint)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"phase-model checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("phase-model checkpoint must contain a mapping")
        raw_state = payload.get("model")
        if not isinstance(raw_state, Mapping):
            raise ValueError("phase-model checkpoint is missing a model state dict")
        missing = self._REQUIRED_STATE_KEYS.difference(raw_state.keys())
        if missing:
            raise ValueError(f"phase-model state dict is missing keys: {sorted(missing)}")
        self.model_version = model_version or str(payload.get("release_id") or release_id or "phase-model")
        if not self.model_version.strip():
            raise ValueError("model_version must be non-empty")
        if not embedding_version.strip():
            raise ValueError("embedding_version must be non-empty")
        self.embedding_version = embedding_version
        self._model = PhaseAwareFusionModel(short_dim=512, joints=17, hidden_dim=128).to(self.device)
        self._load_state_dict(raw_state)
        self._model.eval()
        self.short_branch_quality = 0.0
        self._release_id = release_id
        self._calibrator = Calibrator(
            temperature=float(temperature),
            threshold=float(threshold) if threshold is not None else 0.5,
        )

    @classmethod
    def from_release(
        cls,
        release_dir: Path,
        *,
        device: str = "auto",
        temperature: float | None = None,
        threshold: float | None = None,
    ) -> "TorchPhasePredictor":
        """Load a predictor from a release directory, using its calibration."""
        bundle = load_release_bundle(release_dir)
        config = release_inference_config(bundle)
        checkpoint_path = release_dir / "checkpoint.pt"
        return cls(
            checkpoint_path,
            device=device,
            model_version=bundle.release_id,
            temperature=float(temperature if temperature is not None else config["temperature"]),
            threshold=float(threshold if threshold is not None else config["threshold"]),
            release_id=bundle.release_id,
        )

    @property
    def release_id(self) -> str | None:
        return self._release_id

    @property
    def temperature(self) -> float:
        return float(self._calibrator.temperature)

    @property
    def threshold(self) -> float:
        return float(self._calibrator.threshold)

    def _resolve_device(self, requested: str):
        torch = self._torch
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        try:
            return torch.device(requested)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"invalid inference device: {requested}") from error

    def _load_state_dict(self, raw_state: Mapping[str, object]) -> None:
        modules = {
            "long_branch": self._model.long_branch.network,
            "short_projection": self._model.short_projection,
            "long_projection": self._model.long_projection,
            "phase_head": self._model.phase_head,
            "fall_head": self._model.fall_head,
            "prefall_head": self._model.prefall_head,
            "recovery_head": self._model.recovery_head,
            "abstain_head": self._model.abstain_head,
        }
        for name, module in modules.items():
            try:
                module.load_state_dict(raw_state[name])
            except (KeyError, TypeError, RuntimeError) as error:
                raise ValueError(f"invalid phase-model state dict for {name}") from error

    def predict(self, window: DualWindow) -> PhaseModelOutput:
        if not isinstance(window, DualWindow) or len(window.long) != 64:
            raise ValueError("phase predictor requires a DualWindow with exactly 64 long observations")
        pose, quality_score = self._pose_tensor(window)
        torch = self._torch
        short = torch.zeros((1, 512), dtype=torch.float32, device=self.device)
        short_quality = torch.zeros((1,), dtype=torch.float32, device=self.device)
        long_quality = torch.ones((1,), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = self._model(short, pose, short_quality, long_quality)
            phase_probs = torch.softmax(output.phase_logits, dim=-1)[0].detach().cpu().tolist()
            fall_logit = float(output.fall_event_logit[0].detach().cpu())
            fall_prob = float(self._calibrator.calibrate([fall_logit])[0])
            prefall_prob = float(torch.sigmoid(output.prefall_logit)[0].detach().cpu())
            recovery_prob = float(torch.sigmoid(output.recovery_logit)[0].detach().cpu())
        phase_index = max(range(len(phase_probs)), key=phase_probs.__getitem__)
        return PhaseModelOutput(
            phase_probs=tuple(float(value) for value in phase_probs),
            fall_event_prob=fall_prob,
            prefall_prob=prefall_prob,
            recovery_prob=recovery_prob,
            quality_score=quality_score,
            embedding_version=self.embedding_version,
            model_version=self.model_version,
            phase=tuple(Phase)[phase_index],
            fall_decision=self._calibrator.decide([fall_prob])[0],
        )

    def _pose_tensor(self, window: DualWindow):
        import numpy as np

        raw = np.zeros((64, 17, 3), dtype=np.float32)
        frame_scores: list[float] = []
        for frame_index, observation in enumerate(window.long):
            scores = []
            for joint_index, point in enumerate(observation.keypoints[:17]):
                if len(point) != 2:
                    continue
                score = float(observation.scores[joint_index]) if joint_index < len(observation.scores) else 0.0
                if not np.isfinite(score):
                    score = 0.0
                score = max(0.0, min(1.0, score))
                x, y = float(point[0]), float(point[1])
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                raw[frame_index, joint_index] = (x, y, score)
                scores.append(score)
            frame_scores.append(sum(scores) / len(scores) if scores else 0.0)
        normalized = normalize_pose_array(raw)
        tensor = self._torch.as_tensor(normalized, dtype=self._torch.float32, device=self.device).unsqueeze(0)
        quality = sum(frame_scores) / len(frame_scores) if frame_scores else 0.0
        return tensor, max(0.0, min(1.0, float(quality)))


__all__ = ["TorchPhasePredictor"]
