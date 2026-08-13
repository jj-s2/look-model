"""Optional research-only multimodal shadow assessment for voluntary check-ins.

The module deliberately does not ship clinical weights.  Callers may inject a
reviewed text/audio predictor in experiments; without one, the result abstains.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Callable, Mapping

from mental.contracts import ModalityEvidence, VoluntaryCheckin


def _probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite probability")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be a finite probability")
    return result


@dataclass(frozen=True)
class ShadowModelConfig:
    text_model_name: str = "hfl/chinese-macbert-base"
    audio_feature_set: str = "eGeMAPSv02"
    model_version: str = "research_shadow_v1"
    min_text_chars: int = 4
    min_asr_confidence: float = 0.7
    min_audio_quality: float = 0.5
    max_modality_disagreement: float = 0.4
    required_modalities: tuple[str, ...] = ("text",)
    mode: str = "research_shadow"
    promoted: bool = False

    def __post_init__(self) -> None:
        for name in ("text_model_name", "audio_feature_set", "model_version", "mode"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if isinstance(self.min_text_chars, bool) or not isinstance(self.min_text_chars, int) or self.min_text_chars < 1:
            raise ValueError("min_text_chars must be a positive integer")
        for name in ("min_asr_confidence", "min_audio_quality", "max_modality_disagreement"):
            value = _probability(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if isinstance(self.required_modalities, (str, bytes)):
            raise ValueError("required_modalities must be a sequence")
        modalities = tuple(self.required_modalities)
        if not modalities or any(item not in {"text", "audio"} for item in modalities) or len(set(modalities)) != len(modalities):
            raise ValueError("required_modalities must contain unique text/audio values")
        object.__setattr__(self, "required_modalities", modalities)
        if self.mode != "research_shadow" or self.promoted is not False:
            raise ValueError("shadow models must remain research_shadow and unpromoted")


@dataclass(frozen=True)
class ShadowAssessment:
    session_id: str
    subject_alias: str
    combined_score: float | None
    coverage: float
    uncertainty: float
    evidence_domains: tuple[str, ...]
    evidence_codes: tuple[str, ...]
    counterfactual_stability: float | None
    model_version: str
    model_mode: str
    abstention_reasons: tuple[str, ...] = ()
    modality_evidence: tuple[ModalityEvidence, ...] = ()
    promoted: bool = False
    research_only: bool = True

    @property
    def abstained(self) -> bool:
        return bool(self.abstention_reasons) or self.combined_score is None


TextPredictor = Callable[[str], float]
AudioPredictor = Callable[[VoluntaryCheckin], float]


class VoluntaryShadowModel:
    """Run injected MacBERT/eGeMAPS-compatible predictors with abstention."""

    def __init__(
        self,
        config: ShadowModelConfig | None = None,
        *,
        text_predictor: TextPredictor | None = None,
        audio_predictor: AudioPredictor | None = None,
    ) -> None:
        self.config = config or ShadowModelConfig()
        self.text_predictor = text_predictor
        self.audio_predictor = audio_predictor

    def assess(self, checkin: VoluntaryCheckin) -> ShadowAssessment:
        if not isinstance(checkin, VoluntaryCheckin):
            raise ValueError("checkin must be VoluntaryCheckin")
        reasons: list[str] = []
        if checkin.model_version not in {"unavailable", self.config.model_version}:
            reasons.append("model_version_mismatch")
        text = (checkin.transcript or " ".join(checkin.answers.values())).strip()
        text_score: float | None = None
        audio_score: float | None = None
        modality_evidence: list[ModalityEvidence] = []
        if "text" in self.config.required_modalities:
            if len(text) < self.config.min_text_chars:
                reasons.append("text_too_short")
            elif self.text_predictor is None:
                reasons.append("research_model_unavailable")
            else:
                try:
                    text_score = _probability(self.text_predictor(text), "text_prediction")
                except (TypeError, ValueError, OverflowError):
                    reasons.append("text_prediction_invalid")
        elif self.text_predictor is not None and len(text) >= self.config.min_text_chars:
            try:
                text_score = _probability(self.text_predictor(text), "text_prediction")
            except (TypeError, ValueError, OverflowError):
                reasons.append("text_prediction_invalid")

        audio_present = checkin.audio_quality is not None or checkin.asr_confidence is not None
        if "audio" in self.config.required_modalities:
            if not audio_present:
                reasons.append("audio_missing")
            elif checkin.asr_confidence is not None and checkin.asr_confidence < self.config.min_asr_confidence:
                reasons.append("asr_confidence_low")
            elif checkin.audio_quality is not None and checkin.audio_quality < self.config.min_audio_quality:
                reasons.append("audio_quality_low")
            elif self.audio_predictor is None:
                reasons.append("research_model_unavailable")
            else:
                try:
                    audio_score = _probability(self.audio_predictor(checkin), "audio_prediction")
                except (TypeError, ValueError, OverflowError):
                    reasons.append("audio_prediction_invalid")
        elif audio_present and self.audio_predictor is not None:
            if checkin.asr_confidence is not None and checkin.asr_confidence < self.config.min_asr_confidence:
                reasons.append("asr_confidence_low")
            elif checkin.audio_quality is not None and checkin.audio_quality < self.config.min_audio_quality:
                reasons.append("audio_quality_low")
            else:
                try:
                    audio_score = _probability(self.audio_predictor(checkin), "audio_prediction")
                except (TypeError, ValueError, OverflowError):
                    reasons.append("audio_prediction_invalid")

        if text_score is not None:
            modality_evidence.append(ModalityEvidence("text", text_score, True, 0.9, 0.1, ("macbert",), checkin.session_id))
        if audio_score is not None:
            modality_evidence.append(ModalityEvidence("audio", audio_score, True, 0.8, 0.2, ("egemaps",), checkin.session_id))
        if reasons:
            return self._result(checkin, None, 0.0, 1.0, modality_evidence, reasons)
        scores = [score for score in (text_score, audio_score) if score is not None]
        if not scores:
            return self._result(checkin, None, 0.0, 1.0, modality_evidence, ["research_model_unavailable"])
        if text_score is not None and audio_score is not None and abs(text_score - audio_score) > self.config.max_modality_disagreement:
            return self._result(checkin, None, 1.0, 1.0, modality_evidence, ["modalities_conflict"])
        stability = 1.0
        if text_score is not None and self.text_predictor is not None:
            try:
                deleted = _probability(self.text_predictor(""), "text_counterfactual")
                stability = max(0.0, 1.0 - abs(text_score - deleted))
            except (TypeError, ValueError, OverflowError):
                stability = 0.0
        score = sum(scores) / len(scores)
        reliability = sum(item.reliability for item in modality_evidence) / len(modality_evidence)
        return self._result(checkin, score, len(scores) / 2.0, min(1.0, 1.0 - reliability), modality_evidence, (), stability)

    def _result(
        self,
        checkin: VoluntaryCheckin,
        score: float | None,
        coverage: float,
        uncertainty: float,
        modality_evidence: list[ModalityEvidence],
        reasons: list[str] | tuple[str, ...],
        stability: float | None = None,
    ) -> ShadowAssessment:
        return ShadowAssessment(
            session_id=checkin.session_id,
            subject_alias=checkin.subject_alias,
            combined_score=score,
            coverage=coverage,
            uncertainty=uncertainty,
            evidence_domains=("voluntary_episode",) if modality_evidence else (),
            evidence_codes=tuple(item.evidence_codes[0] for item in modality_evidence),
            counterfactual_stability=stability,
            model_version=self.config.model_version,
            model_mode=self.config.mode,
            abstention_reasons=tuple(dict.fromkeys(reasons)),
            modality_evidence=tuple(modality_evidence),
            promoted=False,
            research_only=True,
        )
