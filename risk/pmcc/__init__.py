"""Uncertainty-aware PMCC data contracts."""

from .schema import (
    BaselineState,
    ChangeEvent,
    DailyObservation,
    DecisionBand,
    EvidenceTier,
    OutcomeFeedback,
    OutcomeType,
    PMCCForecast,
    TemporalChain,
)

__all__ = [
    "BaselineState",
    "ChangeEvent",
    "DailyObservation",
    "DecisionBand",
    "EvidenceTier",
    "OutcomeFeedback",
    "OutcomeType",
    "PMCCForecast",
    "TemporalChain",
    "OptionalTCNSurvivalModel",
    "RuleSurvivalCalibrator",
    "SurvivalRiskModel",
]


def __getattr__(name: str):
    """Expose optional PMCC model contracts without importing ML extras."""
    if name in {"OptionalTCNSurvivalModel", "RuleSurvivalCalibrator", "SurvivalRiskModel"}:
        from .survival import OptionalTCNSurvivalModel, RuleSurvivalCalibrator, SurvivalRiskModel

        return {
            "OptionalTCNSurvivalModel": OptionalTCNSurvivalModel,
            "RuleSurvivalCalibrator": RuleSurvivalCalibrator,
            "SurvivalRiskModel": SurvivalRiskModel,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
