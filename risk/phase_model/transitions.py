"""Conservative phase transition constraints and temporal smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from .schema import Phase, PhaseModelOutput


_ALLOWED: dict[Phase, frozenset[Phase]] = {
    Phase.NORMAL_ADL: frozenset({Phase.NORMAL_ADL, Phase.PREFALL_ABNORMAL, Phase.DESCENDING}),
    Phase.PREFALL_ABNORMAL: frozenset({Phase.PREFALL_ABNORMAL, Phase.NORMAL_ADL, Phase.DESCENDING}),
    Phase.DESCENDING: frozenset({Phase.DESCENDING, Phase.IMPACT, Phase.FALLEN}),
    Phase.IMPACT: frozenset({Phase.IMPACT, Phase.FALLEN, Phase.RECOVERING}),
    Phase.FALLEN: frozenset({Phase.FALLEN, Phase.RECOVERING}),
    Phase.RECOVERING: frozenset({Phase.RECOVERING, Phase.NORMAL_ADL}),
}


def allowed_transition(previous: Phase, current: Phase) -> bool:
    return current in _ALLOWED[previous]


def transition_penalty(probabilities: Sequence[Sequence[float]]) -> float:
    """Return probability mass assigned to transitions forbidden by the matrix."""
    if len(probabilities) != len(Phase) or any(len(row) != len(Phase) for row in probabilities):
        raise ValueError("probabilities must be a square six-phase matrix")
    penalty = 0.0
    phases = tuple(Phase)
    for previous, row in zip(phases, probabilities):
        for current, probability in zip(phases, row):
            if not allowed_transition(previous, current):
                penalty += max(0.0, float(probability))
    return penalty


@dataclass(frozen=True)
class SmoothedPhase:
    phase: Phase
    timestamp: datetime
    prefall_warning: bool
    confirmed_fall: bool
    recovery_confirmed: bool = False


class PhaseSmoother:
    def __init__(self, *, min_persistence: int = 2) -> None:
        if min_persistence <= 0:
            raise ValueError("min_persistence must be positive")
        self.min_persistence = int(min_persistence)
        self._last_phase: Phase | None = None
        self._run_length = 0
        self._previous_phase: Phase | None = None
        self._seen_descending_or_impact = False
        self._recovering_run = 0

    def update(self, output: PhaseModelOutput, timestamp: datetime) -> SmoothedPhase:
        phase = output.phase or tuple(Phase)[max(range(len(Phase)), key=lambda index: output.phase_probs[index])]
        if self._last_phase == phase:
            self._run_length += 1
        else:
            self._previous_phase = self._last_phase
            self._last_phase = phase
            self._run_length = 1
        if phase in (Phase.DESCENDING, Phase.IMPACT):
            self._seen_descending_or_impact = True
        if phase is Phase.RECOVERING:
            self._recovering_run += 1
        else:
            self._recovering_run = 0
        prefall_warning = phase is Phase.PREFALL_ABNORMAL and self._run_length >= self.min_persistence
        confirmed_fall = (
            phase in (Phase.IMPACT, Phase.FALLEN)
            and self._run_length >= self.min_persistence
            and self._seen_descending_or_impact
        ) or (
            phase is Phase.IMPACT
            and self._previous_phase is Phase.DESCENDING
        )
        recovery_confirmed = (
            phase is Phase.NORMAL_ADL
            and self._previous_phase is Phase.RECOVERING
            and self._run_length >= self.min_persistence
        )
        return SmoothedPhase(phase, timestamp, prefall_warning, confirmed_fall, recovery_confirmed)
