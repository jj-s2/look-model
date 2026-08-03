from __future__ import annotations

import pytest

from risk.pmcc.schema import EvidenceTier
from risk.pmcc.uncertainty import UncertaintyGate


def _members(value: float = 0.2) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(value for _ in range(7)) for _ in range(5))


@pytest.mark.parametrize(("coverage", "quality", "reason"), ((0.49, 1.0, "insufficient_coverage"), (1.0, 0.49, "insufficient_quality")))
def test_gate_abstains_for_low_observation_reliability(coverage: float, quality: float, reason: str) -> None:
    result = UncertaintyGate().evaluate(_members(), coverage, quality, EvidenceTier.REAL_PUBLIC, False)
    assert result.abstained is True
    assert reason in result.reasons


def test_gate_abstains_for_wide_72_hour_interval() -> None:
    members = ((0.0,) * 7,) * 2 + ((1.0,) * 7,) * 3
    result = UncertaintyGate().evaluate(members, 1.0, 1.0, EvidenceTier.REAL_PUBLIC, False)
    assert result.width_72h > 0.35
    assert "wide_72h_interval" in result.reasons


@pytest.mark.parametrize("tier", (EvidenceTier.SYNTHETIC_RESEARCH, EvidenceTier.OFFLINE_FIXTURE))
def test_release_mode_abstains_for_non_release_evidence(tier: EvidenceTier) -> None:
    result = UncertaintyGate().evaluate(_members(), 1.0, 1.0, tier, True)
    assert result.abstained is True
    assert "non_release_evidence" in result.reasons


def test_gate_requires_at_least_five_bootstrap_members() -> None:
    with pytest.raises(ValueError, match="five"):
        UncertaintyGate().evaluate(_members()[:4], 1.0, 1.0, EvidenceTier.REAL_PUBLIC, False)
