from datetime import datetime, timedelta, timezone

from risk.pmcc.chains import TemporalChainBuilder
from risk.pmcc.schema import ChangeEvent


START = datetime(2026, 7, 10, 9, tzinfo=timezone.utc)


def event(feature, offset, *, quality=1.0, persistence=1.0):
    return ChangeEvent(
        subject_id="resident-3",
        occurred_at=START + timedelta(hours=offset),
        feature=feature,
        previous_value=1.0,
        current_value=2.0,
        confidence=quality,
        provenance={"event_id": f"{feature}-{offset}", "persistence": persistence, "quality": quality},
    )


def test_chain_retains_node_ids_gaps_and_association_boundary():
    chains = TemporalChainBuilder().build((
        event("sleep_duration_minutes", 0, persistence=0.8),
        event("step_count", 24, quality=0.5, persistence=0.9),
    ))

    assert len(chains) == 1
    chain = chains[0]
    assert chain.provenance["association_only"] is True
    assert chain.provenance["node_ids"] == ("sleep_duration_minutes-0", "step_count-24")
    assert chain.provenance["gap_hours"] == (24.0,)
    assert 0.0 < chain.provenance["score"] < 0.5


def test_chain_rejects_reverse_order_and_gaps_over_seventy_two_hours():
    builder = TemporalChainBuilder()

    assert builder.build((event("step_count", 0), event("sleep_duration_minutes", 1))) == ()
    assert builder.build((event("sleep_duration_minutes", 0), event("step_count", 73))) == ()
