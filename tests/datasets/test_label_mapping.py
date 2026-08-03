from __future__ import annotations

import pytest

from datasets.unified.labels import map_source_label


@pytest.mark.parametrize(
    "dataset,label,phase,coarse,hard_negative",
    [
        ("GMDCSA24", "ADL", "normal_adl", "adl", None),
        ("GMDCSA24", "Falling (FW)", None, "fall", None),
        ("Pre-VFall", "Abnormal", "prefall_abnormal", None, None),
        ("Pre-VFall", "Fall", None, "fall", None),
        ("CAUCAFall", "sitting", "normal_adl", "adl", "sit_down"),
        ("UP-Fall-3D-Skeletons", "impact", "impact", "fall", None),
        ("NTU120", "A42", "prefall_abnormal", None, None),
        ("NTU120", "A80", "normal_adl", "adl", "squat"),
    ],
)
def test_source_mapping(
    dataset: str,
    label: str,
    phase: str | None,
    coarse: str | None,
    hard_negative: str | None,
) -> None:
    mapped = map_source_label(dataset, label, {})

    assert (mapped.phase, mapped.coarse_event, mapped.hard_negative) == (
        phase,
        coarse,
        hard_negative,
    )


def test_fall_coarse_does_not_claim_a_fine_phase() -> None:
    mapped = map_source_label("GMDCSA24", "Falling", {})

    assert mapped.phase is None
    assert mapped.supervision_mask == ("fall_event",)


def test_unknown_label_is_masked_instead_of_guessed() -> None:
    mapped = map_source_label("GMDCSA24", "UnspecifiedMotion", {"impact_sec": None})

    assert mapped.phase is None
    assert "phase" not in mapped.supervision_mask
    assert mapped.mapping_reason == "unmapped_source_label"


def test_label_mapping_normalizes_case_and_whitespace() -> None:
    mapped = map_source_label("prevfall", " abnormal ", {})

    assert mapped.phase == "prefall_abnormal"
