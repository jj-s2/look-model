from risk.fall_state_machine import FallObservation, FallStateMachine, FallThresholds


def obs(
    timestamp: float,
    *,
    p: float = 0.0,
    gait: float = 0.0,
    angle: float = 0.0,
    speed: float = 0.0,
    ground: float = 0.0,
    quality: float = 1.0,
) -> FallObservation:
    return FallObservation(
        timestamp=timestamp,
        fall_probability=p,
        gait_risk=gait,
        torso_angle_deg=angle,
        vertical_speed=speed,
        ground_ratio=ground,
        keypoint_quality=quality,
    )


def make_machine() -> FallStateMachine:
    return FallStateMachine(
        FallThresholds(
            unstable_evidence_frames=1,
            descending_evidence_frames=1,
            ground_persistence_frames=2,
            recovery_evidence_frames=2,
        )
    )


def test_brief_bending_does_not_emit_fall_event():
    """Removing descent evidence must keep an ordinary bend unconfirmed."""
    machine = make_machine()

    decisions = [
        machine.update(obs(t, p=0.2, angle=70, speed=0.1, ground=0.2))
        for t in range(6)
    ]

    assert not any(item.confirmed_fall for item in decisions)
    assert decisions[-1].state == "normal"


def test_descent_followed_by_ground_persistence_emits_once():
    """Removing the event latch would allow persistent ground evidence to re-emit."""
    machine = make_machine()
    sequence = [
        obs(0, p=0.25, angle=10, speed=0.1, ground=0.1),
        obs(1, p=0.80, angle=45, speed=1.1, ground=0.3),
        obs(2, p=0.92, angle=80, speed=1.4, ground=0.8),
        obs(3, p=0.90, angle=85, speed=0.1, ground=0.9),
        obs(4, p=0.88, angle=86, speed=0.0, ground=0.9),
        obs(5, p=0.90, angle=87, speed=0.0, ground=0.9),
    ]

    decisions = [machine.update(item) for item in sequence]

    assert [item.state for item in decisions[:4]] == [
        "normal",
        "unstable",
        "descending",
        "on_ground",
    ]
    assert sum(item.confirmed_fall for item in decisions) == 1
    assert decisions[-1].state == "on_ground"
    assert "ground_persistence" in decisions[-1].reasons
    assert decisions[-1].evidence["ground_frames"] >= 2


def test_low_keypoint_quality_cannot_confirm_fall():
    """Removing quality gating would turn unreliable pose data into an alert."""
    machine = make_machine()

    decision = machine.update(obs(0, p=0.99, angle=90, speed=2.0, ground=1.0, quality=0.1))

    assert decision.confirmed_fall is False
    assert decision.data_quality == "degraded"
    assert "low_keypoint_quality" in decision.reasons


def test_recovery_rearms_machine_for_a_later_fall():
    """Removing recovery rearming would suppress the second independent fall."""
    machine = make_machine()
    first_fall = [
        obs(0, p=0.8, angle=45, speed=1.1, ground=0.3),
        obs(1, p=0.9, angle=85, speed=1.3, ground=0.8),
        obs(2, p=0.9, angle=86, ground=0.9),
    ]
    recovery = [obs(3, p=0.1, angle=10, ground=0.1), obs(4, p=0.1, angle=10, ground=0.1)]
    second_fall = [
        obs(5, p=0.8, angle=45, speed=1.1, ground=0.3),
        obs(6, p=0.9, angle=85, speed=1.3, ground=0.8),
        obs(7, p=0.9, angle=86, ground=0.9),
    ]

    decisions = [machine.update(item) for item in first_fall + recovery + second_fall]

    assert decisions[4].state == "recovered"
    assert sum(item.confirmed_fall for item in decisions) == 2
