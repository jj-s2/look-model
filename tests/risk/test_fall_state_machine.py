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
        obs(3, p=0.9, angle=86, ground=0.9),
        obs(4, p=0.9, angle=86, ground=0.9),
    ]
    recovery = [obs(5, p=0.1, angle=10, ground=0.1), obs(6, p=0.1, angle=10, ground=0.1)]
    second_fall = [
        obs(7, p=0.8, angle=45, speed=1.1, ground=0.3),
        obs(8, p=0.9, angle=85, speed=1.3, ground=0.8),
        obs(9, p=0.9, angle=86, ground=0.9),
        obs(10, p=0.9, angle=86, ground=0.9),
        obs(11, p=0.9, angle=86, ground=0.9),
    ]

    decisions = [machine.update(item) for item in first_fall + recovery + second_fall]

    assert decisions[6].state == "recovered"
    assert sum(item.confirmed_fall for item in decisions) == 2


def test_invalid_observations_degrade_instead_of_raising():
    """Removing update-boundary validation would let upstream malformed data crash monitoring."""
    machine = make_machine()
    base = {
        "timestamp": 0,
        "fall_probability": 0.0,
        "gait_risk": 0.0,
        "torso_angle_deg": 0.0,
        "vertical_speed": 0.0,
        "ground_ratio": 0.0,
        "keypoint_quality": 1.0,
    }

    def unchecked(**overrides: object) -> FallObservation:
        malformed = object.__new__(FallObservation)
        for name, value in (base | overrides).items():
            object.__setattr__(malformed, name, value)
        return malformed

    decisions = [
        machine.update(None),
        machine.update({}),
        machine.update(unchecked(fall_probability=float("nan"))),
        machine.update(unchecked(fall_probability=1.01)),
        machine.update(unchecked(gait_risk="high")),
    ]

    assert all(item.data_quality == "degraded" for item in decisions)
    assert all(item.confirmed_fall is False for item in decisions)
    assert all("invalid_observation" in item.reasons for item in decisions)


def test_low_quality_interrupts_unconfirmed_descent_before_ground_evidence():
    """Removing state reset would reuse descent evidence across a quality outage."""
    machine = make_machine()
    sequence = [
        obs(0, p=0.8, angle=45, speed=1.1, ground=0.3),
        obs(1, p=0.9, angle=85, speed=1.3, ground=0.8),
        obs(2, p=0.9, angle=85, ground=0.9, quality=0.1),
        obs(3, p=0.9, angle=85, ground=0.9),
        obs(4, p=0.9, angle=85, ground=0.9),
        obs(5, p=0.9, angle=85, ground=0.9),
    ]

    decisions = [machine.update(item) for item in sequence]

    assert decisions[2].state == "normal"
    assert not any(item.confirmed_fall for item in decisions)


def test_low_quality_interrupts_unconfirmed_on_ground_before_ground_evidence():
    """Keeping an unconfirmed on-ground state would reuse evidence after a quality outage."""
    machine = make_machine()
    sequence = [
        obs(0, p=0.8, angle=45, speed=1.1, ground=0.3),
        obs(1, p=0.9, angle=85, speed=1.3, ground=0.8),
        obs(2, p=0.9, angle=85, ground=0.9),
        obs(3, p=0.9, angle=85, ground=0.9, quality=0.1),
        obs(4, p=0.9, angle=85, ground=0.9),
        obs(5, p=0.9, angle=85, ground=0.9),
        obs(6, p=0.9, angle=85, ground=0.9),
    ]

    decisions = [machine.update(item) for item in sequence]

    assert decisions[2].state == "on_ground"
    assert decisions[3].state == "normal"
    assert not any(item.confirmed_fall for item in decisions)


def test_ground_persistence_starts_after_entering_on_ground():
    """Counting ground frames during descent would confirm one observation too early."""
    machine = make_machine()
    sequence = [
        obs(0, p=0.8, angle=45, speed=1.1, ground=0.3),
        obs(1, p=0.9, angle=85, speed=1.3, ground=0.8),
        obs(2, p=0.9, angle=85, ground=0.9),
        obs(3, p=0.9, angle=85, ground=0.9),
        obs(4, p=0.9, angle=85, ground=0.9),
    ]

    decisions = [machine.update(item) for item in sequence]

    assert decisions[2].state == "on_ground"
    assert decisions[2].evidence["ground_frames"] == 0
    assert decisions[3].confirmed_fall is False
    assert decisions[4].confirmed_fall is True


def test_ground_posture_without_prior_descent_never_confirms():
    """Removing the ordered descent transition would alert on a person already lying down."""
    machine = make_machine()

    decisions = [machine.update(obs(t, p=0.9, angle=85, ground=0.9)) for t in range(5)]

    assert not any(item.confirmed_fall for item in decisions)
    assert all(item.state != "on_ground" for item in decisions)


def test_default_threshold_requires_consecutive_high_risk_observations():
    """Reducing the default hysteresis would enter unstable state after one noisy sample."""
    machine = FallStateMachine()

    first = machine.update(obs(0, p=0.8, angle=45))
    second = machine.update(obs(1, p=0.8, angle=45))

    assert first.state == "normal"
    assert second.state == "unstable"
