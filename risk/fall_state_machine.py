"""Explainable, dependency-free fall-event state machine.

The machine deliberately separates a short high-risk movement from a confirmed
fall.  A confirmation needs ordered evidence: descent followed by persistent
ground posture.  It latches one event until consecutive standing observations
show recovery, preventing repeated alerts for the same incident.
"""

from dataclasses import dataclass, field
from numbers import Real
from typing import Literal


FallState = Literal["normal", "unstable", "descending", "on_ground", "recovered"]


@dataclass(frozen=True)
class FallThresholds:
    """All tunable state-machine thresholds, expressed per observation."""

    fall_probability_high: float = 0.75
    gait_risk_high: float = 0.65
    torso_angle_unstable_deg: float = 35.0
    torso_angle_descending_deg: float = 65.0
    descending_speed_min: float = 0.8
    ground_ratio_min: float = 0.7
    keypoint_quality_min: float = 0.5
    unstable_evidence_frames: int = 2
    descending_evidence_frames: int = 1
    ground_persistence_frames: int = 2
    recovery_evidence_frames: int = 2
    recovery_probability_max: float = 0.35
    recovery_gait_risk_max: float = 0.35
    recovery_torso_angle_max_deg: float = 30.0
    recovery_ground_ratio_max: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "fall_probability_high",
            "gait_risk_high",
            "ground_ratio_min",
            "keypoint_quality_min",
            "recovery_probability_max",
            "recovery_gait_risk_max",
            "recovery_ground_ratio_max",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "torso_angle_unstable_deg",
            "torso_angle_descending_deg",
            "descending_speed_min",
            "recovery_torso_angle_max_deg",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")
        for name in (
            "unstable_evidence_frames",
            "descending_evidence_frames",
            "ground_persistence_frames",
            "recovery_evidence_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class FallObservation:
    """Per-frame fall-related evidence supplied by upstream vision analytics."""

    timestamp: object
    fall_probability: float
    gait_risk: float
    torso_angle_deg: float
    vertical_speed: float
    ground_ratio: float
    keypoint_quality: float

    def __post_init__(self) -> None:
        for name in ("fall_probability", "gait_risk", "ground_ratio", "keypoint_quality"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("torso_angle_deg", "vertical_speed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a number")


@dataclass(frozen=True)
class FallDecision:
    """A traceable output for one observation."""

    state: FallState
    confirmed_fall: bool
    data_quality: Literal["usable", "degraded"]
    reasons: tuple[str, ...]
    evidence: dict[str, object]


@dataclass
class FallStateMachine:
    """Finite-state fall detector with quality gating, persistence, and rearming."""

    thresholds: FallThresholds = field(default_factory=FallThresholds)
    _state: FallState = field(default="normal", init=False)
    _event_latched: bool = field(default=False, init=False)
    _unstable_frames: int = field(default=0, init=False)
    _descending_frames: int = field(default=0, init=False)
    _ground_frames: int = field(default=0, init=False)
    _recovery_frames: int = field(default=0, init=False)

    @property
    def state(self) -> FallState:
        return self._state

    def update(self, observation: FallObservation) -> FallDecision:
        """Consume one observation and return its state, reason codes, and evidence."""
        if observation.keypoint_quality < self.thresholds.keypoint_quality_min:
            self._clear_evidence()
            return self._decision(
                confirmed_fall=False,
                data_quality="degraded",
                reasons=("low_keypoint_quality",),
                observation=observation,
            )

        unstable = self._is_unstable(observation)
        descending = self._is_descending(observation, unstable)
        on_ground = self._is_on_ground(observation)
        recovered = self._is_recovered(observation)
        self._unstable_frames = self._next_count(unstable, self._unstable_frames)
        self._descending_frames = self._next_count(descending, self._descending_frames)
        self._ground_frames = self._next_count(on_ground, self._ground_frames)
        self._recovery_frames = self._next_count(recovered, self._recovery_frames)
        reasons: list[str] = []
        confirmed_fall = False

        if self._state in ("normal", "recovered"):
            if self._unstable_frames >= self.thresholds.unstable_evidence_frames:
                self._state = "unstable"
                reasons.append("high_risk_persistence")
            if self._state == "unstable" and self._descending_frames >= self.thresholds.descending_evidence_frames:
                self._state = "descending"
                reasons.append("descent_evidence")
        elif self._state == "unstable":
            if self._descending_frames >= self.thresholds.descending_evidence_frames:
                self._state = "descending"
                reasons.append("descent_evidence")
            elif not unstable:
                self._state = "normal"
                reasons.append("risk_cleared")
        elif self._state == "descending":
            if on_ground:
                self._state = "on_ground"
                reasons.append("ground_posture")
            elif not unstable:
                self._state = "normal"
                reasons.append("descent_interrupted")
        elif self._state == "on_ground":
            if recovered and self._recovery_frames >= self.thresholds.recovery_evidence_frames:
                self._state = "recovered"
                self._event_latched = False
                reasons.append("recovery_persistence")

        if self._state == "on_ground" and on_ground:
            reasons.append("ground_persistence")
            if self._ground_frames >= self.thresholds.ground_persistence_frames and not self._event_latched:
                confirmed_fall = True
                self._event_latched = True
                reasons.append("fall_confirmed")

        if not reasons:
            reasons.append("monitoring")
        return self._decision(
            confirmed_fall=confirmed_fall,
            data_quality="usable",
            reasons=tuple(reasons),
            observation=observation,
        )

    def _is_unstable(self, observation: FallObservation) -> bool:
        return (
            observation.fall_probability >= self.thresholds.fall_probability_high
            or observation.gait_risk >= self.thresholds.gait_risk_high
        ) and observation.torso_angle_deg >= self.thresholds.torso_angle_unstable_deg

    def _is_descending(self, observation: FallObservation, unstable: bool) -> bool:
        return (
            unstable
            and observation.torso_angle_deg >= self.thresholds.torso_angle_descending_deg
            and observation.vertical_speed >= self.thresholds.descending_speed_min
        )

    def _is_on_ground(self, observation: FallObservation) -> bool:
        return (
            observation.fall_probability >= self.thresholds.fall_probability_high
            and observation.torso_angle_deg >= self.thresholds.torso_angle_descending_deg
            and observation.ground_ratio >= self.thresholds.ground_ratio_min
        )

    def _is_recovered(self, observation: FallObservation) -> bool:
        return (
            observation.fall_probability <= self.thresholds.recovery_probability_max
            and observation.gait_risk <= self.thresholds.recovery_gait_risk_max
            and observation.torso_angle_deg <= self.thresholds.recovery_torso_angle_max_deg
            and observation.ground_ratio <= self.thresholds.recovery_ground_ratio_max
        )

    @staticmethod
    def _next_count(signal: bool, current: int) -> int:
        return current + 1 if signal else 0

    def _clear_evidence(self) -> None:
        self._unstable_frames = 0
        self._descending_frames = 0
        self._ground_frames = 0
        self._recovery_frames = 0

    def _decision(
        self,
        *,
        confirmed_fall: bool,
        data_quality: Literal["usable", "degraded"],
        reasons: tuple[str, ...],
        observation: FallObservation,
    ) -> FallDecision:
        return FallDecision(
            state=self._state,
            confirmed_fall=confirmed_fall,
            data_quality=data_quality,
            reasons=reasons,
            evidence={
                "timestamp": observation.timestamp,
                "fall_probability": observation.fall_probability,
                "gait_risk": observation.gait_risk,
                "torso_angle_deg": observation.torso_angle_deg,
                "vertical_speed": observation.vertical_speed,
                "ground_ratio": observation.ground_ratio,
                "keypoint_quality": observation.keypoint_quality,
                "unstable_frames": self._unstable_frames,
                "descending_frames": self._descending_frames,
                "ground_frames": self._ground_frames,
                "recovery_frames": self._recovery_frames,
                "event_latched": self._event_latched,
            },
        )
