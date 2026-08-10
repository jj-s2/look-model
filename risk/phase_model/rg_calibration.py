"""Bounded validation-only temperature calibration for RG-PCNet.

Non-finite fit logits disable calibration. For finite audit metrics, infinities
retain their sign at a bounded logit of ``+/-30`` and NaNs use the wrong-sign
bound for their aligned label. This conservative convention keeps evidence
distinguishable without allowing invalid data to appear favorable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


_MIN_TEMPERATURE = 0.5
_MAX_TEMPERATURE = 5.0
_TEMPERATURE_CANDIDATE_COUNT = 181
_IMPROVEMENT_EPSILON = 1e-6
_NONFINITE_AUDIT_LOGIT = 30.0


class _ImmutableClassCounts(dict[str, int]):
    """A snapshotted dict that retains dict compatibility without mutation."""

    __slots__ = ("_initialized",)

    def __init__(self, *args: object, **kwargs: object) -> None:
        if getattr(self, "_initialized", False):
            self._immutable()
        dict.__init__(self, *args, **kwargs)
        object.__setattr__(self, "_initialized", True)

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("class_counts is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self) -> _ImmutableClassCounts:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _ImmutableClassCounts:
        memo[id(self)] = self
        return self

    def __reduce_ex__(self, _protocol: int) -> tuple[object, tuple[dict[str, int]]]:
        return type(self), (dict(self),)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _softplus(value: float) -> float:
    if value >= 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def _temperature_candidates() -> tuple[float, ...]:
    """Return the exact deterministic 181-point log grid in [0.5, 5.0]."""
    log_ratio = math.log(_MAX_TEMPERATURE / _MIN_TEMPERATURE)
    candidates = [
        _MIN_TEMPERATURE
        * math.exp(log_ratio * index / (_TEMPERATURE_CANDIDATE_COUNT - 1))
        for index in range(_TEMPERATURE_CANDIDATE_COUNT)
    ]
    candidates[0] = _MIN_TEMPERATURE
    candidates[-1] = _MAX_TEMPERATURE
    return tuple(candidates)


def _metrics(
    logits: Sequence[float], labels: Sequence[int], temperature: float
) -> tuple[float, float]:
    sample_count = len(logits)
    nll = 0.0
    brier = 0.0
    for logit, label in zip(logits, labels):
        scaled_logit = logit / temperature
        loss = _softplus(-scaled_logit) if label == 1 else _softplus(scaled_logit)
        probability = _sigmoid(scaled_logit)
        nll += loss / sample_count
        brier += (probability - label) ** 2 / sample_count
    return nll, brier


def _audit_logit(value: float, label: int) -> float:
    """Map a non-finite logit to a finite, evidence-preserving audit bound."""
    if math.isnan(value):
        return -_NONFINITE_AUDIT_LOGIT if label == 1 else _NONFINITE_AUDIT_LOGIT
    if value > 0.0:
        return _NONFINITE_AUDIT_LOGIT
    if value < 0.0:
        return -_NONFINITE_AUDIT_LOGIT
    return value


@dataclass(frozen=True)
class CalibrationArtifact:
    temperature: float
    enabled: bool
    reason: str
    sample_count: int
    class_counts: dict[str, int]
    nll_before: float
    nll_after: float
    brier_before: float
    brier_after: float
    split_hash: str

    def __getattribute__(self, name: str) -> object:
        if name == "class_counts":
            try:
                count0, count1 = object.__getattribute__(self, "_class_counts_values")
            except AttributeError:
                return object.__getattribute__(self, name)
            return _ImmutableClassCounts({"0": count0, "1": count1})
        return object.__getattribute__(self, name)

    def __post_init__(self) -> None:
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or not _MIN_TEMPERATURE <= temperature <= _MAX_TEMPERATURE:
            raise ValueError("temperature must be finite and in [0.5, 5.0]")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        if type(self.sample_count) is not int or self.sample_count < 0:
            raise ValueError("sample_count must be a non-negative integer")
        if not isinstance(self.split_hash, str) or not self.split_hash.strip():
            raise ValueError("split_hash must be a non-empty string")

        class_counts = dict(self.class_counts)
        if set(class_counts) != {"0", "1"}:
            raise ValueError("class_counts must contain exactly '0' and '1'")
        if any(type(value) is not int or value < 0 for value in class_counts.values()):
            raise ValueError("class_counts values must be non-negative integers")
        if sum(class_counts.values()) != self.sample_count:
            raise ValueError("class_counts must sum to sample_count")

        metric_names = ("nll_before", "nll_after", "brier_before", "brier_after")
        try:
            metrics = tuple(float(getattr(self, name)) for name in metric_names)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("calibration metrics must be finite numbers") from exc
        if any(not math.isfinite(value) for value in metrics):
            raise ValueError("calibration metrics must be finite numbers")

        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(
            self, "_class_counts_values", (class_counts["0"], class_counts["1"])
        )
        object.__delattr__(self, "class_counts")
        for name, value in zip(metric_names, metrics):
            object.__setattr__(self, name, value)

    def calibrate(self, logits: Sequence[float]) -> list[float]:
        normalized_logits: list[float] = []
        for value in logits:
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("calibration logits must be finite numbers") from exc
            if not math.isfinite(normalized):
                raise ValueError("calibration logits must be finite numbers")
            normalized_logits.append(normalized)
        return [_sigmoid(value / self.temperature) for value in normalized_logits]


def _artifact(
    *,
    temperature: float,
    enabled: bool,
    reason: str,
    labels: Sequence[int],
    class_counts: dict[str, int],
    nll_before: float,
    nll_after: float,
    brier_before: float,
    brier_after: float,
    split_hash: str,
) -> CalibrationArtifact:
    return CalibrationArtifact(
        temperature=temperature,
        enabled=enabled,
        reason=reason,
        sample_count=len(labels),
        class_counts=class_counts,
        nll_before=nll_before,
        nll_after=nll_after,
        brier_before=brier_before,
        brier_after=brier_after,
        split_hash=split_hash,
    )


def fit_bounded_temperature(
    logits: Sequence[float], labels: Sequence[int], *, split_hash: str
) -> CalibrationArtifact:
    """Fit a bounded temperature and enable it only with metric evidence."""
    if len(logits) == 0 or len(logits) != len(labels):
        raise ValueError("logits and labels must be non-empty and have the same length")
    if not isinstance(split_hash, str) or not split_hash.strip():
        raise ValueError("split_hash must be a non-empty string")

    normalized_labels: list[int] = []
    for value in labels:
        if isinstance(value, (str, bytes)):
            raise ValueError("labels must be binary values 0 or 1")
        try:
            numeric_label = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("labels must be binary values 0 or 1") from exc
        if not math.isfinite(numeric_label) or numeric_label not in (0.0, 1.0):
            raise ValueError("labels must be binary values 0 or 1")
        normalized_labels.append(int(numeric_label))

    normalized_logits: list[float] = []
    for value in logits:
        try:
            normalized_logits.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError("logits must contain numeric values") from exc

    class_counts = {
        "0": normalized_labels.count(0),
        "1": normalized_labels.count(1),
    }

    if any(not math.isfinite(value) for value in normalized_logits):
        safe_logits = [
            value if math.isfinite(value) else _audit_logit(value, label)
            for value, label in zip(normalized_logits, normalized_labels)
        ]
        identity_nll, identity_brier = _metrics(safe_logits, normalized_labels, 1.0)
        return _artifact(
            temperature=1.0,
            enabled=False,
            reason="calibration data contains non-finite logits",
            labels=normalized_labels,
            class_counts=class_counts,
            nll_before=identity_nll,
            nll_after=identity_nll,
            brier_before=identity_brier,
            brier_after=identity_brier,
            split_hash=split_hash,
        )

    nll_before, brier_before = _metrics(normalized_logits, normalized_labels, 1.0)
    if len(set(normalized_labels)) < 2:
        return _artifact(
            temperature=1.0,
            enabled=False,
            reason="calibration requires both classes",
            labels=normalized_labels,
            class_counts=class_counts,
            nll_before=nll_before,
            nll_after=nll_before,
            brier_before=brier_before,
            brier_after=brier_before,
            split_hash=split_hash,
        )

    best_temperature = 1.0
    best_nll = nll_before
    best_brier = brier_before
    best_key = (nll_before, brier_before, 0.0, 1.0)
    for temperature in _temperature_candidates():
        candidate_nll, candidate_brier = _metrics(
            normalized_logits, normalized_labels, temperature
        )
        if not math.isfinite(candidate_nll) or not math.isfinite(candidate_brier):
            continue
        candidate_key = (
            candidate_nll,
            candidate_brier,
            abs(math.log(temperature)),
            temperature,
        )
        if candidate_key < best_key:
            best_temperature = temperature
            best_nll = candidate_nll
            best_brier = candidate_brier
            best_key = candidate_key

    improved = (
        best_nll < nll_before - _IMPROVEMENT_EPSILON
        or best_brier < brier_before - _IMPROVEMENT_EPSILON
    )
    if not improved:
        return _artifact(
            temperature=1.0,
            enabled=False,
            reason="calibration did not improve nll or brier",
            labels=normalized_labels,
            class_counts=class_counts,
            nll_before=nll_before,
            nll_after=nll_before,
            brier_before=brier_before,
            brier_after=brier_before,
            split_hash=split_hash,
        )

    return _artifact(
        temperature=best_temperature,
        enabled=True,
        reason="calibration improved nll or brier",
        labels=normalized_labels,
        class_counts=class_counts,
        nll_before=nll_before,
        nll_after=best_nll,
        brier_before=brier_before,
        brier_after=best_brier,
        split_hash=split_hash,
    )
