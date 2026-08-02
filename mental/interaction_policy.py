"""Low-burden prompting policy with a fixed local quiet period."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class InteractionContext:
    now: datetime | str
    sustained_change: bool = False
    confirmed_fall: bool = False
    fall_check_already_sent: bool = False
    last_full_screening: datetime | str | None = None
    last_short_checkin: datetime | str | None = None
    last_full_invitation: datetime | str | None = None
    last_short_invitation: datetime | str | None = None
    user_initiated_full_gds: bool = False
    user_initiated_short_checkin: bool = False


@dataclass(frozen=True)
class InteractionDecision:
    prompt: str | None
    reason: str
    invite_full_gds: bool = False
    invite_short_checkin: bool = False


class InteractionPolicy:
    FULL_COOLDOWN = timedelta(days=28)
    SHORT_COOLDOWN = timedelta(days=7)

    def evaluate(self, context: InteractionContext) -> InteractionDecision:
        now = self._timestamp(context.now)
        if context.confirmed_fall:
            if not context.fall_check_already_sent:
                return InteractionDecision("fall_confirmation", "confirmed_fall")
            return InteractionDecision(None, "fall_confirmation_already_sent")
        if context.user_initiated_full_gds:
            return InteractionDecision("full_gds", "user_initiated")
        if context.user_initiated_short_checkin:
            return InteractionDecision("short_checkin", "user_initiated")
        if self._quiet_hours(now):
            return InteractionDecision(None, "quiet_hours")
        full_allowed = self._all_cooldowns_expired(
            now, (context.last_full_screening, context.last_full_invitation), self.FULL_COOLDOWN
        )
        short_allowed = self._all_cooldowns_expired(
            now, (context.last_short_checkin, context.last_short_invitation), self.SHORT_COOLDOWN
        )
        if context.sustained_change and short_allowed:
            return InteractionDecision("short_checkin", "sustained_change", invite_short_checkin=True)
        return InteractionDecision(None, "cooldown" if context.sustained_change and not short_allowed else "no_invitation", invite_full_gds=full_allowed)

    @staticmethod
    def _timestamp(value: datetime | str) -> datetime:
        timestamp = datetime.fromisoformat(value) if isinstance(value, str) else value
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return timestamp

    def _cooldown_expired(self, now: datetime, value: datetime | str | None, cooldown: timedelta) -> bool:
        return value is None or now - self._timestamp(value) >= cooldown

    def _all_cooldowns_expired(
        self, now: datetime, values: tuple[datetime | str | None, ...], cooldown: timedelta
    ) -> bool:
        return all(self._cooldown_expired(now, value, cooldown) for value in values)

    @staticmethod
    def _quiet_hours(now: datetime) -> bool:
        return now.hour >= 21 or now.hour < 8
