from datetime import datetime, timedelta, timezone
import os

from storage.clip_buffer import CircularClipBuffer
from storage.retention import RetentionPolicy


NOW = datetime(2026, 8, 2, 9, tzinfo=timezone.utc)
FRAME = [[1, 2], [3, 4]]


def test_clip_is_not_written_when_recording_opt_in_is_false(tmp_path) -> None:
    buffer = CircularClipBuffer(tmp_path, recording_opt_in=False, pre_seconds=10)

    buffer.on_frame(FRAME, NOW)

    assert buffer.confirm_event("event-1") is None
    assert list(tmp_path.iterdir()) == []


def test_confirmed_clip_contains_ten_seconds_before_and_twenty_after(tmp_path) -> None:
    buffer = CircularClipBuffer(tmp_path, recording_opt_in=True, pre_seconds=10)
    buffer.on_frame("before", NOW - timedelta(seconds=10))
    buffer.on_frame("event", NOW)

    assert buffer.confirm_event("event-1", post_seconds=20) is None
    buffer.on_frame("after", NOW + timedelta(seconds=20))
    clip = buffer.confirm_event("event-1")

    assert clip is not None
    assert clip.exists()


def test_retention_removes_only_expired_event_clips(tmp_path) -> None:
    expired = tmp_path / "event_expired.npz"
    current = tmp_path / "event_current.npz"
    unrelated = tmp_path / "notes.txt"
    for path in (expired, current, unrelated):
        path.write_text("fixture", encoding="utf-8")
    os.utime(expired, (NOW.timestamp() - 8 * 86400, NOW.timestamp() - 8 * 86400))
    os.utime(current, (NOW.timestamp() - 6 * 86400, NOW.timestamp() - 6 * 86400))

    removed = RetentionPolicy(tmp_path, keep_days=7).prune(NOW)

    assert removed == [expired]
    assert current.exists()
    assert unrelated.exists()
