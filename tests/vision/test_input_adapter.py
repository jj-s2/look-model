from __future__ import annotations

from collections.abc import Iterable
import subprocess
import sys

import pytest

from vision.input_adapter import EzvizStreamAdapter


FRAME = object()


class FakeCapture:
    def __init__(self, sequence: Iterable[tuple[bool, object | None]], opened: bool = True) -> None:
        self._sequence = iter(sequence)
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self._opened and not self.released

    def read(self) -> tuple[bool, object | None]:
        return next(self._sequence, (False, None))

    def release(self) -> None:
        self.released = True


class RaisingOpenCapture(FakeCapture):
    def isOpened(self) -> bool:
        raise RuntimeError("capture status unavailable")


class RaisingReleaseCapture(FakeCapture):
    def release(self) -> None:
        self.released = True
        raise RuntimeError("capture release unavailable")


class FakeCaptureFactory:
    def __init__(self) -> None:
        self.opened_urls: list[str] = []
        self.captures: list[FakeCapture] = []
        self._sequences: list[Iterable[tuple[bool, object | None]]] = []

    @property
    def release_count(self) -> int:
        return sum(capture.released for capture in self.captures)

    def with_sequences(self, sequences: list[Iterable[tuple[bool, object | None]]]) -> "FakeCaptureFactory":
        self._sequences = sequences
        return self

    def __call__(self, url: str) -> FakeCapture:
        self.opened_urls.append(url)
        sequence = self._sequences.pop(0) if self._sequences else []
        capture = FakeCapture(sequence)
        self.captures.append(capture)
        return capture


@pytest.fixture
def fake_capture_factory() -> FakeCaptureFactory:
    return FakeCaptureFactory()


def test_ezviz_adapter_refreshes_url_after_read_failure(fake_capture_factory: FakeCaptureFactory) -> None:
    """A failed capture read must discard its URL and return a frame from a refreshed stream."""
    urls = iter(["url-1", "url-2"])
    adapter = EzvizStreamAdapter(
        url_provider=lambda: next(urls),
        capture_factory=fake_capture_factory.with_sequences(
            [
                [(False, None)],
                [(True, FRAME)],
            ]
        ),
        backoff_seconds=(0.0,),
    )

    ok, frame = adapter.read()

    assert ok is True
    assert frame is FRAME
    assert fake_capture_factory.opened_urls == ["url-1", "url-2"]
    assert fake_capture_factory.release_count == 1
    assert adapter.health.state == "healthy"
    assert adapter.health.consecutive_failures == 0
    assert adapter.health.last_success_at is not None
    assert adapter.health.reason is None


def test_release_is_idempotent(fake_capture_factory: FakeCaptureFactory) -> None:
    """Closing a stream more than once must not double-release its capture."""
    adapter = EzvizStreamAdapter(
        lambda: "url", fake_capture_factory.with_sequences([[(True, FRAME)]])
    )
    assert adapter.read() == (True, FRAME)

    adapter.release()
    adapter.release()

    assert fake_capture_factory.release_count == 1
    assert adapter.health.state == "closed"


def test_ezviz_adapter_stops_after_bounded_retries(fake_capture_factory: FakeCaptureFactory) -> None:
    """An unavailable stream must return control after the configured retry budget."""
    adapter = EzvizStreamAdapter(
        url_provider=lambda: "rtsp://secret-stream",
        capture_factory=fake_capture_factory.with_sequences([[(False, None)]] * 3),
        max_retries=2,
        backoff_seconds=(0.0,),
    )

    assert adapter.read() == (False, None)

    assert len(fake_capture_factory.opened_urls) == 3
    assert adapter.health.state == "offline"
    assert adapter.health.consecutive_failures == 3
    assert "secret-stream" not in (adapter.health.reason or "")


def test_constructor_defers_capture_creation_until_read(fake_capture_factory: FakeCaptureFactory) -> None:
    """Constructing the adapter must not block on a network capture open."""
    adapter = EzvizStreamAdapter(lambda: "url", fake_capture_factory)

    assert fake_capture_factory.opened_urls == []
    assert adapter.health.state == "connecting"


def test_is_opened_error_becomes_offline_health() -> None:
    """Capture status-check errors must not escape the bounded read loop."""
    adapter = EzvizStreamAdapter(
        lambda: "rtsp://secret-stream",
        lambda _url: RaisingOpenCapture([]),
        max_retries=0,
    )

    assert adapter.read() == (False, None)

    assert adapter.health.state == "offline"
    assert adapter.health.consecutive_failures == 1
    assert adapter.health.reason == "capture_open_failed"


def test_release_error_becomes_offline_health() -> None:
    """A capture release error must not stop reconnection or leak its message."""
    adapter = EzvizStreamAdapter(
        lambda: "rtsp://secret-stream",
        lambda _url: RaisingReleaseCapture([(False, None)]),
        max_retries=0,
    )

    assert adapter.read() == (False, None)

    assert adapter.health.state == "offline"
    assert adapter.health.consecutive_failures == 1
    assert adapter.health.reason == "read_failed"


def test_explicit_release_suppresses_capture_release_error() -> None:
    """Lifecycle release must stay idempotent even if OpenCV release raises."""
    adapter = EzvizStreamAdapter(
        lambda: "rtsp://secret-stream",
        lambda _url: RaisingReleaseCapture([(True, FRAME)]),
    )
    assert adapter.read() == (True, FRAME)

    adapter.release()
    adapter.release()

    assert adapter.health.state == "closed"


def test_initial_open_failure_updates_health(fake_capture_factory: FakeCaptureFactory) -> None:
    """An unopened first capture must return control with explicit offline health."""
    capture = FakeCapture([], opened=False)
    adapter = EzvizStreamAdapter(lambda: "url", lambda _url: capture, max_retries=0)

    assert adapter.read() == (False, None)

    assert adapter.health.state == "offline"
    assert adapter.health.consecutive_failures == 1
    assert adapter.health.reason == "capture_open_failed"
    assert capture.released is True


def test_nonzero_backoff_is_called_between_failed_attempts(fake_capture_factory: FakeCaptureFactory) -> None:
    """Retries wait through the configured capped exponential sequence."""
    waits: list[float] = []
    adapter = EzvizStreamAdapter(
        lambda: "url",
        fake_capture_factory.with_sequences([[(False, None)]] * 3),
        max_retries=2,
        sleep=waits.append,
    )

    assert adapter.read() == (False, None)

    assert waits == [0.5, 1.0]
    assert len(fake_capture_factory.opened_urls) == 3


def test_input_adapter_imports_without_opencv_installed() -> None:
    """The fake-capture adapter must remain usable in an OpenCV-free test environment."""
    program = """
import builtins
import numpy
original_import = builtins.__import__
def import_without_cv2(name, *args, **kwargs):
    if name == 'cv2':
        raise ImportError('simulated missing cv2')
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_cv2
import vision.input_adapter
print('imported')
"""

    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"


def test_default_stream_capture_reports_missing_opencv_clearly() -> None:
    """Production capture must explain an absent optional OpenCV dependency."""
    program = """
import builtins
import numpy
original_import = builtins.__import__
def import_without_cv2(name, *args, **kwargs):
    if name == 'cv2':
        raise ImportError('simulated missing cv2')
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_cv2
from vision.input_adapter import EzvizStreamAdapter
try:
    EzvizStreamAdapter(lambda: 'rtsp://stream')
except RuntimeError as exc:
    print(exc)
"""

    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "OpenCV is required" in result.stdout
