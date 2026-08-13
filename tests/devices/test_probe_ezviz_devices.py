from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _probe_module():
    spec = importlib.util.spec_from_file_location("probe_ezviz_devices", PROJECT_ROOT / "scripts" / "probe_ezviz_devices.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_fixture_reports_unavailable_device_without_full_serial(tmp_path: Path) -> None:
    fixture = tmp_path / "offline-devices.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "deviceSerial": "ABCDEF1234",
                    "deviceName": "C6C ABCDEF1234",
                    "deviceType": "CS-C6C",
                    "status": 2,
                    "cameraNum": 1,
                    "supportTalk": 1,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/probe_ezviz_devices.py", "--offline-fixture", str(fixture)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "unavailable:" in result.stdout
    assert "model=CS-C6C" in result.stdout
    assert "talk=full_duplex" in result.stdout
    assert "serial=***1234" in result.stdout
    assert "ABCDEF1234" not in result.stdout


def test_offline_fixture_never_calls_injected_session(tmp_path: Path) -> None:
    fixture = tmp_path / "offline-devices.json"
    fixture.write_text(json.dumps([]), encoding="utf-8")

    class FakeSession:
        calls = 0

        def post(self, *args: object, **kwargs: object) -> object:
            self.calls += 1
            raise AssertionError("offline fixture must not make HTTP calls")

    session = FakeSession()
    assert _probe_module().main(["--offline-fixture", str(fixture)], session=session) == 0
    assert session.calls == 0


def test_short_serial_is_fully_masked(capsys: object) -> None:
    from devices.models import EzvizDevice

    device = EzvizDevice("1234", "CS-C6C", False, 1, "unknown", {})
    module = _probe_module()
    module._print_device(device)
    output = capsys.readouterr().out

    assert "1234" not in output
    assert "serial=***" in output


def test_live_report_labels_inventory_as_live_evidence(tmp_path: Path) -> None:
    from devices.models import EzvizDevice

    report = tmp_path / "live-report.md"
    module = _probe_module()
    module._write_report(report, [EzvizDevice("ABCDEF1234", "CS-C6C", True, 1, "unknown", {})], fixture=False)

    text = report.read_text(encoding="utf-8")

    assert "**live inventory**" in text
    assert "generated from an offline fixture" not in text
    assert "live inventory; no live stream" in text


def test_probe_without_credentials_names_missing_variables_only() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/probe_ezviz_devices.py"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={},
    )

    assert result.returncode == 2
    assert "EZVIZ_APP_KEY" in result.stderr
    assert "EZVIZ_APP_SECRET" in result.stderr
