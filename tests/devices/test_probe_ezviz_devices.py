from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_offline_fixture_reports_unavailable_device_without_full_serial(tmp_path: Path) -> None:
    fixture = tmp_path / "offline-devices.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "deviceSerial": "ABCDEF1234",
                    "deviceName": "C6C",
                    "status": 0,
                    "channelNumber": 1,
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
    assert "model=C6C" in result.stdout
    assert "talk=full_duplex" in result.stdout
    assert "serial=***1234" in result.stdout
    assert "ABCDEF1234" not in result.stdout


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
