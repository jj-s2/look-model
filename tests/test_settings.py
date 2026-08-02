from pathlib import Path

from core.settings import Settings


def test_project_root_is_derived_from_repository():
    settings = Settings.from_env({})
    assert (settings.project_root / "paths.py").exists()


def test_sensitive_values_are_redacted():
    settings = Settings.from_env({
        "EZVIZ_APP_KEY": "app-key",
        "EZVIZ_APP_SECRET": "app-secret",
        "EZVIZ_DEVICE_SERIAL": "device-serial",
        "EZVIZ_DEVICE_CODE": "device-code",
        "EZVIZ_STREAM_URL": "rtsp://camera.example/live?token=stream-token",
        "SDNL1_DATA_URL": "https://data.example/feed?signature=data-token",
    })
    exported = str(settings.redacted())
    for secret in (
        "app-key",
        "app-secret",
        "device-serial",
        "device-code",
        "stream-token",
        "data-token",
    ):
        assert secret not in exported
    assert "***" in exported


def test_explicit_cache_path_overrides_default(tmp_path: Path):
    settings = Settings.from_env({"MODEL_CACHE_DIR": str(tmp_path)})
    assert settings.model_cache_dir == tmp_path.resolve()


def test_sdnl1_status_is_explicit_when_not_configured():
    assert Settings.from_env({}).sdnl1_status == "unavailable"


def test_sdnl1_status_is_available_when_data_url_is_configured():
    settings = Settings.from_env({"SDNL1_DATA_URL": "https://data.example/feed"})
    assert settings.sdnl1_status == "available"
