from pathlib import Path

from core.settings import Settings


def test_project_root_is_derived_from_repository():
    settings = Settings.from_env({})
    assert (settings.project_root / "paths.py").exists()


def test_secret_values_are_redacted():
    settings = Settings.from_env({
        "EZVIZ_APP_KEY": "app-key",
        "EZVIZ_APP_SECRET": "app-secret",
    })
    exported = str(settings.redacted())
    assert "app-key" not in exported
    assert "app-secret" not in exported
    assert "***" in exported


def test_explicit_cache_path_overrides_default(tmp_path: Path):
    settings = Settings.from_env({"MODEL_CACHE_DIR": str(tmp_path)})
    assert settings.model_cache_dir == tmp_path.resolve()
