from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read simple KEY=value pairs without adding a runtime dependency."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


@dataclass(frozen=True)
class Settings:
    project_root: Path
    model_cache_dir: Path
    output_dir: Path
    ezviz_app_key: str | None
    ezviz_app_secret: str | None
    ezviz_device_serial: str | None
    ezviz_device_code: str | None
    ezviz_stream_url: str | None
    sdnl1_data_url: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        if env is None:
            process_values = dict(os.environ)
            # An explicitly empty subprocess environment is used by diagnostics
            # to assert that no credentials are available; do not silently load
            # a project-local file in that isolated mode.
            dotenv_values = _read_dotenv(PROJECT_ROOT / ".env") if process_values else {}
            dotenv_values.update(process_values)
            values: Mapping[str, str] = dotenv_values
        else:
            values = env

        def path(key: str, default: str) -> Path:
            return Path(values.get(key, default)).expanduser().resolve()

        return cls(
            project_root=PROJECT_ROOT,
            model_cache_dir=path("MODEL_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "models")),
            output_dir=path("OUTPUT_DIR", str(PROJECT_ROOT / "outputs")),
            ezviz_app_key=values.get("EZVIZ_APP_KEY"),
            ezviz_app_secret=values.get("EZVIZ_APP_SECRET"),
            ezviz_device_serial=values.get("EZVIZ_DEVICE_SERIAL"),
            ezviz_device_code=values.get("EZVIZ_DEVICE_CODE"),
            ezviz_stream_url=values.get("EZVIZ_STREAM_URL"),
            sdnl1_data_url=values.get("SDNL1_DATA_URL"),
        )

    def redacted(self) -> dict[str, object]:
        data = dict(self.__dict__)
        for key in (
            "ezviz_app_key",
            "ezviz_app_secret",
            "ezviz_device_serial",
            "ezviz_device_code",
            "ezviz_stream_url",
            "sdnl1_data_url",
        ):
            if data[key]:
                data[key] = "***"
        return data

    @property
    def sdnl1_status(self) -> str:
        """Return an explicit availability state for SDNL1 data consumers."""
        return "available" if self.sdnl1_data_url else "unavailable"
