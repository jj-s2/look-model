from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        values = os.environ if env is None else env

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
        ):
            if data[key]:
                data[key] = "***"
        return data
