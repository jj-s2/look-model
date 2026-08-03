"""Public dataset adapters that emit UnifiedClip records."""

from .base import DatasetAdapter
from .caucafall import CAUCAFallAdapter
from .gmdcsa24 import GMDCSA24Adapter
from .omnifall import OmniFallAdapter
from .prevfall import PreVFallAdapter
from .upfall3d import UPFall3DAdapter


def get_adapter(name: str) -> DatasetAdapter:
    key = name.strip().lower().replace("-", "").replace("_", "")
    adapters = {
        "gmdcsa24": GMDCSA24Adapter,
        "prevfall": PreVFallAdapter,
        "caucafall": CAUCAFallAdapter,
        "upfall3d": UPFall3DAdapter,
        "upfall3dskeletons": UPFall3DAdapter,
        "omnifall": OmniFallAdapter,
    }
    try:
        return adapters[key]()
    except KeyError as error:
        raise ValueError(f"unknown dataset adapter: {name}") from error


__all__ = ["DatasetAdapter", "get_adapter"]
