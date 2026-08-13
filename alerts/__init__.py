"""Local, deduplicated alert delivery."""

from .dispatcher import AlertDispatcher, DispatchResult

__all__ = ["AlertDispatcher", "DispatchResult"]
