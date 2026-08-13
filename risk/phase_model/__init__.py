"""Phase-aware fall-risk contracts and optional model implementations.

The package root intentionally does not import the PyTorch model so that
schema, buffering, and quality logic remain usable on lightweight devices.
"""

from .schema import Phase, PhaseModelOutput, PoseObservation

__all__ = ["Phase", "PhaseModelOutput", "PoseObservation"]
