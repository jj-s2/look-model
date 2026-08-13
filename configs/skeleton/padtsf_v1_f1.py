"""PA-DTSF v1 F1-optimized configuration.

Extends ``padtfs_v1.py`` with evaluation-time probability calibration
and threshold-search settings used by the inner 3-fold search.
"""

from configs.skeleton.padtfs_v1 import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_DIM,
    JOINTS,
    LEARNING_RATE,
    LONG_FPS,
    LONG_FRAMES,
    RELEASE_ID,
    SEED,
    SHORT_DIM,
    SHORT_FPS,
    SHORT_FRAMES,
    WEIGHT_DECAY,
)

# Inference / calibration
ENABLE_PROBABILITY: bool = True
TEMPERATURE_SCALING: bool = True

# Inner threshold search on the validation split
INNER_FOLDS: int = 3
INNER_RECALL_FLOOR: float = 0.75
INNER_MIN_RECALL_PER_SUBJECT: float = 0.60
INNER_FPR_CEILING: float | None = None
INNER_THRESHOLD_AGGREGATION: str = "mean"  # "mean" or "median"

# Promotion gate for the optimized checkpoint
PROMOTION_MIN_SUBJECT_MACRO_F1: float = 0.78
PROMOTION_MIN_RECALL: float = 0.75
PROMOTION_MIN_WORST_SUBJECT_RECALL: float = 0.60
PROMOTION_MAX_ECE: float = 0.10
PROMOTION_MAX_BRIER: float = 0.20
