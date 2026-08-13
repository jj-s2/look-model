"""Small, reproducible PA-DTSF training configuration for the RTX 4060 target."""

RELEASE_ID = "padtfs-v1-seed42"
SEED = 42
SHORT_DIM = 512
JOINTS = 17
HIDDEN_DIM = 128
SHORT_FRAMES = 48
SHORT_FPS = 10.0
LONG_FRAMES = 64
LONG_FPS = 2.0
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
