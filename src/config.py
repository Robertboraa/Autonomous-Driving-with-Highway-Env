# src/config.py

# Environment
ENV_NAME: str = "highway-v0"

# Training
TOTAL_TIMESTEPS: int = 200_000
LEARNING_RATE: float = 5e-4
GAMMA: float = 0.99
BATCH_SIZE: int = 32
BUFFER_SIZE: int = 5_000

# Checkpoints
CHECKPOINT_DIR: str = "checkpoints/"
LOG_DIR: str = "logs/"

# Evaluation
N_EVAL_EPISODES: int = 5