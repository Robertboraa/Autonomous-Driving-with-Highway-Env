# src/train.py
# Main training script. Trains the DQN agent and saves:
# - 3 checkpoints (untrained, half, final) for the evolution video
# - A rewards log for plotting the training graph

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import config
from model import make_env, build_model
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback


class RewardLoggerCallback(BaseCallback):
    """
    Custom callback that records the total reward
    at the end of every episode during training.
    We save this to a file so we can plot it later.
    """
    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: list = []
        self.current_reward: float = 0.0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
        return True

    def save(self, path: str) -> None:
        np.save(path, np.array(self.episode_rewards))
        print(f"Reward log saved to {path}")


def train() -> None:
    env = make_env()
    print(f"Training on: {config.ENV_NAME}")
    print(f"Total timesteps: {config.TOTAL_TIMESTEPS}")

    model = build_model(env)

    # Save untrained model (stage 1)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    model.save(f"{config.CHECKPOINT_DIR}untrained_model")
    print("Saved untrained checkpoint.")

    # Save checkpoints at 50% and 100%
    checkpoint_cb = CheckpointCallback(
        save_freq=config.TOTAL_TIMESTEPS // 2,
        save_path=config.CHECKPOINT_DIR,
        name_prefix="rl_model",
        verbose=1,
    )

    # Log rewards every episode
    reward_cb = RewardLoggerCallback()

    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, reward_cb],
        progress_bar=True,
    )

    model.save(f"{config.CHECKPOINT_DIR}final_model")
    reward_cb.save("logs/rewards.npy")
    print("Training complete. Final model saved.")

    env.close()


if __name__ == "__main__":
    train()