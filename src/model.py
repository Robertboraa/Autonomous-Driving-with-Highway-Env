# src/model.py
# This file creates the highway environment and builds the DQN agent.
# Keeping this separate from train.py means we can swap algorithms
# or environments later without touching the training logic.

import gymnasium as gym
import highway_env  # registers the highway environments with gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import sys
import os
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
# This lets us import config.py from the src/ folder
sys.path.insert(0, os.path.dirname(__file__))
import config


def make_env() -> gym.Env:
    """
    Creates and returns the highway-v0 environment.
    The observation type is set to Kinematics — this means the agent
    sees a matrix of nearby vehicles (position, speed, heading)
    rather than raw pixels. Much faster to train on CPU.
    """
    env = gym.make(config.ENV_NAME, render_mode=None)
    return env


def build_model(env: gym.Env) -> DQN:
    """
    Builds and returns the DQN model.
    DQN works by maintaining a replay buffer of past experiences
    and learning from random samples of that buffer — this breaks
    the correlation between consecutive steps and stabilizes training.
    """
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        batch_size=config.BATCH_SIZE,
        buffer_size=config.BUFFER_SIZE,
        verbose=1,
        tensorboard_log=config.LOG_DIR,
        device=device,            # ← uses M3 GPU
    )
    return model