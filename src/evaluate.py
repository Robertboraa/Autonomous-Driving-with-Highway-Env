# src/evaluate.py
# This script loads a saved checkpoint and records the agent playing.
# We run it 3 times (untrained, half-trained, fully trained) to get
# the footage we need for the evolution video.

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
import highway_env
import imageio
import numpy as np
from stable_baselines3 import DQN
import config


def record_agent(model_path: str, output_path: str, n_episodes: int = 3) -> None:
    """
    Loads a model checkpoint, runs it for n_episodes,
    and saves the frames as a GIF.
    """
    env = gym.make(config.ENV_NAME, render_mode="rgb_array")
    model = DQN.load(model_path, env=env)

    frames = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Render the current frame and store it
            frame = env.render()
            frames.append(frame)

            # Agent picks an action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        print(f"  Episode {episode + 1}: reward = {total_reward:.2f}")

    env.close()

    # Save all frames as a GIF
    imageio.mimsave(output_path, frames, fps=10)
    print(f"  Saved to {output_path}")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)

    print("Recording untrained agent...")
    record_agent("checkpoints/untrained_model", "assets/untrained.gif")

    print("Recording half-trained agent...")
    record_agent("checkpoints/rl_model_25000_steps", "assets/half_trained.gif")

    print("Recording fully trained agent...")
    record_agent("checkpoints/final_model", "assets/fully_trained.gif")

    print("All recordings done. Check the assets/ folder.")