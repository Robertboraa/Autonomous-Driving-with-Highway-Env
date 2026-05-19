# src/utils.py
# Helper functions for plotting and video creation.

import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List


def add_label_to_frame(frame: np.ndarray, label: str) -> np.ndarray:
    """
    Adds a text label banner at the top of a frame.
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Draw a dark banner at the top
    draw.rectangle([0, 0, img.width, 40], fill=(0, 0, 0))

    # Draw the label text
    draw.text((10, 10), label, fill=(255, 255, 255))

    return np.array(img)


def stitch_gifs(input_paths: List[str], output_path: str) -> None:
    """
    Combines multiple GIFs into one sequential GIF with stage labels.
    """
    all_frames = []
    labels = ["STAGE 1: UNTRAINED", "STAGE 2: HALF-TRAINED", "STAGE 3: FULLY TRAINED"]

    for path, label in zip(input_paths, labels):
        frames = imageio.mimread(path)
        print(f"  {label}: {len(frames)} frames loaded from {path}")

        for frame in frames:
            labeled_frame = add_label_to_frame(frame, label)
            all_frames.append(labeled_frame)

    imageio.mimsave(output_path, all_frames, fps=10, loop=0)
    print(f"Evolution GIF saved to {output_path}")


def plot_rewards(rewards: List[float], save_path: str = "assets/reward_plot.png") -> None:
    """
    Plots raw and smoothed reward over training episodes.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.4, color="steelblue", label="Raw reward")

    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(smoothed, color="steelblue", linewidth=2, label=f"Smoothed (window={window})")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Reward plot saved to {save_path}")