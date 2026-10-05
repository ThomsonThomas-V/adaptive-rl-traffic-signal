"""Plot Monitor and EvalCallback learning curves from one or more training runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("results/plots/training_curves.png"))
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for run in args.runs:
        monitor_path = run / "monitor.csv"
        monitor = pd.read_csv(monitor_path, comment="#")
        steps = monitor["l"].cumsum()
        label = run.name
        axes[0].plot(steps, monitor["r"].rolling(args.window, min_periods=1).mean(), label=label)
        if "average_waiting_time" in monitor:
            axes[1].plot(
                steps,
                monitor["average_waiting_time"].rolling(args.window, min_periods=1).mean(),
                label=label,
            )
        evaluations = run / "evaluations.npz"
        if evaluations.exists():
            data = np.load(evaluations)
            axes[2].plot(data["timesteps"], data["results"].mean(axis=1), label=label)
    axes[0].set_title("Rolling episode reward")
    axes[0].set_ylabel("Reward")
    axes[1].set_title("Rolling average waiting time")
    axes[1].set_ylabel("Seconds / arrived vehicle")
    axes[2].set_title("Deterministic evaluation reward")
    for axis in axes:
        axis.set_xlabel("Training steps")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
