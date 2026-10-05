"""Publication-oriented controller comparison charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOTS = {
    "average_waiting_time": "Average waiting time (s / arrived vehicle)",
    "max_queue_length": "Maximum queue (vehicles)",
    "throughput": "Throughput (vehicles)",
    "average_queue_length": "Average total queue (vehicles)",
}


def create_comparison_plots(results: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create grouped mean charts with one-standard-deviation error bars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for metric, ylabel in PLOTS.items():
        means = results.pivot_table(index="scenario", columns="controller", values=metric)
        errors = results.pivot_table(
            index="scenario", columns="controller", values=metric, aggfunc="std"
        ).fillna(0.0)
        ax = means.plot(kind="bar", yerr=errors, capsize=3, figsize=(10, 5))
        ax.set_xlabel("Traffic scenario")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" (")[0] + " by controller")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=0)
        plt.tight_layout()
        path = output_dir / f"{metric}.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)
    return paths
