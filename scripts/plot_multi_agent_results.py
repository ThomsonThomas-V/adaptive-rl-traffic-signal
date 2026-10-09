"""Create publication-ready charts from the multi-scenario grid evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "shared_dqn": "Shared DQN",
    "queue_heuristic": "Queue heuristic",
    "fixed_time": "Fixed time",
    "random": "Random",
}
COLORS = {
    "shared_dqn": "#35d49a",
    "queue_heuristic": "#66a7ff",
    "fixed_time": "#f2b35d",
    "random": "#9b8afb",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "results" / "multi_agent_evaluation_300k.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "plots_300k"
    )
    parser.add_argument(
        "--sumo-input", type=Path, default=ROOT / "results" / "sumo_policy_evaluation_300k.json"
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    scenarios = list(data["scenarios"])
    controllers = ("shared_dqn", "queue_heuristic", "fixed_time", "random")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric, title, ylabel, filename in (
        ("vehicle_average_wait", "Vehicle waiting time", "Seconds", "vehicle_wait.png"),
        ("overflows", "Queue spillback / rejected demand", "Vehicles", "overflows.png"),
        ("max_queue", "Maximum approach queue", "Vehicles", "max_queue.png"),
    ):
        figure, axis = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
        x = np.arange(len(scenarios))
        width = 0.19
        for index, controller in enumerate(controllers):
            summaries = [
                data["scenarios"][scenario]["controllers"][controller]["metrics"][metric]
                for scenario in scenarios
            ]
            means = np.asarray([item["mean"] for item in summaries])
            errors = np.asarray(
                [item["ci95_high"] - item["mean"] for item in summaries]
            )
            axis.bar(
                x + (index - 1.5) * width,
                means,
                width,
                yerr=errors,
                capsize=3,
                label=LABELS[controller],
                color=COLORS[controller],
                alpha=0.9,
            )
        axis.set_title(f"{title} across held-out stress scenarios")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [name.replace("_", " ").title() for name in scenarios], rotation=20)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
        figure.savefig(args.output_dir / filename, dpi=180)
        plt.close(figure)

    if args.sumo_input.exists():
        sumo = json.loads(args.sumo_input.read_text(encoding="utf-8"))
        metrics = ("mean_queue", "peak_queue", "mean_waiting_seconds")
        labels = ("Mean queue", "Peak queue", "Waiting load")
        native = np.asarray([sumo["summary_mean"]["native"][metric] for metric in metrics])
        dqn = np.asarray([sumo["summary_mean"]["shared_dqn"][metric] for metric in metrics])
        figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
        x = np.arange(len(metrics))
        axis.bar(x - 0.18, np.full(3, 100.0), 0.36, label="Native", color=COLORS["fixed_time"])
        axis.bar(x + 0.18, 100 * dqn / native, 0.36, label="Shared DQN", color=COLORS["shared_dqn"])
        axis.axhline(100, color="#66758c", linewidth=1)
        axis.set_xticks(x, labels)
        axis.set_ylabel("Percent of native controller (lower is better)")
        axis.set_title("Held-out SUMO Lemgo comparison · five seeds")
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
        figure.savefig(args.output_dir / "sumo_comparison.png", dpi=180)
        plt.close(figure)

    print(f"Saved evaluation charts to {args.output_dir}")


if __name__ == "__main__":
    main()
