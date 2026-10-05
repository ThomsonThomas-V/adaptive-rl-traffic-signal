"""Compare DQN with fixed, heuristic, and random policies on matched demand."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import ROOT
from stable_baselines3 import DQN

from traffic_rl.controllers import FixedTimeController, QueueHeuristicController, RandomController
from traffic_rl.evaluation import evaluate_suite, summarize_results
from traffic_rl.evaluation.plots import create_comparison_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--episode-length", type=int, default=1_200)
    args = parser.parse_args()
    dqn = DQN.load(args.model)
    factories = {
        "DQN": lambda: dqn,
        "Heuristic": lambda: QueueHeuristicController(),
        "Fixed": lambda: FixedTimeController(),
        "Random": lambda: RandomController(),
    }
    scenarios = ["low", "medium", "heavy", "unbalanced", "rush_hour"]
    seeds = range(args.seed_start, args.seed_start + args.episodes)
    results = evaluate_suite(
        factories, scenarios, seeds, {"episode_length": args.episode_length}
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "raw_results.csv", index=False)
    summary = summarize_results(results)
    summary.to_csv(args.output_dir / "summary_results.csv", index=False)
    create_comparison_plots(results, args.output_dir / "plots")
    columns = [
        "controller",
        "scenario",
        "average_waiting_time_mean",
        "average_waiting_time_std",
        "max_queue_length_mean",
        "throughput_mean",
        "average_queue_length_mean",
    ]
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
