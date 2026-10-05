"""Evaluate one deterministic DQN policy and save episode-level metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import ROOT  # noqa: F401  # adds the repository root to sys.path
from stable_baselines3 import DQN

from traffic_rl.envs import get_scenario
from traffic_rl.evaluation import evaluate_controller, summarize_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scenario", default="medium")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("results/dqn_evaluation.csv"))
    args = parser.parse_args()
    model = DQN.load(args.model)
    seeds = range(args.seed_start, args.seed_start + args.episodes)
    results = evaluate_controller(model, get_scenario(args.scenario), seeds, "DQN")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(summarize_results(results).to_string(index=False))


if __name__ == "__main__":
    main()
