"""Run compact reward-shaping or DQN hyperparameter experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from _common import ROOT  # noqa: F401  # adds the repository root to sys.path
from stable_baselines3 import DQN

from traffic_rl.envs import TrafficLightEnv, get_scenario


REWARDS = {
    "queue_only": {"waiting_penalty": 0.0, "throughput_reward": 0.0, "switching_penalty": 0.0, "starvation_penalty": 0.0},
    "queue_throughput": {"waiting_penalty": 0.0, "throughput_reward": 1.0, "switching_penalty": 0.0, "starvation_penalty": 0.0},
    "full": {},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("reward", "hyperparameter"), required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("models/experiments"))
    args = parser.parse_args()
    base = get_scenario("medium")
    if args.kind == "reward":
        experiments = [(name, replace(base, **settings), 1e-4, 0.99) for name, settings in REWARDS.items()]
    else:
        experiments = [
            (f"lr_{lr:g}_gamma_{gamma:g}", base, lr, gamma)
            for lr in (1e-3, 3e-4, 1e-4)
            for gamma in (0.95, 0.99, 0.995)
        ]
    for name, config, learning_rate, gamma in experiments:
        destination = args.output_dir / args.kind / name
        destination.mkdir(parents=True, exist_ok=True)
        model = DQN(
            "MlpPolicy",
            TrafficLightEnv(config),
            learning_rate=learning_rate,
            gamma=gamma,
            buffer_size=100_000,
            learning_starts=min(5_000, args.timesteps // 10),
            batch_size=64,
            seed=args.seed,
            verbose=0,
        )
        model.learn(args.timesteps)
        model.save(destination / "model")
        (destination / "experiment.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "kind": args.kind,
                    "seed": args.seed,
                    "timesteps": args.timesteps,
                    "learning_rate": learning_rate,
                    "gamma": gamma,
                    "environment": config.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        model.get_env().close()


if __name__ == "__main__":
    main()
