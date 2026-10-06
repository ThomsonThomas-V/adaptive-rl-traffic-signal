"""Evaluate shared DQN and baseline controllers on identical grid demand seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

from _common import ROOT
import numpy as np
import torch

from traffic_rl.envs.network_env import GridConfig, TrafficGridEnv, cooperative_actions
from train_multi_agent_dqn import SharedQNetwork


Controller = Callable[[TrafficGridEnv, np.random.Generator], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=1_200)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "multi_agent_evaluation.json",
    )
    return parser.parse_args()


def load_policy(path: Path) -> SharedQNetwork:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    policy = SharedQNetwork(int(checkpoint.get("observation_size", 11)))
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy


def dqn_controller(policy: SharedQNetwork) -> Controller:
    def act(env: TrafficGridEnv, _: np.random.Generator) -> np.ndarray:
        observations = np.stack(tuple(env.agent_observations().values()))
        with torch.no_grad():
            return policy(torch.as_tensor(observations)).argmax(dim=1).numpy()

    return act


def heuristic(env: TrafficGridEnv, _: np.random.Generator) -> np.ndarray:
    return cooperative_actions(env.agent_observations())


def fixed(env: TrafficGridEnv, _: np.random.Generator) -> np.ndarray:
    desired = (env.timestep // 30) % 2
    return np.asarray(
        [0 if int(phase) == desired else desired + 1 for phase in env.phases],
        dtype=np.int64,
    )


def random_controller(_: TrafficGridEnv, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 3, size=4, dtype=np.int64)


def run_episode(
    controller: Controller, seed: int, config: GridConfig
) -> dict[str, float]:
    env = TrafficGridEnv(config)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 99_000)
    reward_total = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(controller(env, rng))
        reward_total += reward
        if terminated or truncated:
            break
    return {
        "reward": reward_total,
        "vehicle_average_wait": float(info["vehicle_average_wait"]),
        "pedestrian_average_wait": float(info["pedestrian_average_wait"]),
        "bicycle_average_wait": float(info["bicycle_average_wait"]),
        "throughput": float(info["throughput"]),
        "completion_ratio": float(info["completion_ratio"]),
        "pedestrian_completion_ratio": float(info["pedestrian_completion_ratio"]),
        "bicycle_completion_ratio": float(info["bicycle_completion_ratio"]),
        "conflicts": float(info["conflicts"]),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, object]:
    metrics: dict[str, dict[str, float]] = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        metrics[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return {"runs": len(rows), "metrics": metrics}


def main() -> None:
    args = parse_args()
    config = GridConfig(episode_length=args.episode_length)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    collected: dict[str, list[dict[str, float]]] = defaultdict(list)

    baselines: dict[str, Controller] = {
        "fixed_time": fixed,
        "queue_heuristic": heuristic,
        "random": random_controller,
    }
    for name, controller in baselines.items():
        for seed in seeds:
            collected[name].append(run_episode(controller, seed, config))

    for model_path in args.models:
        policy = load_policy(model_path)
        for seed in seeds:
            collected["shared_dqn"].append(
                run_episode(dqn_controller(policy), seed, config)
            )

    result = {
        "status": "measured",
        "environment": "TrafficGridEnv protected multimodal cycle",
        "episode_length": args.episode_length,
        "evaluation_seeds": seeds,
        "models": [str(path) for path in args.models],
        "controllers": {name: summarize(rows) for name, rows in collected.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
