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
from traffic_rl.envs.network_scenarios import SCENARIOS, grid_scenario
from train_multi_agent_dqn import SharedQNetwork


Controller = Callable[[TrafficGridEnv, np.random.Generator], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--episode-length", type=int, default=1_200)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "multi_agent_evaluation.json",
    )
    return parser.parse_args()


class NumpyPolicy:
    """Small NumPy inference path avoids per-second PyTorch dispatch overhead."""

    def __init__(self, network: SharedQNetwork) -> None:
        linears = [layer for layer in network.layers if isinstance(layer, torch.nn.Linear)]
        self.parameters = [
            (
                layer.weight.detach().cpu().numpy().copy(),
                layer.bias.detach().cpu().numpy().copy(),
            )
            for layer in linears
        ]

    def act(self, observations: np.ndarray) -> np.ndarray:
        values = observations
        for index, (weight, bias) in enumerate(self.parameters):
            values = values @ weight.T + bias
            if index < len(self.parameters) - 1:
                values = np.maximum(values, 0)
        return values.argmax(axis=1).astype(np.int64)


def load_policy(path: Path) -> NumpyPolicy:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    policy = SharedQNetwork(int(checkpoint.get("observation_size", 11)))
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return NumpyPolicy(policy)


def dqn_controller(policy: NumpyPolicy) -> Controller:
    def act(env: TrafficGridEnv, _: np.random.Generator) -> np.ndarray:
        observations = np.stack(tuple(env.agent_observations().values()))
        return policy.act(observations)

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
        "overflows": float(info["overflows"]),
        "max_queue": float(info["max_queue"]),
        "emergency_average_wait": float(info["emergency_average_wait"]),
        "emergency_completion_ratio": float(info["emergency_completion_ratio"]),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, object]:
    metrics: dict[str, dict[str, float]] = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        metrics[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci95_low": float(values.mean() - 1.96 * values.std(ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else float(values.mean()),
            "ci95_high": float(values.mean() + 1.96 * values.std(ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else float(values.mean()),
        }
    return {"runs": len(rows), "metrics": metrics}


def paired_comparison(
    candidate: list[dict[str, float]], baseline: list[dict[str, float]], metric: str
) -> dict[str, float | bool]:
    """Return a paired normal-approximation interval and improvement percentage."""
    repeats = max(len(candidate) // len(baseline), 1)
    baseline_values = np.tile(
        np.asarray([row[metric] for row in baseline], dtype=np.float64), repeats
    )[: len(candidate)]
    candidate_values = np.asarray([row[metric] for row in candidate], dtype=np.float64)
    difference = candidate_values - baseline_values
    standard_error = difference.std(ddof=1) / np.sqrt(len(difference))
    low = float(difference.mean() - 1.96 * standard_error)
    high = float(difference.mean() + 1.96 * standard_error)
    baseline_mean = float(baseline_values.mean())
    return {
        "mean_difference": float(difference.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "statistically_significant_95": bool(high < 0 or low > 0),
        "improvement_percent": float(
            100.0 * (baseline_mean - candidate_values.mean()) / max(abs(baseline_mean), 1e-9)
        ),
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    baselines: dict[str, Controller] = {
        "fixed_time": fixed,
        "queue_heuristic": heuristic,
        "random": random_controller,
    }
    policies = [load_policy(path) for path in args.models]
    scenario_results: dict[str, object] = {}
    for scenario in args.scenarios:
        config = grid_scenario(scenario, args.episode_length)
        collected: dict[str, list[dict[str, float]]] = defaultdict(list)
        for name, controller in baselines.items():
            for seed in seeds:
                collected[name].append(run_episode(controller, seed, config))
        for policy in policies:
            for seed in seeds:
                collected["shared_dqn"].append(
                    run_episode(dqn_controller(policy), seed, config)
                )
        comparisons = {
            baseline: {
                metric: paired_comparison(
                    collected["shared_dqn"], collected[baseline], metric
                )
                for metric in ("vehicle_average_wait", "overflows", "max_queue")
            }
            for baseline in ("fixed_time", "queue_heuristic")
        }
        scenario_results[scenario] = {
            "controllers": {
                name: summarize(rows) for name, rows in collected.items()
            },
            "paired_dqn_comparisons": comparisons,
        }

    result = {
        "status": "measured",
        "environment": "TrafficGridEnv protected multimodal cycle",
        "episode_length": args.episode_length,
        "evaluation_seeds": seeds,
        "models": [str(path) for path in args.models],
        "scenarios": scenario_results,
        "statistical_note": "95% normal-approximation confidence intervals; paired comparisons match each model to identical demand seeds.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
