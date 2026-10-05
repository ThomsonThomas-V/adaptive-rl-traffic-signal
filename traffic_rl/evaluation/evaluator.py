"""Fair, seeded evaluation loops for learned and hand-written policies."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from traffic_rl.envs.traffic_config import TrafficConfig, get_scenario
from traffic_rl.envs.traffic_env import TrafficLightEnv


class Controller(Protocol):
    def predict(self, observation: NDArray[np.float32], deterministic: bool = True) -> Any: ...


def _action(controller: Controller, observation: NDArray[np.float32]) -> int:
    prediction = controller.predict(observation, deterministic=True)
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    return int(np.asarray(prediction).item())


def evaluate_controller(
    controller: Controller,
    config: TrafficConfig,
    seeds: Iterable[int],
    controller_name: str,
) -> pd.DataFrame:
    """Evaluate a controller over identical reproducible traffic seeds."""
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        if hasattr(controller, "reset"):
            controller.reset(seed=seed)
        env = TrafficLightEnv(config)
        observation, _ = env.reset(seed=seed)
        done = False
        episode_reward = 0.0
        final_info: dict[str, Any] = {}
        while not done:
            observation, reward, terminated, truncated, final_info = env.step(
                _action(controller, observation)
            )
            episode_reward += reward
            done = terminated or truncated
        rows.append(
            {
                "controller": controller_name,
                "scenario": config.scenario,
                "seed": seed,
                "average_waiting_time": final_info["average_waiting_time"],
                "max_queue_length": final_info["max_queue_length"],
                "average_queue_length": final_info["average_queue_length"],
                "throughput": final_info["throughput"],
                "completion_ratio": final_info["completion_ratio"],
                "signal_switches": final_info["number_of_signal_switches"],
                "total_waiting_time": final_info["total_waiting_time"],
                "episode_reward": episode_reward,
            }
        )
        env.close()
    return pd.DataFrame(rows)


def evaluate_suite(
    controller_factories: Mapping[str, Callable[[], Controller]],
    scenarios: Iterable[str],
    seeds: Iterable[int],
    config_overrides: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate every policy/scenario pair with the same materialized seeds."""
    seed_list = list(seeds)
    frames: list[pd.DataFrame] = []
    for scenario in scenarios:
        config = get_scenario(scenario, **(config_overrides or {}))
        for name, factory in controller_factories.items():
            frames.append(evaluate_controller(factory(), config, seed_list, name))
    return pd.concat(frames, ignore_index=True)
