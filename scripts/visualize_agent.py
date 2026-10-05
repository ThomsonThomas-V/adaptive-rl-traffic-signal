"""Run a trained or baseline controller in the Pygame animation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _common import ROOT  # noqa: F401  # adds the repository root to sys.path
from stable_baselines3 import DQN

from traffic_rl.controllers import FixedTimeController, QueueHeuristicController, RandomController
from traffic_rl.envs import TrafficLightEnv, get_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--controller", choices=("dqn", "fixed", "heuristic", "random"), default="dqn")
    parser.add_argument("--scenario", default="heavy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.controller == "dqn":
        if args.model is None:
            parser.error("--model is required for --controller dqn")
        controller = DQN.load(args.model)
    else:
        controller = {
            "fixed": FixedTimeController,
            "heuristic": QueueHeuristicController,
            "random": RandomController,
        }[args.controller]()
    if hasattr(controller, "reset"):
        controller.reset(seed=args.seed)
    env = TrafficLightEnv(get_scenario(args.scenario), render_mode="human")
    observation, _ = env.reset(seed=args.seed)
    done = False
    while not done:
        prediction = controller.predict(observation, deterministic=True)
        action = prediction[0] if isinstance(prediction, tuple) else prediction
        observation, _, terminated, truncated, _ = env.step(int(np.asarray(action).item()))
        done = terminated or truncated
    env.close()


if __name__ == "__main__":
    main()
