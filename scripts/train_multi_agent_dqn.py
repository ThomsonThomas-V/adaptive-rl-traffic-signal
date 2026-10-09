"""Train a cooperative parameter-sharing DQN on the 2x2 traffic grid."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from _common import ROOT  # noqa: F401  # adds the repository root to sys.path
import numpy as np
import torch
from torch import nn

from traffic_rl.envs.network_env import TrafficGridEnv
from traffic_rl.envs.network_scenarios import SCENARIOS, grid_scenario


class SharedQNetwork(nn.Module):
    """One policy shared by all four homogeneous intersection agents."""

    def __init__(self, observation_size: int = 11) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


@dataclass(slots=True)
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("models/grid_shared_dqn.pt"))
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument(
        "--scenario",
        choices=(*SCENARIOS, "domain_randomized"),
        default="domain_randomized",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    rng = np.random.default_rng(args.seed)
    training_scenarios = tuple(SCENARIOS)
    active_scenario = (
        str(rng.choice(training_scenarios))
        if args.scenario == "domain_randomized"
        else args.scenario
    )
    env = TrafficGridEnv(grid_scenario(active_scenario))
    env.reset(seed=args.seed)
    observation_size = next(iter(env.agent_observations().values())).shape[0]
    policy, target = SharedQNetwork(observation_size), SharedQNetwork(observation_size)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    replay: deque[Transition] = deque(maxlen=args.buffer_size)
    episode_return = 0.0
    episode_returns: list[float] = []
    losses: list[float] = []

    for step in range(1, args.timesteps + 1):
        local = env.agent_observations()
        observations = np.stack(tuple(local.values()))
        epsilon = max(0.05, 1.0 - step / max(args.timesteps * 0.30, 1))
        with torch.no_grad():
            greedy = policy(torch.as_tensor(observations)).argmax(dim=1).numpy()
        actions = np.asarray(
            [rng.integers(3) if rng.random() < epsilon else greedy[index] for index in range(4)],
            dtype=np.int64,
        )
        _, reward, terminated, truncated, _ = env.step(actions)
        next_observations = np.stack(tuple(env.agent_observations().values()))
        done = terminated or truncated
        # A shared team reward trains every local decision toward network flow.
        for index in range(4):
            replay.append(
                Transition(observations[index], int(actions[index]), reward, next_observations[index], done)
            )
        episode_return += reward

        if len(replay) >= args.batch_size:
            batch = random.sample(replay, args.batch_size)
            obs = torch.as_tensor(np.stack([item.observation for item in batch]))
            action = torch.as_tensor([item.action for item in batch], dtype=torch.int64)
            rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32)
            next_obs = torch.as_tensor(np.stack([item.next_observation for item in batch]))
            dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32)
            selected = policy(obs).gather(1, action[:, None]).squeeze(1)
            with torch.no_grad():
                bootstrap = target(next_obs).max(dim=1).values
                expected = rewards + args.gamma * (1.0 - dones) * bootstrap
            loss = nn.functional.smooth_l1_loss(selected, expected)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.item()))

        if step % 1_000 == 0:
            target.load_state_dict(policy.state_dict())
        if done:
            episode_returns.append(episode_return)
            if step % 10_000 < env.config.episode_length:
                print(f"step={step} return={episode_return:.1f} epsilon={epsilon:.3f}")
            if args.scenario == "domain_randomized":
                active_scenario = str(rng.choice(training_scenarios))
                env.close()
                env = TrafficGridEnv(grid_scenario(active_scenario))
            env.reset(seed=args.seed + step)
            episode_return = 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "seed": args.seed,
            "timesteps": args.timesteps,
            "observation_size": observation_size,
            "actions": 3,
            "agents": list(env.NODE_NAMES),
            "training_scenario": args.scenario,
        },
        args.output,
    )
    metrics_output = args.metrics_output or args.output.with_suffix(".training.json")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "timesteps": args.timesteps,
                "episodes": len(episode_returns),
                "mean_episode_return": float(np.mean(episode_returns))
                if episode_returns
                else episode_return,
                "final_episode_return": episode_returns[-1]
                if episode_returns
                else episode_return,
                "mean_training_loss": float(np.mean(losses[-1000:])) if losses else None,
                "safety_conflicts": env.conflicts,
                "training_scenario": args.scenario,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    env.close()
    print(f"Saved shared multi-agent DQN to {args.output}")


if __name__ == "__main__":
    main()
