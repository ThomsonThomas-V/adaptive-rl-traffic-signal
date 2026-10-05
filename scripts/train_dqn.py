"""Train one or more reproducible Stable-Baselines3 DQN agents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import parse_seeds
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from traffic_rl.envs import TrafficLightEnv, get_scenario


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scenario", default="medium")
    result.add_argument("--timesteps", type=int, default=500_000)
    result.add_argument("--seeds", default="0,1,2", help="Comma-separated training seeds")
    result.add_argument("--seed", type=int, help="Train one seed; overrides --seeds")
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--buffer-size", type=int, default=100_000)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--gamma", type=float, default=0.99)
    result.add_argument("--learning-starts", type=int, default=5_000)
    result.add_argument("--output-dir", type=Path, default=Path("models"))
    result.add_argument("--episode-length", type=int, default=1_200)
    return result


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    print(json.dumps({**vars(args), "output_dir": str(args.output_dir)}, indent=2))
    seeds = [args.seed] if args.seed is not None else parse_seeds(args.seeds)
    for seed in seeds:
        set_random_seed(seed)
        run_dir = args.output_dir / f"{args.scenario}_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = get_scenario(args.scenario, episode_length=args.episode_length)
        train_env = Monitor(
            TrafficLightEnv(config),
            filename=str(run_dir / "monitor.csv"),
            info_keywords=("average_waiting_time",),
        )
        eval_env = Monitor(TrafficLightEnv(config))
        model = DQN(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.20,
            exploration_final_eps=0.05,
            verbose=1,
            seed=seed,
            tensorboard_log=str(Path("logs") / args.scenario / f"seed_{seed}"),
        )
        callbacks = [
            CheckpointCallback(
                save_freq=max(args.timesteps // 10, 1),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="dqn",
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir),
                log_path=str(run_dir),
                eval_freq=max(args.timesteps // 20, 1),
                n_eval_episodes=5,
                deterministic=True,
            ),
        ]
        (run_dir / "config.json").write_text(
            json.dumps({"seed": seed, "dqn": vars(args), "environment": config.to_dict()}, default=str, indent=2),
            encoding="utf-8",
        )
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
        model.save(run_dir / "final_model")
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
