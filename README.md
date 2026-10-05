# Adaptive Traffic Signal Control with Deep Reinforcement Learning

A reproducible reinforcement-learning project that models one four-way intersection as a Gymnasium environment and trains a Stable-Baselines3 DQN signal controller. The policy is compared with fixed-time, queue-responsive, and random control using operational traffic metrics—not reward alone.

**Interactive demo:** [Signal Lab — Adaptive Traffic Control](https://signal-lab-traffic-control.thomsonthomasoffice.chatgpt.site) (private ChatGPT-authenticated Site)

## Overview

Traditional fixed-time plans cannot respond to stochastic or directionally changing demand. This project treats signal selection as a sequential decision problem. The agent sees queues, accumulated delay, signal state, phase age, and recent observed arrivals; it requests North–South or East–West green while the simulator enforces yellow and minimum-green safety rules.

The simulator is intentionally lightweight: one step is one second, each approach is one straight-through lane, arrivals are Poisson, and opposing approaches discharge simultaneously. This makes experiments fast enough for multi-seed analysis while keeping the code open to extensions.

## RL formulation

- **Agent:** the traffic-light controller.
- **Environment:** an isolated four-way intersection.
- **State:** four normalized queues, four normalized directional delay totals, active phase, phase age, yellow status, and four causal recent arrival-rate estimates (15 floats).
- **Action:** `0` keep, `1` request NS green, `2` request EW green.
- **Reward:** configurable queue and oldest-wait penalties, throughput bonus, switching cost, illegal-switch cost, and starvation penalty.
- **Episode:** 1,200 simulated seconds by default; reaching the horizon is a Gymnasium truncation.

The agent never controls yellow directly. A legal phase change creates an all-stop yellow interval. A request before `minimum_green_time` is ignored and receives a small penalty. `maximum_green_time` can force the opposing phase to prevent permanent starvation.

## Traffic simulation

Every direction receives an independent Poisson arrival count. Defaults are calibrated against a simplified saturation service rate of 0.80 vehicle/second per active approach. Even the balanced heavy case (1.36 arrivals/second total) remains below the 1.60 vehicles/second ideal intersection capacity, while yellow lost time keeps it challenging rather than mathematically impossible:

| Scenario | North/South | East/West | Behavior |
|---|---:|---:|---|
| Low | 0.08 | 0.08 | Spare capacity |
| Medium | 0.20 | 0.20 | Typical balanced demand |
| Heavy | 0.34 | 0.34 | Congested but controllable |
| Unbalanced | 0.36 | 0.09 | Strong NS bias |
| Rush hour | Varies | Varies | Low, NS peak, EW peak, medium |

Traffic generation has its own seeded NumPy generator. Therefore two controllers reset with the same seed receive identical arrival sequences even if their actions differ. This is essential for fair paired comparisons.

Waiting time is a queue integral: every queued vehicle adds one vehicle-second per step. No per-vehicle objects are required. The reported average is total vehicle-seconds divided by total arrivals, so delay accumulated by vehicles still queued at the horizon is retained. Overflow at the configurable 100-vehicle approach limit is tracked separately.

## Reward function

The default reward is:

```text
- 0.06 × total queue
- 0.02 × sum of oldest approach waits
+ 1.00 × vehicles discharged
- 0.25 × legal phase changes
- 0.05 × premature switch attempts
- 0.50 × starved approaches
```

All coefficients live in `TrafficConfig`. `scripts/run_experiments.py --kind reward` trains queue-only, queue-plus-throughput, and full-reward variants. A throughput-heavy objective can ignore lingering vehicles, an excessive switching cost can make control sluggish, and raw queue-only control often works surprisingly well but offers no direct incentive to complete trips. Reward scale affects DQN gradients and should be interpreted alongside real metrics.

## DQN and baselines

The DQN uses an MLP policy, replay buffer, target network, epsilon-greedy exploration, and deterministic predictions for evaluation. Key hyperparameters are CLI options; the default starting point uses a `1e-4` learning rate, 100,000-transition replay buffer, batch size 64, and discount 0.99.

- **Fixed:** alternates after 30 green seconds; the environment supplies yellow.
- **Heuristic:** switches when the opposing aggregate queue exceeds the current queue by 20%, with hysteresis and minimum-green timing.
- **Random:** seeded uniform action sampling, used as a lower bound.

## Evaluation and results

`compare_agents.py` uses the same held-out seed set for every controller and every named scenario, writes every episode to `results/raw_results.csv`, writes mean and sample standard deviation to `results/summary_results.csv`, and creates grouped error-bar plots. Metrics are average waiting time, maximum and average queue, throughput, completion ratio, signal switches, total delay, and episode reward.

No benchmark numbers are committed in this initial source tree because long DQN training has not been run as part of repository creation. This is deliberate: fabricated results are worse than an empty table. After training, generate the actual tables and figures with:

```powershell
python scripts/compare_agents.py --model models/medium_seed_0/best_model.zip --episodes 30
```

Use `scripts/plot_training.py` to plot rolling training reward, rolling waiting time, and deterministic evaluation reward. For a defensible portfolio claim, train at least three seeds, report variation between them, and distinguish the medium training distribution from low, heavy, unbalanced, and rush-hour generalization tests. Neural initialization, exploration, minibatch sampling, hardware, and PyTorch kernels all contribute to seed variation.

## Installation

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
pytest
```

The SB3 checker can also be run directly:

```python
from stable_baselines3.common.env_checker import check_env
from traffic_rl import TrafficLightEnv

check_env(TrafficLightEnv())
```

## Usage

Train three seeds and inspect TensorBoard:

```powershell
python scripts/train_dqn.py --scenario medium --timesteps 500000 --seeds 0,1,2
tensorboard --logdir logs/
```

Evaluate one model or perform the complete matched comparison:

```powershell
python scripts/evaluate.py --model models/medium_seed_0/best_model.zip --scenario medium
python scripts/compare_agents.py --model models/medium_seed_0/best_model.zip --episodes 30
```

Visualize learned or baseline behavior:

```powershell
python scripts/visualize_agent.py --model models/medium_seed_0/best_model.zip --scenario heavy
python scripts/visualize_agent.py --controller heuristic --scenario rush_hour
```

Run controlled experiments and plot learning curves:

```powershell
python scripts/run_experiments.py --kind reward --timesteps 100000
python scripts/run_experiments.py --kind hyperparameter --timesteps 100000
python scripts/plot_training.py models/medium_seed_0 models/medium_seed_1 models/medium_seed_2
```

## Interactive frontend

The `frontend/` directory contains a responsive static control-room interface for exploring the simulator. It provides animated queues and signal phases, traffic-profile selection, DQN-demo and baseline controller modes, simulation speed controls, operational metrics, queue-pressure charts, and an event stream.

The browser simulation mirrors the Python environment's queue, signal-safety, and demand concepts for demonstration. It is not a replacement for Stable-Baselines3 evaluation: benchmark results and trained-policy claims must continue to come from the Python pipeline and saved CSV outputs.

Run the frontend locally from `frontend/dist` with any static file server, for example:

```powershell
python -m http.server 4173 --directory frontend/dist
```

## Project structure

```text
traffic_rl/
  envs/             configuration, scenarios, Gymnasium environment
  controllers/      fixed, heuristic, and random policies
  evaluation/       seeded evaluation, aggregation, and plots
  visualization/    lazy-loaded Pygame renderer
scripts/            train, evaluate, compare, experiment, plot, visualize
tests/              API, dynamics, reproducibility, controller, metric tests
models/             saved models and checkpoints
results/            episode CSVs and publication-quality plots
logs/               TensorBoard event files
frontend/dist/      interactive browser dashboard
```

## Experiment protocol

1. Validate the simulator with `pytest` and `check_env`.
2. Compare baselines before training.
3. Train DQN under medium demand with seeds `0,1,2` (preferably `0..4`).
4. Select hyperparameters only from training/validation results.
5. Freeze policies and evaluate on held-out seeds in all five scenarios.
6. Report mean ± standard deviation and operational metrics.
7. Calculate percentage improvement only from saved CSVs, for example `(fixed_wait - dqn_wait) / fixed_wait × 100`.

Sanity checks represented in the tests include zero demand, single-axis arrivals, queue non-negativity, green-only discharge, deterministic demand, signal safety, and exact episode truncation. Additional experiment-level checks should confirm balanced traffic is not permanently starved and adaptive policies allocate more green to the busy axis.

## Limitations

This is one isolated intersection with straight-through vehicles, one lane per approach, aggregate queues, simplified saturation flow, no turn conflicts, no pedestrians, no emergency priority, no spillback network, and no neighboring-signal coordination. Poisson arrivals and queue-count waiting time are useful abstractions, not a calibrated microscopic traffic model. Results do not transfer directly to a real junction.

## Future work

Promising extensions include PPO and Double DQN, prioritized replay, domain randomization, turning movements and pedestrian phases, emergency priority, traffic forecasting, real detector data, coordinated intersections, multi-agent RL, graph neural networks, and SUMO/TraCI validation. SUMO is intentionally not required for version 1.

## What I learned

The central engineering lesson is that environment validity and experimental fairness matter more than an impressive reward curve. Separating exogenous randomness, enforcing safety inside the environment, retaining unfinished-vehicle delay, evaluating deterministic policies on shared seeds, and publishing episode-level data make the conclusions inspectable. RL may outperform a fixed plan under changing demand, but a strong queue heuristic is the comparison that reveals whether the added complexity is worthwhile.
