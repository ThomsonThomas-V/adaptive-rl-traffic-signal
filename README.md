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

### Cooperative 2×2 network

`TrafficGridEnv` adds four connected intersections (agents A–D), straight-through vehicle transfer between neighbors, configurable left/right turn ratios, pedestrian demand, bicycle demand, and independent minimum-green state. Its 44-value centralized observation and four local 11-value observations include the protected-mode state. The action space is `MultiDiscrete([3, 3, 3, 3])`.

```python
from traffic_rl.envs import TrafficGridEnv, cooperative_actions

env = TrafficGridEnv()
observation, info = env.reset(seed=42)
action = cooperative_actions(env.agent_observations())
observation, reward, terminated, truncated, info = env.step(action)
```

The browser dashboard can switch between the detailed single junction, the cooperative 2×2 grid, and a verified SUMO profile for central Lemgo, Germany.

`scripts/train_multi_agent_dqn.py` implements cooperative parameter-sharing DQN: agents A–D use the same neural policy, contribute local transitions to one replay buffer, and learn from a shared network reward. This exploits the intersections' common observation/action structure while still producing four simultaneous decisions.

```powershell
python scripts/train_multi_agent_dqn.py --timesteps 30000 --seed 0
python scripts/evaluate_multi_agent.py --model-glob "models/grid_shared_dqn_seed_*.pt"
```

### Verified SUMO integration

`traffic_rl.envs.sumo_adapter.SumoAdapter` connects policies to a locally installed SUMO/TraCI runtime. The committed `networks/lemgo/lemgo.sumocfg` uses a real central-Lemgo OpenStreetMap extract with passenger cars, bicycles, protected pedestrian routes, turning routes, 9 signal controllers, and 48 controlled lanes. Pedestrian routes are limited to explicit pedestrian lanes in `pedestrians_safe.rou.xml`. The adapter resolves both normal SUMO installations and the official `eclipse-sumo` Python wheel.

```python
from pathlib import Path
from traffic_rl.envs.sumo_adapter import SumoAdapter, SumoConfig

sumo = SumoAdapter(SumoConfig(Path("networks/lemgo/lemgo.sumocfg")))
sumo.start()
sumo.step()
metrics = sumo.lane_metrics(["north_0", "south_0"])
sumo.close()
```

Run the reproducible native verification:

```powershell
python -m pip install eclipse-sumo
python scripts/run_sumo_scenario.py --steps 900
```

The checked run on SUMO 1.27.1 completed 900 TraCI steps with 426 departures, 232 arrivals, and zero collisions. Its machine-readable output is saved in `results/sumo_lemgo_verification.json`.

Run the shared policy against SUMO's native program on matched seeds:

```powershell
python scripts/run_sumo_policy.py --seeds 42,43,44 --steps 900
```

The policy can request an earlier transition only after a 20-second minimum green. SUMO still performs the yellow/all-red stages encoded in the native program, so the adapter never jumps directly from one conflicting green to another.

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

`compare_agents.py` evaluates the original single-junction policies. `evaluate_multi_agent.py` performs matched-seed evaluation of the cooperative grid. Three independently initialized shared DQN models were trained for 30,000 steps and evaluated on ten held-out seeds (10000–10009), 1,200 seconds each:

| Grid controller | Mean vehicle wait | Mean throughput | Completion ratio | Conflicts |
|---|---:|---:|---:|---:|
| Shared DQN | **9.35 s** | 1,517.47 | 98.73% | 0 |
| Queue heuristic | 12.15 s | 1,512.70 | 98.84% | 0 |
| Fixed time | 15.78 s | 1,517.30 | 98.08% | 0 |
| Random | 13.02 s | 1,513.40 | 98.62% | 0 |

On this preliminary benchmark, DQN reduced mean vehicle wait by 40.7% versus fixed time and 23.0% versus the queue heuristic. The complete per-policy aggregates are committed in `results/multi_agent_evaluation.json`.

The Lemgo SUMO comparison covers three seeds and 900 seconds per seed:

| SUMO controller | Mean controlled-lane waiting load | Mean queue | Mean arrivals | Collisions |
|---|---:|---:|---:|---:|
| Native signal program | 1,191.80 | **20.86** | 230.00 | 0 |
| Shared DQN adapter | **1,089.01** | 22.15 | **232.33** | 0 |

The DQN adapter lowered aggregate waiting load by 8.6% and slightly increased arrivals, but mean queue rose by 6.2%. This mixed result is reported deliberately: the 30,000-step models validate the complete pipeline, but they are not a claim of convergence or field readiness. Full run data is in `results/sumo_policy_evaluation.json`.

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

### Multimodal safety cycle

The browser visualization uses a deterministic 30-second conflict-free display cycle around the controller output:

| Cycle interval | Permitted movement |
|---|---|
| Seconds 0–10 | East–west vehicles and bicycles |
| Second 11 | All-red clearance |
| Seconds 12–22 | North–south vehicles and bicycles |
| Second 23 | All-red clearance |
| Seconds 24–27 | Protected pedestrian crossing only |
| Seconds 28–29 | All-red clearance before vehicles restart |

Waiting pedestrians are placed on the curb and waiting bicycles and vehicles are placed at stop lines; an inactive animation is reset to that safe waiting position rather than frozen inside the conflict zone. Pedestrians move only during the protected crossing phase, and all motorized and bicycle streams remain stopped until the final clearance interval has completed. The crossing animation scales with the selected simulation speed.

The Python grid environment uses the same 30-second protection principle while allowing each agent to choose its vehicle phase during seconds 0–22: second 23 is pre-pedestrian clearance, seconds 24–27 serve pedestrians only, and seconds 28–29 are post-pedestrian clearance. Invalid actions during protected intervals are ignored by the environment.

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

The original `TrafficLightEnv` remains one isolated intersection with straight-through vehicles, one lane per approach, aggregate queues, simplified saturation flow, no turn conflicts, no pedestrians, no emergency priority, and no spillback. `TrafficGridEnv` adds four neighboring intersections, turning proportions, bicycles, protected pedestrian service, and clearance intervals, but it is still a queue-level abstraction rather than a calibrated microscopic model. The Lemgo SUMO scenario improves geometric and road-user realism; three short evaluation seeds are still not field calibration. The SUMO metric named waiting load is a per-step aggregate over controlled lanes, not a per-person travel-time average. Results do not transfer directly to a real junction without detector data, calibration, safety review, and controlled deployment testing.

## Future work

Promising extensions include longer training, PPO and Double DQN, prioritized replay, domain randomization, emergency priority, traffic forecasting, real detector calibration, and graph neural networks. SUMO remains optional for the core unit-test suite.

## What I learned

The central engineering lesson is that environment validity and experimental fairness matter more than an impressive reward curve. Separating exogenous randomness, enforcing safety inside the environment, retaining unfinished-vehicle delay, evaluating deterministic policies on shared seeds, and publishing episode-level data make the conclusions inspectable. RL may outperform a fixed plan under changing demand, but a strong queue heuristic is the comparison that reveals whether the added complexity is worthwhile.
