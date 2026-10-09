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
python scripts/train_multi_agent_dqn.py --timesteps 300000 --seed 0 --scenario domain_randomized
python scripts/evaluate_multi_agent.py --models models/grid_shared_dqn_300k_seed_1.pt --episodes 30
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

The checked native run completed 900 TraCI steps with 426 departures, 232 arrivals, and zero collisions. Its machine-readable output is saved in `results/sumo_lemgo_verification.json`.

The `calibrated_lemgo` grid profile is grounded in Fraunhofer IOSB-INA's public [LemgoRL OWL322 model](https://github.com/RL-INA/LemgoRL) and the associated [German government data-catalog record](https://data.gov.de/suche/daten/simulationsmodell-ki-schnittstelle-und-verkehrszahldaten-einer-kreuzung-in-der-alten-hansestadt). It uses the published 15:00–15:20 five-minute motorized and pedestrian rates. `networks/lemgo/calibration.json` records the extracted values and scaling assumptions. This is explicitly a calibrated proxy: OWL322 is one intersection, while this project's four-node grid and central-Lemgo OSM network have different geometry.

Run the shared policy against SUMO's native program on matched seeds:

```powershell
python scripts/run_sumo_policy.py --model models/grid_shared_dqn_300k_seed_1.pt --seeds 42 43 44 45 46 --steps 900 --decision-interval 15 --minimum-green 20
```

The policy can request an earlier transition only after a 20-second minimum green. SUMO still performs the yellow/all-red stages encoded in the native program, so the adapter never jumps directly from one conflicting green to another.

### Offline hardware-in-the-loop safety contract

`traffic_rl.safety.SafetyGateway` is a dry-run integration boundary for future controller hardware. The RL policy can request only a high-level NS/EW transition; it cannot command individual lamps. The gateway rejects invalid phases, minimum-green violations, and requests during protected pedestrian intervals. A stale heartbeat immediately restores native-controller authority. Run `python scripts/run_hil_validation.py`; the committed result in `results/hil_safety_validation.json` passes all five checks. No live road hardware is connected or authorized by this project.

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

`compare_agents.py` evaluates the original single-junction policies. `evaluate_multi_agent.py` performs matched-seed evaluation of the cooperative grid with 95% confidence intervals. Five independently initialized policies were trained for 300,000 steps each using domain randomization. Model selection used validation seeds 5000–5009 and an equal-weight score of waiting time, overflow, and maximum queue versus the heuristic. Seed 1 was frozen before final evaluation on 30 unseen seeds (10000–10029), 1,200 seconds each.

| Grid controller | Mean vehicle wait | Mean throughput | Completion ratio | Conflicts |
|---|---:|---:|---:|---:|
| Shared DQN | **10.99 s** | 1,520.73 | 98.73% | 0 |
| Queue heuristic | 11.99 s | 1,519.23 | 98.81% | 0 |
| Fixed time | 15.90 s | 1,512.93 | 98.08% | 0 |
| Random | 13.08 s | 1,523.80 | 98.65% | 0 |

On balanced traffic, DQN reduced mean vehicle wait by 30.9% versus fixed time and 8.3% versus the queue heuristic; both paired 95% intervals exclude zero. Across all seven scenarios, it beat the heuristic on waiting time, including rush hour, imbalance, road closure, emergency surge, sensor failure, and the Lemgo-calibrated proxy. Fixed timing remained better in several heavy-disruption cases, so the result does not support a universal superiority claim. Full results are in `results/multi_agent_evaluation_300k.json`; selection evidence is in `results/model_selection_300k.json`.

The final Lemgo SUMO comparison covers five untouched seeds and 900 seconds per seed. Safe timing was selected separately on seeds 40–41:

| SUMO controller | Mean controlled-lane waiting load | Mean queue | Mean arrivals | Collisions |
|---|---:|---:|---:|---:|
| Native signal program | 1,174.55 | 20.72 | 230.20 | 0 |
| Shared DQN adapter | **1,102.53** | **20.66** | 229.00 | 0 |

The corrected adapter infers NS/EW movement from each signal's actual green controlled links instead of assuming a phase-index mapping. It lowered aggregate waiting load by 6.1%, peak queue by 7.6%, and mean queue by 0.3%. The waiting-load and peak-queue reductions are statistically significant at 95%; the tiny mean-queue and throughput differences are not. Full paired statistics are in `results/sumo_policy_evaluation_300k.json`.

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
  safety/           fail-safe hardware-in-the-loop gateway
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
3. Train five parameter-sharing policies for 300,000 steps with domain-randomized demand and incidents.
4. Select the final model only on validation seeds 5000–5009.
5. Freeze the policy and evaluate it on 30 held-out grid seeds in all seven scenarios.
6. Select safe SUMO timing separately, then evaluate on five untouched SUMO seeds.
7. Report 95% paired confidence intervals and operational metrics, not reward alone.
8. Run the offline HIL safety contract before considering any controller connection.

Generate the final figures with `python scripts/plot_multi_agent_results.py`. The committed charts in `results/plots_300k/` cover waiting time, overflow, maximum queue, and the SUMO comparison.

Sanity checks represented in the tests include zero demand, single-axis arrivals, queue non-negativity, green-only discharge, deterministic demand, signal safety, and exact episode truncation. Additional experiment-level checks should confirm balanced traffic is not permanently starved and adaptive policies allocate more green to the busy axis.

## Limitations

The original `TrafficLightEnv` remains one isolated intersection with straight-through vehicles, one lane per approach, aggregate queues, simplified saturation flow, no turn conflicts, no pedestrians, no emergency priority, and no spillback. `TrafficGridEnv` adds four neighboring intersections, turning proportions, bicycles, protected pedestrian service, spillback penalties, emergency demand, road closures, sensor failures, and clearance intervals, but it is still a queue-level abstraction rather than a calibrated microscopic model. The Lemgo profile uses published OWL322 demand as a proxy; it is not detector calibration of this project's different four-node grid or OSM network. Five 900-second SUMO seeds are research evidence, not a field trial. The SUMO waiting-load metric is a per-step aggregate over controlled lanes, not a per-person travel-time average. Results do not transfer directly to a real junction without local detector data, calibration, independent traffic-engineering safety review, authority approval, controller-vendor integration, and staged closed-course testing.

## Future work

Promising extensions include PPO and Double DQN, prioritized replay, traffic forecasting, real detector calibration, graph neural networks, and closed-loop tests with an approved traffic-controller emulator. SUMO remains optional for the core unit-test suite. The offline HIL gateway is intentionally fail-safe scaffolding and must not be treated as authorization or certification for a live road installation.

## What I learned

The central engineering lesson is that environment validity and experimental fairness matter more than an impressive reward curve. Separating exogenous randomness, enforcing safety inside the environment, retaining unfinished-vehicle delay, evaluating deterministic policies on shared seeds, and publishing episode-level data make the conclusions inspectable. RL may outperform a fixed plan under changing demand, but a strong queue heuristic is the comparison that reveals whether the added complexity is worthwhile.
