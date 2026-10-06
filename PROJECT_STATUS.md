# Project handoff — 7 October 2026

## Completed

- Built the original Gymnasium traffic environment, baselines, Stable-Baselines3 DQN workflow, experiment tooling, Pygame renderer, and responsive frontend.
- Added a cooperative 2×2 environment with four agents, turning traffic, bicycles, pedestrians, network transfer, centralized observations, and local observations.
- Added an enforced 30-second multimodal safety cycle. Pedestrians cross only in their protected phase; vehicles and bicycles stop during pedestrian and clearance intervals; invalid actions in protected intervals are ignored.
- Added parameter-sharing multi-agent DQN training and saved three independently initialized 30,000-step policies.
- Added matched-seed evaluation against fixed-time, queue-heuristic, and random controllers.
- Added a real central-Lemgo SUMO/TraCI scenario with 9 signal controllers, 48 controlled lanes, protected pedestrian-only routes, and a safe policy adapter that preserves native yellow/all-red transitions.
- Added native-versus-DQN SUMO evaluation across three matched seeds.
- Added measured results and honest trade-offs to the browser dashboard and README.
- Published the frontend privately at <https://signal-lab-traffic-control.thomsonthomasoffice.chatgpt.site>.

## Verification performed

- `pytest`: **23 passed**.
- Python source compilation: passed.
- Three grid policies trained for 30,000 steps each and saved under `models/`.
- Grid evaluation: 10 held-out seeds × 1,200 seconds × all baselines; output saved in `results/multi_agent_evaluation.json`.
- SUMO 1.27.1 policy evaluation: 3 seeds × 900 seconds × two controllers; output saved in `results/sumo_policy_evaluation.json`.
- Grid and SUMO evaluations both recorded **zero safety conflicts/collisions**.
- Frontend results panel shows the exact committed benchmark aggregates and clearly labels the work preliminary.

## Measured results

| Evaluation | Native / fixed | Shared DQN | Interpretation |
|---|---:|---:|---|
| Grid mean vehicle wait | 15.78 s | 9.35 s | 40.7% lower than fixed |
| Grid heuristic comparison | 12.15 s | 9.35 s | 23.0% lower than heuristic |
| SUMO waiting load | 1,191.80 | 1,089.01 | 8.6% lower |
| SUMO mean queue | 20.86 | 22.15 | 6.2% higher |
| SUMO mean arrivals | 230.00 | 232.33 | Slightly higher |

These are pipeline-validation results from short training, not a converged policy or field-deployment claim. The SUMO waiting metric is aggregate controlled-lane waiting load per simulation step.

## Main commands

```powershell
pytest -q
python scripts/train_multi_agent_dqn.py --timesteps 30000 --seed 0
python scripts/evaluate_multi_agent.py --model-glob "models/grid_shared_dqn_seed_*.pt"
python scripts/run_sumo_scenario.py --steps 900 --seed 42
python scripts/run_sumo_policy.py --seeds 42,43,44 --steps 900
python -m http.server 4173 --directory frontend/dist
```

## Recommended next research stage

Train at least 300,000–500,000 steps per seed, add more held-out seeds and demand regimes, tune only on a validation set, then test the frozen policy against detector-calibrated traffic. Before any real-world use, obtain traffic-engineering safety review and use a staged simulation/hardware-in-the-loop deployment process.
