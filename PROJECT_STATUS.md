# Project handoff — 9 October 2026

## All six requested tasks are complete

1. Trained five independently initialized, domain-randomized shared DQN policies for 300,000 steps each.
2. Corrected the SUMO phase classifier and selected a 15-second decision interval with a 20-second minimum green on separate validation seeds.
3. Added seven grid scenarios: balanced, rush hour, unbalanced demand, road closure, emergency surge, sensor failure, and a Lemgo-calibrated proxy.
4. Grounded the Lemgo proxy in the public Fraunhofer OWL322 demand files; extraction and mapping assumptions are recorded in `networks/lemgo/calibration.json`.
5. Added held-out paired statistics with 95% confidence intervals and publication-ready plots.
6. Added an offline hardware-in-the-loop safety gateway with protected intervals, minimum-green enforcement, invalid-command rejection, and heartbeat fallback.

The browser dashboard was updated with the final evidence and remains available privately at <https://signal-lab-traffic-control.thomsonthomasoffice.chatgpt.site>.

## Verification performed

- Unit and integration tests: **28 passed**.
- Python source compilation: passed.
- Frontend JavaScript syntax check: passed.
- Grid training: five seeds × 300,000 steps.
- Model selection: ten validation seeds (5000–5009), separate from final evaluation.
- Grid evaluation: 30 held-out seeds (10000–10029) × 1,200 seconds × seven scenarios × four controllers.
- SUMO timing selection: seeds 40–41 only.
- SUMO final evaluation: five held-out seeds (42–46) × 900 seconds × two controllers.
- Safety: zero grid conflicts and zero SUMO collisions in the final evaluations; all five offline HIL checks passed.

## Final measured results

| Evaluation | Reference | Shared DQN | Interpretation |
|---|---:|---:|---|
| Balanced grid wait vs fixed | 15.90 s | **10.99 s** | 30.9% lower, significant at 95% |
| Balanced grid wait vs heuristic | 11.99 s | **10.99 s** | 8.3% lower, significant at 95% |
| SUMO waiting load | 1,174.55 | **1,102.53** | 6.1% lower, significant at 95% |
| SUMO peak queue | 44.8 | **41.4** | 7.6% lower, significant at 95% |
| SUMO mean queue | 20.72 | **20.66** | 0.3% lower, not significant |
| SUMO mean arrivals | 230.2 | 229.0 | Difference not significant |

The DQN beat the queue heuristic on waiting time in all seven grid scenarios. Fixed timing remained better in several severe-disruption cases, so the evidence does not establish universal superiority.

## Main commands

```powershell
pytest -q
python scripts/train_multi_agent_dqn.py --timesteps 300000 --seed 0 --scenario domain_randomized
python scripts/select_multi_agent_model.py --model-glob "models/grid_shared_dqn_300k_seed_*.pt"
python scripts/evaluate_multi_agent.py --models models/grid_shared_dqn_300k_seed_1.pt --episodes 30
python scripts/run_sumo_policy.py --model models/grid_shared_dqn_300k_seed_1.pt --seeds 42 43 44 45 46 --steps 900 --decision-interval 15 --minimum-green 20
python scripts/run_hil_validation.py
python scripts/plot_multi_agent_results.py
python -m http.server 4173 --directory frontend/dist
```

## Deployment boundary

The HIL gateway is an offline safety contract only. No physical controller or public-road signal is connected, authorized, or certified. Any next deployment stage requires local detector calibration, independent traffic-engineering review, road-authority approval, vendor-specific controller integration, and controlled testing before a public-road trial.
