# Project handoff — 5 October 2026

## Completed

- Implemented the custom `TrafficLightEnv` using the current Gymnasium API.
- Added configurable low, medium, heavy, unbalanced, and rush-hour demand profiles.
- Added isolated seeded arrival generation so controller actions cannot alter future demand randomness.
- Implemented yellow safety transitions, minimum and maximum green times, queue limits, overflow tracking, throughput, delay, starvation, and signal-switch metrics.
- Added fixed-time, queue-heuristic, and seeded random baseline controllers.
- Added matched-seed multi-scenario evaluation, CSV aggregation, and uncertainty plots.
- Added Stable-Baselines3 DQN training, checkpoints, TensorBoard output, deterministic evaluation, multi-seed CLI support, reward experiments, and hyperparameter experiments.
- Added the Pygame renderer and agent-visualization script.
- Added package metadata, requirements, MIT license, `.gitignore`, and portfolio documentation.
- Added a responsive interactive frontend under `frontend/dist`.
- Published the private frontend at <https://signal-lab-traffic-control.thomsonthomasoffice.chatgpt.site>.

## Verification performed

- `pytest`: **18 passed**.
- Stable-Baselines3 `check_env`: passed through the environment test suite.
- Python source compilation: passed.
- Pygame RGB rendering: returned a valid `720 × 720 × 3` frame.
- DQN smoke training: 1,000 steps completed, with model save and reload verified.
- Evaluation smoke run: all four controllers across all five scenarios completed.
- Frontend: desktop and 390 px mobile layouts verified with no horizontal overflow.
- Frontend WebMCP tools registered for configuring the simulation and reading live metrics.

## Important artifact distinction

`models/smoke/` and `results/smoke/` validate that the complete pipeline works. They are not converged research results and must not be used for quantitative portfolio claims. Run the documented 500,000-step multi-seed training and 30+ held-out evaluation episodes before adding improvement percentages to the README or résumé.

## Recommended continuation

1. Train medium-demand DQN models for seeds `0,1,2` (preferably `0..4`).
2. Plot training variation with `scripts/plot_training.py`.
3. Evaluate every trained policy on held-out low, medium, heavy, unbalanced, and rush-hour seeds.
4. Select the model using validation data, not the final generalization test set.
5. Run reward and hyperparameter experiments and save their configurations.
6. Replace the README's empty-results explanation with tables generated from `results/summary_results.csv`.
7. Add percentage-improvement claims only when supported by the saved episode-level data.

## Environment note

The machine's ordinary `python.exe` currently resolves to a disabled Windows Store shim. Development verification used the Codex bundled Python runtime. A normal Python 3.11+ installation or activated virtual environment is recommended for future terminal work.

The frontend was initially published from its own temporary Sites Git checkout. That checkout metadata is preserved locally at `frontend/.sites-source-git/` and ignored by the unified project repository. Rename it back to `frontend/.git/` only if a future Sites workflow specifically needs the original checkout; ordinary frontend edits remain tracked by the main project repository.

## Useful commands

```powershell
pytest
python scripts/train_dqn.py --scenario medium --timesteps 500000 --seeds 0,1,2
python scripts/compare_agents.py --model models/medium_seed_0/best_model.zip --episodes 30
python scripts/visualize_agent.py --model models/medium_seed_0/best_model.zip --scenario heavy
python -m http.server 4173 --directory frontend/dist
```
