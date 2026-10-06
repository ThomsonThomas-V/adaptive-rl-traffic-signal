"""Compare native and shared-DQN control on the Lemgo SUMO scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT
import numpy as np
import torch

from traffic_rl.envs.sumo_adapter import SumoAdapter, SumoConfig
from train_multi_agent_dqn import SharedQNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "networks" / "lemgo" / "lemgo.sumocfg"
    )
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "sumo_policy_evaluation.json"
    )
    return parser.parse_args()


def load_policy(path: Path) -> SharedQNetwork:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    policy = SharedQNetwork(int(checkpoint.get("observation_size", 11)))
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy


def lane_orientation(traci: object, lane_id: str) -> str:
    shape = traci.lane.getShape(lane_id)
    if len(shape) < 2:
        return "horizontal"
    (x0, y0), (x1, y1) = shape[-2], shape[-1]
    return "vertical" if abs(y1 - y0) >= abs(x1 - x0) else "horizontal"


def signal_observation(adapter: SumoAdapter, signal_id: str, phase_age: int) -> np.ndarray:
    traci = adapter._require_started()
    lanes = adapter.controlled_lanes(signal_id)
    vertical = [lane for lane in lanes if lane_orientation(traci, lane) == "vertical"]
    horizontal = [lane for lane in lanes if lane not in vertical]

    def halted(group: list[str]) -> float:
        return sum(float(traci.lane.getLastStepHaltingNumber(lane)) for lane in group)

    ns, ew = halted(vertical), halted(horizontal)
    state = str(traci.trafficlight.getRedYellowGreenState(signal_id))
    phase = float(traci.trafficlight.getPhase(signal_id) % 2)
    mode = 0.5 if "y" in state.lower() or not any(v in state for v in "Gg") else 0.0
    return np.asarray(
        [
            min(ns / 160, 1),
            min(ns / 160, 1),
            min(ew / 160, 1),
            min(ew / 160, 1),
            0.0,
            0.0,
            phase,
            min(phase_age / 90, 1),
            min(ew / 320, 1),
            min(ns / 320, 1),
            mode,
        ],
        dtype=np.float32,
    )


def run_once(
    args: argparse.Namespace, controller: str, policy: SharedQNetwork, seed: int
) -> dict[str, object]:
    adapter = SumoAdapter(SumoConfig(args.config.resolve(), seed=seed))
    adapter.start()
    try:
        traci = adapter._require_started()
        signal_ids = adapter.signal_ids()
        lanes = sorted({lane for signal in signal_ids for lane in adapter.controlled_lanes(signal)})
        ages = {signal: 0 for signal in signal_ids}
        previous_phases = {signal: int(traci.trafficlight.getPhase(signal)) for signal in signal_ids}
        requested_transitions = 0
        cumulative_queue = 0.0
        cumulative_wait = 0.0
        peak_queue = 0.0
        collisions = 0
        collision_details: list[str] = []
        departed = arrived = 0

        for step in range(args.steps):
            if controller == "shared_dqn" and step % args.decision_interval == 0:
                observations = np.stack(
                    [signal_observation(adapter, signal, ages[signal]) for signal in signal_ids]
                )
                with torch.no_grad():
                    actions = policy(torch.as_tensor(observations)).argmax(dim=1).numpy()
                for signal, action in zip(signal_ids, actions, strict=True):
                    current_class = int(traci.trafficlight.getPhase(signal) % 2)
                    desired = int(action) - 1 if int(action) else current_class
                    if (
                        ages[signal] >= 20
                        and desired != current_class
                        and adapter.request_next_safe_phase(signal)
                    ):
                        requested_transitions += 1

            adapter.step()
            metrics = adapter.lane_metrics(lanes)
            queue = sum(item["halting"] for item in metrics.values())
            cumulative_queue += queue
            cumulative_wait += sum(item["waiting_time"] for item in metrics.values())
            peak_queue = max(peak_queue, queue)
            step_collisions = adapter.collision_count()
            collisions += step_collisions
            if step_collisions:
                getter = getattr(traci.simulation, "getCollisions", None)
                if getter is not None:
                    collision_details.extend(str(item) for item in getter())
            departed += int(traci.simulation.getDepartedNumber())
            arrived += int(traci.simulation.getArrivedNumber())
            for signal in signal_ids:
                phase = int(traci.trafficlight.getPhase(signal))
                ages[signal] = 0 if phase != previous_phases[signal] else ages[signal] + 1
                previous_phases[signal] = phase

        return {
            "controller": controller,
            "steps": args.steps,
            "seed": seed,
            "signals": len(signal_ids),
            "controlled_lanes": len(lanes),
            "departed": departed,
            "arrived": arrived,
            "vehicles_remaining": int(traci.simulation.getMinExpectedNumber()),
            "mean_queue": cumulative_queue / args.steps,
            "peak_queue": peak_queue,
            "mean_waiting_seconds": cumulative_wait / args.steps,
            "requested_safe_transitions": requested_transitions,
            "collisions": collisions,
            "collision_details": collision_details,
        }
    finally:
        adapter.close()


def main() -> None:
    args = parse_args()
    policy = load_policy(args.model)
    runs = [
        run_once(args, controller, policy, seed)
        for seed in args.seeds
        for controller in ("native", "shared_dqn")
    ]
    summary: dict[str, dict[str, float]] = {}
    for controller in ("native", "shared_dqn"):
        group = [run for run in runs if run["controller"] == controller]
        summary[controller] = {
            key: float(np.mean([float(run[key]) for run in group]))
            for key in (
                "arrived",
                "vehicles_remaining",
                "mean_queue",
                "peak_queue",
                "mean_waiting_seconds",
                "collisions",
            )
        }
    result = {
        "status": "measured",
        "scenario": "Central Lemgo, Germany (OpenStreetMap)",
        "model": str(args.model),
        "runs": runs,
        "summary_mean": summary,
        "safety_note": "TraCI requests shorten only active green phases; native yellow/all-red transitions remain in control.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
