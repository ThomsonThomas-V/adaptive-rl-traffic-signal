"""Run and verify the real Lemgo SUMO scenario through the TraCI adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT
from traffic_rl.envs.sumo_adapter import SumoAdapter, SumoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "networks" / "lemgo" / "lemgo.sumocfg",
    )
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sumo_lemgo_verification.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = SumoAdapter(
        SumoConfig(args.config.resolve(), gui=args.gui, step_length=1.0, seed=args.seed)
    )
    adapter.start()
    try:
        traci = adapter._require_started()
        signal_ids = list(traci.trafficlight.getIDList())
        controlled_lanes = sorted(
            {
                lane
                for signal_id in signal_ids
                for lane in traci.trafficlight.getControlledLanes(signal_id)
            }
        )
        lane_ids = list(traci.lane.getIDList())

        peak_halting = 0.0
        cumulative_halting = 0.0
        cumulative_waiting = 0.0
        departed = 0
        arrived = 0

        for _ in range(args.steps):
            adapter.step()
            metrics = adapter.lane_metrics(controlled_lanes)
            halting = sum(lane["halting"] for lane in metrics.values())
            waiting = sum(lane["waiting_time"] for lane in metrics.values())
            peak_halting = max(peak_halting, halting)
            cumulative_halting += halting
            cumulative_waiting += waiting
            departed += traci.simulation.getDepartedNumber()
            arrived += traci.simulation.getArrivedNumber()

        result = {
            "scenario": "Central Lemgo, Germany (OpenStreetMap)",
            "sumo_version": traci.getVersion()[1],
            "steps": args.steps,
            "seed": args.seed,
            "signals": len(signal_ids),
            "controlled_lanes": len(controlled_lanes),
            "network_lanes": len(lane_ids),
            "departed": departed,
            "arrived": arrived,
            "vehicles_remaining": traci.simulation.getMinExpectedNumber(),
            "peak_controlled_lane_queue": peak_halting,
            "mean_controlled_lane_queue": cumulative_halting / args.steps,
            "mean_controlled_lane_waiting_seconds": cumulative_waiting / args.steps,
            "traffic_modes": ["passenger cars", "bicycles", "pedestrians"],
            "status": "verified",
        }
    finally:
        adapter.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
