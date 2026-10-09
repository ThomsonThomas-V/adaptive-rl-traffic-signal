"""Add confidence intervals and paired comparisons to a SUMO evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT
import numpy as np


METRICS = (
    "arrived",
    "vehicles_remaining",
    "mean_queue",
    "peak_queue",
    "mean_waiting_seconds",
    "collisions",
)


def statistics(values: np.ndarray) -> dict[str, float]:
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    # Student t critical value for five runs (df=4); conservative for fewer runs.
    critical = 2.776 if len(values) == 5 else 1.96
    half_width = critical * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95_low": mean - half_width, "ci95_high": mean + half_width}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "sumo_policy_evaluation_300k.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    controllers = sorted({str(run["controller"]) for run in data["runs"]})
    data["summary_statistics"] = {
        controller: {
            metric: statistics(
                np.asarray(
                    [float(run[metric]) for run in data["runs"] if run["controller"] == controller]
                )
            )
            for metric in METRICS
        }
        for controller in controllers
    }
    if {"native", "shared_dqn"}.issubset(controllers):
        by_controller = {
            controller: {
                int(run["seed"]): run
                for run in data["runs"]
                if run["controller"] == controller
            }
            for controller in controllers
        }
        seeds = sorted(set(by_controller["native"]).intersection(by_controller["shared_dqn"]))
        data["paired_dqn_vs_native"] = {}
        for metric in METRICS:
            native = np.asarray([float(by_controller["native"][seed][metric]) for seed in seeds])
            dqn = np.asarray([float(by_controller["shared_dqn"][seed][metric]) for seed in seeds])
            differences = dqn - native
            stats = statistics(differences)
            stats["improvement_percent"] = float(
                100 * (native.mean() - dqn.mean()) / max(abs(native.mean()), 1e-9)
            )
            stats["statistically_significant_95"] = bool(
                stats["ci95_high"] < 0 or stats["ci95_low"] > 0
            )
            data["paired_dqn_vs_native"][metric] = stats
    data["statistical_note"] = "Paired by SUMO seed; 95% Student-t intervals for five held-out seeds."
    output = args.output or args.input
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Added statistical summary to {output}")


if __name__ == "__main__":
    main()
