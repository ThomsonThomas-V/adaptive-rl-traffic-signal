"""Select a trained policy on validation seeds before final testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "model_selection_300k.json"
    )
    args = parser.parse_args()
    candidates = []
    for path in args.inputs:
        document = json.loads(path.read_text(encoding="utf-8"))
        wait_ratios, overflow_ratios, queue_ratios = [], [], []
        for scenario in document["scenarios"].values():
            controllers = scenario["controllers"]
            candidate = controllers["shared_dqn"]["metrics"]
            baseline = controllers["queue_heuristic"]["metrics"]
            wait_ratios.append(candidate["vehicle_average_wait"]["mean"] / baseline["vehicle_average_wait"]["mean"])
            overflow_ratios.append((candidate["overflows"]["mean"] + 1) / (baseline["overflows"]["mean"] + 1))
            queue_ratios.append(candidate["max_queue"]["mean"] / baseline["max_queue"]["mean"])
        components = {
            "waiting_ratio": float(np.mean(wait_ratios)),
            "overflow_ratio": float(np.mean(overflow_ratios)),
            "max_queue_ratio": float(np.mean(queue_ratios)),
        }
        candidates.append(
            {
                "model": document["models"][0],
                "validation_seeds": document["evaluation_seeds"],
                "components": components,
                "score": float(np.mean(list(components.values()))),
            }
        )
    candidates.sort(key=lambda item: item["score"])
    result = {
        "status": "selected_before_final_test",
        "criterion": "equal-weight mean of wait, overflow, and maximum-queue ratios versus queue heuristic; lower is better",
        "selected_model": candidates[0]["model"],
        "candidates": candidates,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
