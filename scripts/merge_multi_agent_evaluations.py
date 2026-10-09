"""Merge per-scenario evaluation files into one verified result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "multi_agent_evaluation_300k.json",
    )
    args = parser.parse_args()
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    first = documents[0]
    result = {key: value for key, value in first.items() if key != "scenarios"}
    result["scenarios"] = {}
    for document in documents:
        if document["models"] != first["models"] or document["evaluation_seeds"] != first["evaluation_seeds"]:
            raise ValueError("Cannot merge evaluations with different models or seeds")
        overlap = set(result["scenarios"]).intersection(document["scenarios"])
        if overlap:
            raise ValueError(f"Duplicate scenarios: {sorted(overlap)}")
        result["scenarios"].update(document["scenarios"])
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {len(result['scenarios'])} scenarios into {args.output}")


if __name__ == "__main__":
    main()
