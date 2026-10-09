"""Run the offline hardware-in-the-loop safety contract validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT
from traffic_rl.safety import SafetyGateway


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "hil_safety_validation.json"
    )
    args = parser.parse_args()
    gateway = SafetyGateway(minimum_green=20.0, heartbeat_timeout=2.0)
    gateway.register_signal("OWL322", phase=0, now=0.0)
    gateway.heartbeat(0.0)

    invalid = gateway.request("OWL322", 7, 1.0)
    gateway.heartbeat(10.0)
    early = gateway.request("OWL322", 1, 10.0)
    gateway.heartbeat(21.0)
    protected = gateway.request("OWL322", 1, 21.0, protected_mode="pedestrian")
    checks = {
        "reject_conflicting_raw_phase": invalid,
        "reject_early_switch": early,
        "reject_pedestrian_interval_switch": protected,
    }
    checks["accept_safe_high_level_request"] = gateway.request("OWL322", 1, 21.0)
    checks["watchdog_native_fallback"] = gateway.watchdog(24.1)

    passed = {
        "reject_conflicting_raw_phase": not checks["reject_conflicting_raw_phase"].accepted,
        "reject_early_switch": not checks["reject_early_switch"].accepted,
        "reject_pedestrian_interval_switch": not checks[
            "reject_pedestrian_interval_switch"
        ].accepted,
        "accept_safe_high_level_request": checks[
            "accept_safe_high_level_request"
        ].accepted,
        "watchdog_native_fallback": bool(checks["watchdog_native_fallback"]),
    }
    result = {
        "status": "passed" if all(passed.values()) else "failed",
        "scope": "offline safety gateway; no connection to live signal hardware",
        "checks": passed,
        "invariants": [
            "RL cannot command individual lamp states",
            "native controller owns yellow and all-red transitions",
            "protected pedestrian intervals cannot be interrupted",
            "stale heartbeat returns authority to native control",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
