"""Optional TraCI bridge for running learned policies against SUMO networks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import Any


@dataclass(frozen=True, slots=True)
class SumoConfig:
    """Files and runtime options for an existing SUMO scenario."""

    sumocfg: Path
    gui: bool = False
    step_length: float = 1.0
    seed: int = 42


class SumoAdapter:
    """Small production-oriented wrapper around SUMO's TraCI interface.

    SUMO is intentionally optional. Install SUMO and its Python tools, then
    supply a valid `.sumocfg` generated from a German OSM extract or netedit.
    """

    def __init__(self, config: SumoConfig) -> None:
        self.config = config
        self.traci: Any | None = None

    def start(self) -> None:
        if not self.config.sumocfg.exists():
            raise FileNotFoundError(self.config.sumocfg)
        try:
            import traci  # type: ignore[import-not-found]
        except ImportError as error:
            try:
                import sumo  # type: ignore[import-not-found]

                tools_dir = str(Path(sumo.SUMO_HOME) / "tools")
                if tools_dir not in sys.path:
                    sys.path.insert(0, tools_dir)
                import traci  # type: ignore[import-not-found,no-redef]
            except ImportError:
                raise RuntimeError(
                    "SUMO/TraCI is not installed. Follow https://sumo.dlr.de/docs/Installing/"
                ) from error
        binary = self._resolve_binary()
        traci.start(
            [
                binary,
                "-c",
                str(self.config.sumocfg),
                "--step-length",
                str(self.config.step_length),
                "--seed",
                str(self.config.seed),
                "--no-warnings",
                "true",
            ]
        )
        self.traci = traci

    def _resolve_binary(self) -> str:
        """Locate SUMO from PATH or the official ``eclipse-sumo`` wheel."""
        name = "sumo-gui" if self.config.gui else "sumo"
        path_binary = shutil.which(name)
        if path_binary is not None:
            return path_binary

        try:
            import sumo  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "SUMO is not on PATH and the eclipse-sumo package is not installed"
            ) from error

        suffix = ".exe" if shutil.which("where") is not None else ""
        wheel_binary = Path(sumo.SUMO_HOME) / "bin" / f"{name}{suffix}"
        if not wheel_binary.exists():
            raise RuntimeError(f"Could not find the SUMO binary at {wheel_binary}")
        return str(wheel_binary)

    def step(self) -> None:
        self._require_started().simulationStep()

    def lane_metrics(self, lane_ids: list[str]) -> dict[str, dict[str, float]]:
        traci = self._require_started()
        return {
            lane: {
                "halting": float(traci.lane.getLastStepHaltingNumber(lane)),
                "vehicles": float(traci.lane.getLastStepVehicleNumber(lane)),
                "mean_speed": float(traci.lane.getLastStepMeanSpeed(lane)),
                "waiting_time": float(traci.lane.getWaitingTime(lane)),
            }
            for lane in lane_ids
        }

    def set_phase(self, signal_id: str, phase_index: int) -> None:
        self._require_started().trafficlight.setPhase(signal_id, phase_index)

    def signal_ids(self) -> list[str]:
        return list(self._require_started().trafficlight.getIDList())

    def controlled_lanes(self, signal_id: str) -> list[str]:
        return sorted(set(self._require_started().trafficlight.getControlledLanes(signal_id)))

    def request_next_safe_phase(self, signal_id: str, transition_seconds: float = 1.0) -> bool:
        """Ask SUMO's native program to advance, preserving its yellow/all-red phases.

        Direct green-to-green phase jumps are deliberately avoided. The request
        only shortens the current green; SUMO then follows the transition phases
        already encoded in the signal program.
        """
        traci = self._require_started()
        state = str(traci.trafficlight.getRedYellowGreenState(signal_id))
        if "y" in state.lower() or not any(value in state for value in "Gg"):
            return False
        traci.trafficlight.setPhaseDuration(signal_id, float(transition_seconds))
        return True

    def collision_count(self) -> int:
        simulation = self._require_started().simulation
        getter = getattr(simulation, "getCollidingVehiclesNumber", None)
        return int(getter()) if getter is not None else 0

    def close(self) -> None:
        if self.traci is not None:
            self.traci.close()
            self.traci = None

    def _require_started(self) -> Any:
        if self.traci is None:
            raise RuntimeError("Call start() before using the SUMO adapter")
        return self.traci
