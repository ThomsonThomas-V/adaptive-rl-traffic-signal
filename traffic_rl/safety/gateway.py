"""Hardware-in-the-loop boundary that never directly drives signal heads.

The gateway validates proposed high-level phase requests and hands accepted
requests to a certified/native controller.  It deliberately exposes no raw
lamp control operation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SafetyResult:
    accepted: bool
    requested_phase: int | None
    reason: str
    fallback_to_native: bool = False


class SafetyGateway:
    """Validate RL requests before they reach a native traffic controller."""

    def __init__(self, minimum_green: float = 20.0, heartbeat_timeout: float = 2.0) -> None:
        self.minimum_green = minimum_green
        self.heartbeat_timeout = heartbeat_timeout
        self._phase: dict[str, int] = {}
        self._phase_started: dict[str, float] = {}
        self._heartbeat: float | None = None
        self.native_fallback = True

    def register_signal(self, signal_id: str, phase: int, now: float) -> None:
        if phase not in (0, 1):
            raise ValueError("Native phase class must be 0 or 1")
        self._phase[signal_id] = phase
        self._phase_started[signal_id] = now

    def heartbeat(self, now: float) -> None:
        self._heartbeat = now
        self.native_fallback = False

    def request(
        self,
        signal_id: str,
        desired_phase: int,
        now: float,
        protected_mode: str = "vehicle",
    ) -> SafetyResult:
        if signal_id not in self._phase:
            return SafetyResult(False, None, "unknown signal", True)
        if self._heartbeat is None or now - self._heartbeat > self.heartbeat_timeout:
            self.native_fallback = True
            return SafetyResult(False, None, "stale controller heartbeat", True)
        if desired_phase not in (0, 1):
            return SafetyResult(False, None, "invalid or conflicting phase request", True)
        if protected_mode != "vehicle":
            return SafetyResult(False, None, f"{protected_mode} interval is protected")
        if desired_phase == self._phase[signal_id]:
            return SafetyResult(True, desired_phase, "phase already active")
        if now - self._phase_started[signal_id] < self.minimum_green:
            return SafetyResult(False, None, "minimum green not satisfied")
        # Acceptance means request the native program's next transition. The
        # native controller remains responsible for yellow and all-red timing.
        return SafetyResult(True, desired_phase, "request native safe transition")

    def observe_native_phase(self, signal_id: str, phase: int, now: float) -> None:
        if signal_id not in self._phase or phase not in (0, 1):
            raise ValueError("Invalid native signal observation")
        if phase != self._phase[signal_id]:
            self._phase[signal_id] = phase
            self._phase_started[signal_id] = now

    def watchdog(self, now: float) -> bool:
        """Return True when the native controller must retain full authority."""
        if self._heartbeat is None or now - self._heartbeat > self.heartbeat_timeout:
            self.native_fallback = True
        return self.native_fallback
