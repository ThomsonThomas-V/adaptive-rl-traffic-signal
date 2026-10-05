"""Configuration and predefined demand profiles for the traffic simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class TrafficConfig:
    """All physical, demand, observation, and reward parameters."""

    episode_length: int = 1_200
    north_arrival_rate: float = 0.20
    south_arrival_rate: float = 0.20
    east_arrival_rate: float = 0.20
    west_arrival_rate: float = 0.20
    scenario: str = "medium"

    minimum_green_time: int = 5
    maximum_green_time: int | None = 90
    yellow_duration: int = 3
    service_rate: float = 0.80
    max_queue_length: int = 100
    arrival_window: int = 30

    queue_penalty: float = 0.06
    waiting_penalty: float = 0.02
    throughput_reward: float = 1.0
    switching_penalty: float = 0.25
    illegal_switch_penalty: float = 0.05
    starvation_penalty: float = 0.5
    starvation_threshold: int = 90

    def __post_init__(self) -> None:
        rates = self.base_arrival_rates
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if any(rate < 0 for rate in rates):
            raise ValueError("arrival rates must be non-negative")
        if self.minimum_green_time < 0 or self.yellow_duration < 0:
            raise ValueError("signal durations must be non-negative")
        if self.maximum_green_time is not None:
            if self.maximum_green_time < self.minimum_green_time:
                raise ValueError("maximum_green_time must be >= minimum_green_time")
        if not 0 < self.service_rate:
            raise ValueError("service_rate must be positive")
        if self.max_queue_length <= 0 or self.arrival_window <= 0:
            raise ValueError("queue length and arrival window must be positive")

    @property
    def base_arrival_rates(self) -> tuple[float, float, float, float]:
        """Return arrival rates ordered north, south, east, west."""
        return (
            self.north_arrival_rate,
            self.south_arrival_rate,
            self.east_arrival_rate,
            self.west_arrival_rate,
        )

    def arrival_rates_at(self, timestep: int) -> tuple[float, float, float, float]:
        """Return causal rates for a timestep; the rush profile scales to episode length."""
        if self.scenario != "rush_hour":
            return self.base_arrival_rates
        progress = timestep / self.episode_length
        if progress < 0.25:
            return (0.08, 0.08, 0.08, 0.08)
        if progress < 7 / 12:
            return (0.38, 0.38, 0.10, 0.10)
        if progress < 5 / 6:
            return (0.10, 0.10, 0.38, 0.38)
        return (0.20, 0.20, 0.20, 0.20)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration mapping."""
        return asdict(self)


SCENARIO_RATES: dict[str, tuple[float, float, float, float]] = {
    "low": (0.08, 0.08, 0.08, 0.08),
    "medium": (0.20, 0.20, 0.20, 0.20),
    "heavy": (0.34, 0.34, 0.34, 0.34),
    "unbalanced": (0.36, 0.36, 0.09, 0.09),
    "rush_hour": (0.08, 0.08, 0.08, 0.08),
}


def get_scenario(name: str, **overrides: Any) -> TrafficConfig:
    """Build a validated configuration for a named demand scenario."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in SCENARIO_RATES:
        choices = ", ".join(SCENARIO_RATES)
        raise ValueError(f"Unknown scenario {name!r}; choose from {choices}")
    north, south, east, west = SCENARIO_RATES[key]
    config = TrafficConfig(
        scenario=key,
        north_arrival_rate=north,
        south_arrival_rate=south,
        east_arrival_rate=east,
        west_arrival_rate=west,
    )
    return replace(config, **overrides)
