"""Named demand and fault scenarios for the cooperative traffic grid."""

from __future__ import annotations

from dataclasses import replace

from traffic_rl.envs.network_env import GridConfig


SCENARIOS = (
    "balanced",
    "rush_hour",
    "unbalanced",
    "road_closure",
    "emergency_surge",
    "sensor_failure",
    "calibrated_lemgo",
)


def grid_scenario(name: str, episode_length: int = 1_200) -> GridConfig:
    """Return a reproducible configuration for a named evaluation scenario."""
    base = GridConfig(episode_length=episode_length, demand_profile=name)
    if name == "balanced":
        return base
    if name == "rush_hour":
        return replace(base, boundary_arrival_rate=0.18, pedestrian_rate=0.035, bicycle_rate=0.045)
    if name == "unbalanced":
        return replace(base, boundary_arrival_rate=0.18)
    if name == "road_closure":
        return replace(
            base,
            boundary_arrival_rate=0.18,
            incident_node=1,
            incident_approach=3,
            incident_start=360,
            incident_end=840,
        )
    if name == "emergency_surge":
        return replace(base, boundary_arrival_rate=0.18, emergency_rate=0.012)
    if name == "sensor_failure":
        return replace(
            base,
            sensor_failure_node=2,
            sensor_failure_start=300,
            sensor_failure_end=900,
        )
    if name == "calibrated_lemgo":
        # Rush-hour shape follows the published OWL322 LemgoRL demand model.
        # Absolute rates are scaled to this compact 2x2 abstraction.
        return replace(
            base,
            boundary_arrival_rate=0.225,
            pedestrian_rate=0.018,
            bicycle_rate=0.003,
            left_turn_ratio=0.18,
            right_turn_ratio=0.12,
        )
    raise ValueError(f"Unknown grid scenario: {name}")
