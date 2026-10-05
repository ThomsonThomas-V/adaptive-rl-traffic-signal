"""Behavioral and API tests for TrafficLightEnv."""

from __future__ import annotations

import numpy as np
import pytest

from traffic_rl.envs.traffic_config import TrafficConfig
from traffic_rl.envs.traffic_env import Phase, TrafficLightEnv


def zero_demand(**overrides: object) -> TrafficConfig:
    values: dict[str, object] = {
        "episode_length": 20,
        "north_arrival_rate": 0.0,
        "south_arrival_rate": 0.0,
        "east_arrival_rate": 0.0,
        "west_arrival_rate": 0.0,
        "maximum_green_time": None,
    }
    values.update(overrides)
    return TrafficConfig(**values)  # type: ignore[arg-type]


def test_reset_and_step_contract() -> None:
    env = TrafficLightEnv(zero_demand())
    observation, info = env.reset(seed=42)
    assert observation.shape == (15,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info["total_queue"] == 0
    output = env.step(0)
    assert len(output) == 5
    assert isinstance(output[1], float)
    assert output[2] is False
    assert env.observation_space.contains(output[0])


@pytest.mark.parametrize("action", [0, 1, 2])
def test_all_actions_are_valid(action: int) -> None:
    env = TrafficLightEnv(zero_demand())
    env.reset(seed=1)
    env.step(action)


def test_invalid_action_raises() -> None:
    env = TrafficLightEnv(zero_demand())
    env.reset()
    with pytest.raises(ValueError):
        env.step(3)


def test_fixed_seed_is_reproducible() -> None:
    config = TrafficConfig(episode_length=30)
    first = TrafficLightEnv(config)
    second = TrafficLightEnv(config)
    observations_a = [first.reset(seed=123)[0]]
    observations_b = [second.reset(seed=123)[0]]
    for action in [0, 0, 2, 0, 1] * 5:
        observations_a.append(first.step(action)[0])
        observations_b.append(second.step(action)[0])
    np.testing.assert_array_equal(observations_a, observations_b)
    np.testing.assert_array_equal(first.queues, second.queues)


def test_arrival_stream_does_not_depend_on_actions() -> None:
    config = TrafficConfig(episode_length=20, minimum_green_time=0)
    first = TrafficLightEnv(config)
    second = TrafficLightEnv(config)
    first.reset(seed=77)
    second.reset(seed=77)
    arrivals_a, arrivals_b = [], []
    for index in range(20):
        arrivals_a.append(first.step(0)[4]["arrivals"])
        arrivals_b.append(second.step(index % 3)[4]["arrivals"])
    np.testing.assert_array_equal(arrivals_a, arrivals_b)


def test_zero_arrivals_leave_queues_empty() -> None:
    env = TrafficLightEnv(zero_demand())
    env.reset(seed=9)
    for _ in range(10):
        env.step(0)
    assert np.all(env.queues == 0)


def test_high_arrival_rate_adds_vehicles_without_negative_queues() -> None:
    config = TrafficConfig(
        episode_length=4,
        north_arrival_rate=20.0,
        south_arrival_rate=0.0,
        east_arrival_rate=0.0,
        west_arrival_rate=0.0,
        service_rate=0.1,
    )
    env = TrafficLightEnv(config)
    env.reset(seed=4)
    env.step(0)
    assert env.total_arrived > 0
    assert env.queues[0] > 0
    assert np.all(env.queues >= 0)


def test_green_discharges_only_active_axis() -> None:
    env = TrafficLightEnv(zero_demand(service_rate=1.0))
    env.reset(seed=0)
    env.queues[:] = [2, 2, 2, 2]
    env.step(0)
    np.testing.assert_array_equal(env.queues, [1, 1, 2, 2])


def test_minimum_green_and_yellow_transition() -> None:
    env = TrafficLightEnv(
        zero_demand(minimum_green_time=2, yellow_duration=2)
    )
    env.reset(seed=0)
    _, _, _, _, info = env.step(2)
    assert info["illegal_switch"] is True
    env.step(0)
    _, _, _, _, info = env.step(2)
    assert info["signal_switched"] is True
    assert env.in_yellow
    assert env.current_phase == Phase.NS_GREEN
    env.step(0)
    assert not env.in_yellow
    assert env.current_phase == Phase.EW_GREEN
    assert env.signal_switches == 1


def test_episode_uses_truncation_and_final_metrics() -> None:
    env = TrafficLightEnv(zero_demand(episode_length=3))
    env.reset(seed=0)
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated and not truncated
    _, _, terminated, truncated, info = env.step(0)
    assert not terminated and truncated
    assert info["average_waiting_time"] == 0
    assert info["throughput"] == 0
    assert "completion_ratio" in info


def test_environment_passes_sb3_checker() -> None:
    sb3 = pytest.importorskip("stable_baselines3.common.env_checker")
    sb3.check_env(TrafficLightEnv(zero_demand()), warn=True)
