"""Tests for the cooperative 2x2 traffic grid."""

import numpy as np

from traffic_rl.envs.network_env import GridConfig, TrafficGridEnv, cooperative_actions


def test_grid_contract_and_local_agent_views() -> None:
    env = TrafficGridEnv(GridConfig(episode_length=3))
    observation, _ = env.reset(seed=7)
    assert observation.shape == (40,)
    assert env.observation_space.contains(observation)
    views = env.agent_observations()
    assert set(views) == {"A", "B", "C", "D"}
    assert all(value.shape == (10,) for value in views.values())
    action = cooperative_actions(views)
    assert env.action_space.contains(action)


def test_grid_reproducibility_and_truncation() -> None:
    config = GridConfig(episode_length=5)
    first, second = TrafficGridEnv(config), TrafficGridEnv(config)
    first.reset(seed=19)
    second.reset(seed=19)
    action = np.zeros(4, dtype=np.int64)
    for index in range(5):
        out_a, out_b = first.step(action), second.step(action)
        np.testing.assert_array_equal(out_a[0], out_b[0])
        assert out_a[3] is (index == 4)
    np.testing.assert_array_equal(first.queues, second.queues)


def test_vehicles_transfer_between_neighboring_nodes() -> None:
    env = TrafficGridEnv(
        GridConfig(
            boundary_arrival_rate=0,
            pedestrian_rate=0,
            bicycle_rate=0,
            service_rate=1,
            left_turn_ratio=0,
            right_turn_ratio=0,
        )
    )
    env.reset(seed=1)
    env.phases[:] = 1  # EW green
    env.queues[0, 3] = 1  # vehicle enters A from west and travels east to B
    env.step(np.zeros(4, dtype=np.int64))
    assert env.queues[0, 3] == 0
    assert env.queues[1, 3] == 1
