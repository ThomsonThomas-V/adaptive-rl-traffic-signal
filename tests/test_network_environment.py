"""Tests for the cooperative 2x2 traffic grid."""

import numpy as np

from traffic_rl.envs.network_env import GridConfig, TrafficGridEnv, cooperative_actions


def test_grid_contract_and_local_agent_views() -> None:
    env = TrafficGridEnv(GridConfig(episode_length=3))
    observation, _ = env.reset(seed=7)
    assert observation.shape == (44,)
    assert env.observation_space.contains(observation)
    views = env.agent_observations()
    assert set(views) == {"A", "B", "C", "D"}
    assert all(value.shape == (11,) for value in views.values())
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


def test_protected_pedestrian_and_clearance_phases_are_conflict_free() -> None:
    env = TrafficGridEnv(
        GridConfig(
            boundary_arrival_rate=0,
            pedestrian_rate=0,
            bicycle_rate=0,
            service_rate=1,
            pedestrian_service_rate=2,
            bicycle_service_rate=1,
        )
    )
    env.reset(seed=4)
    env.queues[:, 0] = 3
    env.pedestrians[:] = 3
    env.bicycles[:] = 2

    env.timestep = 23
    queues_before = env.queues.copy()
    pedestrians_before = env.pedestrians.copy()
    bicycles_before = env.bicycles.copy()
    env.step(np.zeros(4, dtype=np.int64))
    np.testing.assert_array_equal(env.queues, queues_before)
    np.testing.assert_array_equal(env.pedestrians, pedestrians_before)
    np.testing.assert_array_equal(env.bicycles, bicycles_before)

    env.step(np.zeros(4, dtype=np.int64))
    np.testing.assert_array_equal(env.queues, queues_before)
    np.testing.assert_array_equal(env.pedestrians, pedestrians_before - 2)
    np.testing.assert_array_equal(env.bicycles, bicycles_before)
    assert env.conflicts == 0


def test_bicycles_move_only_during_vehicle_service() -> None:
    env = TrafficGridEnv(
        GridConfig(
            boundary_arrival_rate=0,
            pedestrian_rate=0,
            bicycle_rate=0,
            service_rate=0,
            bicycle_service_rate=1,
        )
    )
    env.reset(seed=5)
    env.bicycles[:] = 2
    env.timestep = 0
    env.step(np.zeros(4, dtype=np.int64))
    np.testing.assert_array_equal(env.bicycles, np.ones(4, dtype=np.int64))
    assert env.bicycles_served == 4


def test_road_closure_blocks_only_the_configured_movement() -> None:
    env = TrafficGridEnv(
        GridConfig(
            boundary_arrival_rate=0,
            pedestrian_rate=0,
            bicycle_rate=0,
            service_rate=1,
            incident_node=1,
            incident_approach=3,
            incident_start=0,
            incident_end=10,
        )
    )
    env.reset(seed=6)
    env.phases[:] = 1
    env.queues[1, 2:4] = 2
    env.step(np.zeros(4, dtype=np.int64))
    assert env.queues[1, 2] == 1
    assert env.queues[1, 3] == 2


def test_sensor_failure_masks_observation_but_not_physical_queue() -> None:
    env = TrafficGridEnv(
        GridConfig(
            boundary_arrival_rate=0,
            pedestrian_rate=0,
            bicycle_rate=0,
            sensor_failure_node=2,
            sensor_failure_start=0,
            sensor_failure_end=20,
        )
    )
    env.reset(seed=7)
    env.queues[2] = 10
    observation = env.agent_observations()["C"]
    np.testing.assert_array_equal(observation[:4], np.zeros(4))
    np.testing.assert_array_equal(env.queues[2], np.full(4, 10))


def test_emergency_vehicle_is_tracked_and_served_with_priority() -> None:
    env = TrafficGridEnv(
        GridConfig(
            boundary_arrival_rate=0,
            pedestrian_rate=0,
            bicycle_rate=0,
            service_rate=1,
        )
    )
    env.reset(seed=8)
    env.queues[2, 0] = 2
    env.emergencies[2, 0] = 1
    env.step(np.zeros(4, dtype=np.int64))
    assert env.emergencies[2, 0] == 0
    assert env.emergencies_served == 1
