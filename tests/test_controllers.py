"""Unit tests for baseline controller decisions."""

import numpy as np

from traffic_rl.controllers import FixedTimeController, QueueHeuristicController, RandomController


def observation(queues: tuple[float, float, float, float]) -> np.ndarray:
    value = np.zeros(15, dtype=np.float32)
    value[:4] = queues
    return value


def test_fixed_controller_alternates() -> None:
    controller = FixedTimeController(green_duration=2)
    controller.reset()
    assert controller.predict(observation((0, 0, 0, 0))) == 0
    assert controller.predict(observation((0, 0, 0, 0))) == 2
    assert controller.predict(observation((0, 0, 0, 0))) == 0
    assert controller.predict(observation((0, 0, 0, 0))) == 1


def test_heuristic_prefers_larger_opposing_queue() -> None:
    controller = QueueHeuristicController(minimum_green_time=1)
    controller.reset()
    assert controller.predict(observation((0.01, 0.01, 0.5, 0.5))) == 2


def test_heuristic_hysteresis_avoids_small_difference() -> None:
    controller = QueueHeuristicController(switching_threshold=0.20, minimum_green_time=1)
    controller.reset()
    assert controller.predict(observation((0.25, 0.25, 0.29, 0.29))) == 0


def test_random_controller_is_seeded() -> None:
    first = RandomController(seed=5)
    second = RandomController(seed=5)
    sample = observation((0, 0, 0, 0))
    assert [first.predict(sample) for _ in range(20)] == [second.predict(sample) for _ in range(20)]
