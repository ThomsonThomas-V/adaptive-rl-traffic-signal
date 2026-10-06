"""A compact 2x2 cooperative traffic-signal network environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class GridConfig:
    """Physical and demand settings for a four-intersection grid."""

    episode_length: int = 1_200
    boundary_arrival_rate: float = 0.16
    service_rate: float = 0.80
    left_turn_ratio: float = 0.15
    right_turn_ratio: float = 0.10
    pedestrian_rate: float = 0.025
    bicycle_rate: float = 0.035
    minimum_green_time: int = 5
    yellow_duration: int = 3
    max_queue_length: int = 80


class TrafficGridEnv(gym.Env[NDArray[np.float32], NDArray[np.int64]]):
    """Centralized Gymnasium interface for four cooperative signal agents.

    Intersections are ordered A/B on the north row and C/D on the south row.
    Each node receives an independent action: keep, request NS, or request EW.
    Straight-through departures move to the next node when one exists; turning
    traffic exits the small modeled district. Pedestrian and bicycle demand is
    observed and contributes to the network delay objective.
    """

    metadata = {"render_modes": []}
    NODE_NAMES = ("A", "B", "C", "D")

    def __init__(self, config: GridConfig | None = None) -> None:
        super().__init__()
        self.config = config or GridConfig()
        self.action_space = spaces.MultiDiscrete(np.full(4, 3, dtype=np.int64))
        # Per node: four queues, pedestrian and bicycle demand, phase, phase
        # age, and horizontal/vertical neighbor pressure = ten values.
        self.observation_space = spaces.Box(0.0, 1.0, shape=(40,), dtype=np.float32)
        self.queues = np.zeros((4, 4), dtype=np.int64)
        self.pedestrians = np.zeros(4, dtype=np.int64)
        self.bicycles = np.zeros(4, dtype=np.int64)
        self.phases = np.zeros(4, dtype=np.int64)
        self.phase_age = np.zeros(4, dtype=np.int64)
        self.yellow = np.zeros(4, dtype=np.int64)
        self.pending = np.full(4, -1, dtype=np.int64)
        self._credits = np.zeros((4, 4), dtype=np.float64)
        self._rng = np.random.default_rng()
        self.timestep = 0
        self.arrived = 0
        self.processed = 0
        self.total_delay = 0.0
        self.switches = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        for array in (
            self.queues,
            self.pedestrians,
            self.bicycles,
            self.phases,
            self.phase_age,
            self.yellow,
            self._credits,
        ):
            array.fill(0)
        self.pending.fill(-1)
        self.timestep = self.arrived = self.processed = self.switches = 0
        self.total_delay = 0.0
        return self._observation(), self._info(False)

    def step(
        self, action: NDArray[np.int64]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(action):
            raise ValueError("Grid action must contain four values in {0, 1, 2}")
        self._apply_actions(action)
        self._boundary_arrivals()
        self.total_delay += float(
            self.queues.sum() + self.pedestrians.sum() + self.bicycles.sum()
        )
        passed = self._advance_network()
        vulnerable_wait = int(self.pedestrians.sum() + self.bicycles.sum())
        reward = -0.045 * float(self.queues.sum()) - 0.025 * vulnerable_wait + passed
        self.timestep += 1
        truncated = self.timestep >= self.config.episode_length
        return self._observation(), reward, False, truncated, self._info(truncated)

    def _apply_actions(self, actions: NDArray[np.int64]) -> None:
        for node, action in enumerate(actions):
            if self.yellow[node] > 0:
                continue
            requested = int(action) - 1 if action else int(self.phases[node])
            if requested == self.phases[node] or self.phase_age[node] < self.config.minimum_green_time:
                continue
            self.pending[node] = requested
            self.yellow[node] = self.config.yellow_duration
            self.switches += 1

    def _boundary_arrivals(self) -> None:
        # (node, approach) pairs on the outside edge of the district.
        boundaries = ((0, 0), (1, 0), (2, 1), (3, 1), (0, 3), (2, 3), (1, 2), (3, 2))
        for node, approach in boundaries:
            count = int(self._rng.poisson(self.config.boundary_arrival_rate))
            self.queues[node, approach] = min(
                self.config.max_queue_length, self.queues[node, approach] + count
            )
            self.arrived += count
        self.pedestrians += self._rng.poisson(self.config.pedestrian_rate, 4)
        self.bicycles += self._rng.poisson(self.config.bicycle_rate, 4)

    def _advance_network(self) -> int:
        transfers: list[tuple[int, int]] = []
        processed = 0
        for node in range(4):
            if self.yellow[node] > 0:
                self.yellow[node] -= 1
                if self.yellow[node] == 0:
                    self.phases[node] = self.pending[node]
                    self.pending[node] = -1
                    self.phase_age[node] = 0
                continue
            active = (0, 1) if self.phases[node] == 0 else (2, 3)
            self.phase_age[node] += 1
            for approach in active:
                self._credits[node, approach] += self.config.service_rate
                capacity = int(self._credits[node, approach])
                departing = min(capacity, int(self.queues[node, approach]))
                self.queues[node, approach] -= departing
                self._credits[node, approach] -= departing
                for _ in range(departing):
                    if self._rng.random() < self.config.left_turn_ratio + self.config.right_turn_ratio:
                        processed += 1
                        continue
                    downstream = self._downstream(node, approach)
                    if downstream is None:
                        processed += 1
                    else:
                        transfers.append(downstream)
            # Vulnerable users cross the stopped axis in small batches.
            cross_capacity = 2
            self.pedestrians[node] = max(0, self.pedestrians[node] - cross_capacity)
            self.bicycles[node] = max(0, self.bicycles[node] - cross_capacity)
        for node, approach in transfers:
            if self.queues[node, approach] < self.config.max_queue_length:
                self.queues[node, approach] += 1
            else:
                processed += 1
        self.processed += processed
        return processed

    @staticmethod
    def _downstream(node: int, approach: int) -> tuple[int, int] | None:
        row, column = divmod(node, 2)
        # N traffic goes south, S north, E west, W east.
        targets = {
            0: (row + 1, column, 0),
            1: (row - 1, column, 1),
            2: (row, column - 1, 2),
            3: (row, column + 1, 3),
        }
        next_row, next_column, inbound = targets[approach]
        if 0 <= next_row < 2 and 0 <= next_column < 2:
            return next_row * 2 + next_column, inbound
        return None

    def _observation(self) -> NDArray[np.float32]:
        features: list[float] = []
        for node in range(4):
            row, column = divmod(node, 2)
            horizontal = sum(self.queues[row * 2 + c, [2, 3]].sum() for c in range(2))
            vertical = sum(self.queues[r * 2 + column, [0, 1]].sum() for r in range(2))
            features.extend(np.clip(self.queues[node] / self.config.max_queue_length, 0, 1))
            features.extend(
                (
                    min(self.pedestrians[node] / 20, 1),
                    min(self.bicycles[node] / 20, 1),
                    float(self.phases[node]),
                    min(self.phase_age[node] / 90, 1),
                    min(horizontal / (4 * self.config.max_queue_length), 1),
                    min(vertical / (4 * self.config.max_queue_length), 1),
                )
            )
        return np.asarray(features, dtype=np.float32)

    def agent_observations(self) -> dict[str, NDArray[np.float32]]:
        """Return the centralized observation split into four local agent views."""
        observation = self._observation().reshape(4, 10)
        return {name: observation[index].copy() for index, name in enumerate(self.NODE_NAMES)}

    def _info(self, final: bool) -> dict[str, Any]:
        info: dict[str, Any] = {
            "timestep": self.timestep,
            "queues": self.queues.copy(),
            "pedestrians": self.pedestrians.copy(),
            "bicycles": self.bicycles.copy(),
            "phases": self.phases.copy(),
            "arrived": self.arrived,
            "processed": self.processed,
            "signal_switches": self.switches,
        }
        if final:
            info.update(
                average_waiting_time=self.total_delay / max(self.arrived, 1),
                throughput=self.processed,
                completion_ratio=self.processed / max(self.arrived, 1),
            )
        return info


def cooperative_actions(observations: dict[str, NDArray[np.float32]]) -> NDArray[np.int64]:
    """Reference decentralized max-pressure policy for the four local agents."""
    actions: list[int] = []
    for observation in observations.values():
        ns = float(observation[0] + observation[1])
        ew = float(observation[2] + observation[3])
        phase = int(round(float(observation[6])))
        desired = 0 if ns >= ew else 1
        actions.append(0 if desired == phase else desired + 1)
    return np.asarray(actions, dtype=np.int64)
