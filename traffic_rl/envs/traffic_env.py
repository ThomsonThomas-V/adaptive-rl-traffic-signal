"""A lightweight, reproducible four-way traffic intersection environment."""

from __future__ import annotations

from collections import deque
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from traffic_rl.envs.traffic_config import TrafficConfig


class Phase(IntEnum):
    """Compatible simultaneous traffic movements."""

    NS_GREEN = 0
    EW_GREEN = 1


class Action(IntEnum):
    """Actions exposed to an agent."""

    KEEP = 0
    SET_NS_GREEN = 1
    SET_EW_GREEN = 2


class TrafficLightEnv(gym.Env[NDArray[np.float32], int]):
    """Gymnasium environment for one isolated signalized intersection.

    Arrivals use a dedicated RNG. Consequently, controllers evaluated with the
    same reset seed see the same exogenous demand, independent of their actions.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: TrafficConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TrafficConfig()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        # queues, directional cumulative delay, phase, phase age, yellow flag,
        # and four recent empirically observed arrival rates.
        self.observation_space = spaces.Box(
            low=np.zeros(15, dtype=np.float32),
            high=np.ones(15, dtype=np.float32),
            dtype=np.float32,
        )
        self._renderer: Any | None = None
        self._arrival_rng = np.random.default_rng()
        self.queues = np.zeros(4, dtype=np.int64)
        self.directional_delay = np.zeros(4, dtype=np.float64)
        self.oldest_wait = np.zeros(4, dtype=np.int64)
        self._service_credit = np.zeros(4, dtype=np.float64)
        self._arrival_history: deque[NDArray[np.int64]] = deque(
            maxlen=self.config.arrival_window
        )
        self.timestep = 0
        self.current_phase = Phase.NS_GREEN
        self.pending_phase: Phase | None = None
        self.green_elapsed = 0
        self.yellow_remaining = 0
        self.total_arrived = 0
        self.total_processed = 0
        self.total_overflow = 0
        self.total_delay = 0.0
        self.queue_sum = 0.0
        self.max_observed_queue = 0
        self.signal_switches = 0
        self.last_reward = 0.0

    @property
    def in_yellow(self) -> bool:
        """Whether the intersection is in an all-stop safety transition."""
        return self.yellow_remaining > 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        # SeedSequence-derived streams keep demand isolated from future internals.
        seed_sequence = np.random.SeedSequence(seed)
        arrival_seed, _ = seed_sequence.spawn(2)
        self._arrival_rng = np.random.default_rng(arrival_seed)
        self.queues.fill(0)
        self.directional_delay.fill(0.0)
        self.oldest_wait.fill(0)
        self._service_credit.fill(0.0)
        self._arrival_history.clear()
        self.timestep = 0
        initial_phase = (options or {}).get("initial_phase", Phase.NS_GREEN)
        self.current_phase = Phase(int(initial_phase))
        self.pending_phase = None
        self.green_elapsed = 0
        self.yellow_remaining = 0
        self.total_arrived = 0
        self.total_processed = 0
        self.total_overflow = 0
        self.total_delay = 0.0
        self.queue_sum = 0.0
        self.max_observed_queue = 0
        self.signal_switches = 0
        self.last_reward = 0.0
        return self._observation(), self._info(episode_end=False)

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0, 1, or 2")

        switched, illegal_switch = self._apply_action(Action(action))
        arrivals = self._sample_arrivals()
        available = self.config.max_queue_length - self.queues
        admitted = np.minimum(arrivals, available)
        overflow = arrivals - admitted
        self.queues += admitted
        self.total_arrived += int(arrivals.sum())
        self.total_overflow += int(overflow.sum())
        self._arrival_history.append(arrivals)

        # Queue integral: one queued vehicle for one step equals one vehicle-second.
        self.directional_delay += self.queues
        self.total_delay += float(self.queues.sum())
        self.oldest_wait = np.where(self.queues > 0, self.oldest_wait + 1, 0)

        processed = self._discharge()
        total_queue = int(self.queues.sum())
        starved = int(np.count_nonzero(self.oldest_wait > self.config.starvation_threshold))
        reward = (
            -self.config.queue_penalty * total_queue
            -self.config.waiting_penalty * float(self.oldest_wait.sum())
            +self.config.throughput_reward * processed
            -self.config.switching_penalty * int(switched)
            -self.config.illegal_switch_penalty * int(illegal_switch)
            -self.config.starvation_penalty * starved
        )
        self.last_reward = float(reward)
        self.queue_sum += total_queue
        self.max_observed_queue = max(self.max_observed_queue, int(self.queues.max()))
        self.timestep += 1
        truncated = self.timestep >= self.config.episode_length
        observation = self._observation()
        info = self._info(episode_end=truncated)
        info.update(
            {
                "vehicles_passed": processed,
                "arrivals": arrivals.copy(),
                "signal_switched": switched,
                "illegal_switch": illegal_switch,
            }
        )
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), False, truncated, info

    def _apply_action(self, action: Action) -> tuple[bool, bool]:
        if self.in_yellow:
            return False, False
        requested: Phase | None = None
        if action == Action.SET_NS_GREEN:
            requested = Phase.NS_GREEN
        elif action == Action.SET_EW_GREEN:
            requested = Phase.EW_GREEN

        forced = (
            self.config.maximum_green_time is not None
            and self.green_elapsed >= self.config.maximum_green_time
        )
        if forced:
            requested = Phase(1 - int(self.current_phase))
        if requested is None or requested == self.current_phase:
            return False, False
        if self.green_elapsed < self.config.minimum_green_time:
            return False, True
        self.pending_phase = requested
        self.signal_switches += 1
        if self.config.yellow_duration == 0:
            self.current_phase = requested
            self.pending_phase = None
            self.green_elapsed = 0
        else:
            self.yellow_remaining = self.config.yellow_duration
        return True, False

    def _sample_arrivals(self) -> NDArray[np.int64]:
        rates = np.asarray(self.config.arrival_rates_at(self.timestep), dtype=np.float64)
        return self._arrival_rng.poisson(rates).astype(np.int64)

    def _discharge(self) -> int:
        if self.in_yellow:
            self._service_credit.fill(0.0)
            self.yellow_remaining -= 1
            if self.yellow_remaining == 0 and self.pending_phase is not None:
                self.current_phase = self.pending_phase
                self.pending_phase = None
                self.green_elapsed = 0
            return 0

        active = (0, 1) if self.current_phase == Phase.NS_GREEN else (2, 3)
        inactive = (2, 3) if self.current_phase == Phase.NS_GREEN else (0, 1)
        self._service_credit[list(inactive)] = 0.0
        processed = 0
        for index in active:
            self._service_credit[index] += self.config.service_rate
            capacity = int(self._service_credit[index])
            departed = min(capacity, int(self.queues[index]))
            self.queues[index] -= departed
            self._service_credit[index] -= departed
            processed += departed
            if self.queues[index] == 0:
                self.oldest_wait[index] = 0
        self.total_processed += processed
        self.green_elapsed += 1
        return processed

    def _observation(self) -> NDArray[np.float32]:
        queue_scale = float(self.config.max_queue_length)
        delay_scale = float(self.config.max_queue_length * self.config.episode_length)
        duration_scale = float(
            self.config.maximum_green_time or self.config.episode_length
        )
        if self._arrival_history:
            recent = np.stack(tuple(self._arrival_history)).sum(axis=0)
            recent_rates = recent / len(self._arrival_history)
        else:
            recent_rates = np.zeros(4, dtype=np.float64)
        observation = np.concatenate(
            (
                np.clip(self.queues / queue_scale, 0.0, 1.0),
                np.clip(self.directional_delay / delay_scale, 0.0, 1.0),
                [float(self.current_phase)],
                [min(self.green_elapsed / duration_scale, 1.0)],
                [float(self.in_yellow)],
                np.clip(recent_rates, 0.0, 1.0),
            )
        )
        return observation.astype(np.float32)

    def _info(self, episode_end: bool) -> dict[str, Any]:
        info: dict[str, Any] = {
            "timestep": self.timestep,
            "total_queue": int(self.queues.sum()),
            "north_queue": int(self.queues[0]),
            "south_queue": int(self.queues[1]),
            "east_queue": int(self.queues[2]),
            "west_queue": int(self.queues[3]),
            "total_waiting_time": float(self.total_delay),
            "signal_switches": self.signal_switches,
            "current_phase": self.current_phase.name,
            "is_yellow": self.in_yellow,
            "total_vehicles_arrived": self.total_arrived,
            "total_vehicles_processed": self.total_processed,
            "overflow": self.total_overflow,
        }
        if episode_end:
            info.update(
                {
                    "average_waiting_time": self.total_delay / max(self.total_arrived, 1),
                    "max_queue_length": self.max_observed_queue,
                    "throughput": self.total_processed,
                    "completion_ratio": self.total_processed / max(self.total_arrived, 1),
                    "average_queue_length": self.queue_sum / max(self.timestep, 1),
                    "number_of_signal_switches": self.signal_switches,
                }
            )
        return info

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from traffic_rl.visualization.renderer import TrafficRenderer

            self._renderer = TrafficRenderer(mode=self.render_mode)
        return self._renderer.render(self)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
