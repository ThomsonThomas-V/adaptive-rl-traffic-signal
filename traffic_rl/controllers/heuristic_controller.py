"""Queue-responsive traffic-light controller."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class QueueHeuristicController:
    """Serve the opposing axis when its queue exceeds a hysteresis threshold."""

    switching_threshold: float = 0.20
    minimum_green_time: int = 5
    _phase: int = 0
    _elapsed: int = 0

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._phase = 0
        self._elapsed = 0

    def predict(self, observation: NDArray[np.float32], deterministic: bool = True) -> int:
        del deterministic
        if observation[10] > 0.5:
            return 0
        self._phase = int(round(float(observation[8])))
        ns_queue = float(observation[0] + observation[1])
        ew_queue = float(observation[2] + observation[3])
        self._elapsed += 1
        if self._elapsed < self.minimum_green_time:
            return 0
        current = ns_queue if self._phase == 0 else ew_queue
        opposing = ew_queue if self._phase == 0 else ns_queue
        required = current * (1.0 + self.switching_threshold)
        if opposing > required and opposing > 0:
            self._phase = 1 - self._phase
            self._elapsed = 0
            return 1 if self._phase == 0 else 2
        return 0
