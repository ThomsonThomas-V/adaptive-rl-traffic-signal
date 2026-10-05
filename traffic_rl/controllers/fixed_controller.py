"""Fixed-cycle traffic-light controller."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FixedTimeController:
    """Request alternating green phases after a fixed green interval."""

    green_duration: int = 30
    _phase: int = 0
    _elapsed: int = 0

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._phase = 0
        self._elapsed = 0

    def predict(self, observation: NDArray[np.float32], deterministic: bool = True) -> int:
        del deterministic
        if observation[10] > 0.5:  # yellow is controlled by the environment
            return 0
        self._elapsed += 1
        if self._elapsed >= self.green_duration:
            self._phase = 1 - self._phase
            self._elapsed = 0
            return 1 if self._phase == 0 else 2
        return 0
