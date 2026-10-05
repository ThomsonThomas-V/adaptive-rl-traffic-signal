"""Seedable random baseline."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class RandomController:
    """Sample uniformly from keep, NS, and EW actions."""

    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(self.seed if seed is None else seed)

    def predict(self, observation: NDArray[np.float32], deterministic: bool = True) -> int:
        del observation, deterministic
        return int(self._rng.integers(0, 3))
