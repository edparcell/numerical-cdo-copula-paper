"""Hazard rate default model."""

import numpy as np


class FlatHazardRate:
    """Flat hazard rate bootstrapped from a CDS spread.

    lambda = spread / (1 - recovery)
    p(t) = 1 - exp(-lambda * t)
    """

    def __init__(self, spread: float, recovery: float):
        self._spread = spread
        self._recovery = recovery
        self._lambda = spread / (1.0 - recovery)

    def default_prob(self, t: float | np.ndarray) -> float | np.ndarray:
        return 1.0 - np.exp(-self._lambda * t)

    def survival_prob(self, t: float | np.ndarray) -> float | np.ndarray:
        return np.exp(-self._lambda * t)

    @property
    def hazard_rate(self) -> float:
        return self._lambda
