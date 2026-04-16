"""Interest rate curve."""

import numpy as np


class FlatForwardCurve:
    """Flat continuously-compounded forward rate."""

    def __init__(self, rate: float):
        self._rate = rate

    def df(self, t: float | np.ndarray) -> float | np.ndarray:
        return np.exp(-self._rate * t)

    @property
    def rate(self) -> float:
        return self._rate
