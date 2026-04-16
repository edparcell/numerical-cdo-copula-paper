"""Monotonic piecewise cubic interpolation (Steffen 1990)."""

import numpy as np


class SteffenInterpolator:
    """Steffen's monotonic interpolation method.

    Given control points (x_i, y_i), constructs a piecewise cubic interpolant
    that passes through all points, has continuous first derivatives, and is
    monotonic between adjacent data points.

    Reference: M. Steffen, "A simple method for monotonic interpolation in
    one dimension", Astronomy and Astrophysics 239, 443-450 (1990).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 2:
            raise ValueError("Need at least 2 points")
        if not np.all(np.diff(x) > 0):
            raise ValueError("x must be strictly increasing")

        n = len(x)
        h = np.diff(x)
        s = np.diff(y) / h

        # Compute slopes at each point
        yp = np.zeros(n)

        if n == 2:
            yp[0] = s[0]
            yp[1] = s[0]
        else:
            # Interior points (eq 8, 11)
            for i in range(1, n - 1):
                p_i = (s[i - 1] * h[i] + s[i] * h[i - 1]) / (h[i - 1] + h[i])
                yp[i] = (np.sign(s[i - 1]) + np.sign(s[i])) * min(
                    abs(s[i - 1]), abs(s[i]), 0.5 * abs(p_i)
                )

            # Boundary points (eq 24-27)
            p1 = s[0] * (1 + h[0] / (h[0] + h[1])) - s[1] * h[0] / (h[0] + h[1])
            yp[0] = p1
            if p1 * s[0] <= 0:
                yp[0] = 0.0
            elif abs(p1) > 2 * abs(s[0]):
                yp[0] = 2 * s[0]

            pn = s[-1] * (1 + h[-1] / (h[-1] + h[-2])) - s[-2] * h[-1] / (h[-1] + h[-2])
            yp[-1] = pn
            if pn * s[-1] <= 0:
                yp[-1] = 0.0
            elif abs(pn) > 2 * abs(s[-1]):
                yp[-1] = 2 * s[-1]

        # Store cubic coefficients for each interval (eq 2-5)
        self._x = x
        self._y = y
        self._n = n
        self._a = (yp[:-1] + yp[1:] - 2 * s) / h**2
        self._b = (3 * s - 2 * yp[:-1] - yp[1:]) / h
        self._c = yp[:-1]
        self._d = y[:-1]
        self._yp = yp

    def __call__(self, x: np.ndarray | float) -> np.ndarray | float:
        scalar = np.isscalar(x)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.empty_like(x)

        # Find intervals
        idx = np.searchsorted(self._x, x) - 1
        idx = np.clip(idx, 0, self._n - 2)

        t = x - self._x[idx]
        result = self._a[idx] * t**3 + self._b[idx] * t**2 + self._c[idx] * t + self._d[idx]

        # Extrapolation: use endpoint slopes
        lo = x < self._x[0]
        hi = x > self._x[-1]
        result[lo] = self._y[0] + self._yp[0] * (x[lo] - self._x[0])
        result[hi] = self._y[-1] + self._yp[-1] * (x[hi] - self._x[-1])

        return float(result[0]) if scalar else result

    def derivative(self, x: np.ndarray | float) -> np.ndarray | float:
        scalar = np.isscalar(x)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.empty_like(x)

        idx = np.searchsorted(self._x, x) - 1
        idx = np.clip(idx, 0, self._n - 2)

        t = x - self._x[idx]
        result = 3 * self._a[idx] * t**2 + 2 * self._b[idx] * t + self._c[idx]

        lo = x < self._x[0]
        hi = x > self._x[-1]
        result[lo] = self._yp[0]
        result[hi] = self._yp[-1]

        return float(result[0]) if scalar else result

    @property
    def x(self) -> np.ndarray:
        return self._x.copy()

    @property
    def y(self) -> np.ndarray:
        return self._y.copy()
