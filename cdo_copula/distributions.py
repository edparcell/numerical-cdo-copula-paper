"""Distribution classes for the copula model."""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from .mathutils import norm_pdf, norm_cdf, norm_ppf, gauss_hermite_expect
from .steffen import SteffenInterpolator


class Distribution(ABC):
    @abstractmethod
    def pdf(self, x: np.ndarray | float) -> np.ndarray | float: ...

    @abstractmethod
    def cdf(self, x: np.ndarray | float) -> np.ndarray | float: ...

    @abstractmethod
    def ppf(self, p: np.ndarray | float) -> np.ndarray | float: ...

    @abstractmethod
    def negentropy(self) -> float: ...


class Normal(Distribution):
    def pdf(self, x):
        return norm_pdf(x)

    def cdf(self, x):
        return norm_cdf(x)

    def ppf(self, p):
        return norm_ppf(p)

    def negentropy(self):
        return 0.0


class ANTDistribution(Distribution):
    """Arbitrary Normal Transform distribution.

    Defined by F_X(x) = Phi(h(x)) where h is a strictly monotone increasing
    function. The function h can be anything callable — Steffen interpolant,
    Chebyshev polynomial, etc.

    The distribution is automatically standardised to mean 0, variance 1.
    """

    def __init__(
        self,
        h: Callable,
        h_deriv: Callable,
        h_inv: Callable,
    ):
        self._h_raw = h
        self._h_deriv_raw = h_deriv
        self._h_inv_raw = h_inv

        # Compute mean and std for standardisation by direct integration
        # (Gauss-Hermite via h_inv is unreliable for non-normal distributions)
        x_grid = np.linspace(-20, 20, 4000)
        fx = np.asarray(norm_pdf(np.asarray(h(x_grid), dtype=float)), dtype=float) * \
             np.maximum(np.asarray(h_deriv(x_grid), dtype=float), 0.0)
        mass = np.trapezoid(fx, x_grid)
        if mass > 1e-10:
            fx = fx / mass  # normalise in case of truncation
        self._mu = np.trapezoid(x_grid * fx, x_grid)
        var = np.trapezoid((x_grid - self._mu) ** 2 * fx, x_grid)
        self._sigma = np.sqrt(max(var, 1e-24))

    @staticmethod
    def from_knots(
        x_points: np.ndarray,
        y_points: np.ndarray,
        extrap_slope: float = 1.0,
    ) -> "ANTDistribution":
        """Construct from Steffen-interpolated control points.

        Inside [x_0, x_n]: Steffen interpolation (monotonic, C1).
        Outside: linear with the given slope, anchored at the endpoints.
        """
        x_points = np.asarray(x_points, dtype=float)
        y_points = np.asarray(y_points, dtype=float)

        h_steffen = SteffenInterpolator(x_points, y_points)
        x0, xn = x_points[0], x_points[-1]
        y0, yn = y_points[0], y_points[-1]
        s = extrap_slope

        def h(x):
            x = np.asarray(x, dtype=float)
            scalar = x.ndim == 0
            x = np.atleast_1d(x)
            result = np.asarray(h_steffen(x), dtype=float)
            lo = x < x0
            hi = x > xn
            result[lo] = y0 + s * (x[lo] - x0)
            result[hi] = yn + s * (x[hi] - xn)
            return float(result[0]) if scalar else result

        def h_deriv(x):
            x = np.asarray(x, dtype=float)
            scalar = x.ndim == 0
            x = np.atleast_1d(x)
            result = np.asarray(h_steffen.derivative(x), dtype=float)
            lo = x < x0
            hi = x > xn
            result[lo] = s
            result[hi] = s
            return float(result[0]) if scalar else result

        def h_inv(u):
            from scipy.optimize import brentq
            u = np.asarray(u, dtype=float)
            scalar = u.ndim == 0
            u = np.atleast_1d(u)
            result = np.empty_like(u)
            for i, ui in enumerate(u):
                if ui <= y0:
                    result[i] = x0 + (ui - y0) / s
                elif ui >= yn:
                    result[i] = xn + (ui - yn) / s
                else:
                    result[i] = brentq(lambda x: h_steffen(x) - ui, x0, xn, xtol=1e-12)
            return float(result[0]) if scalar else result

        dist = ANTDistribution(h=h, h_deriv=h_deriv, h_inv=h_inv)
        dist._knot_x = x_points.copy()
        dist._knot_y = y_points.copy()
        return dist


    KNOT_RANGE = (-6.0, 6.0)

    @staticmethod
    def _softmax_to_y(raw: np.ndarray) -> np.ndarray:
        """Map n unconstrained values to n+1 ordered y-values spanning KNOT_RANGE.

        softmax(raw) -> cumsum -> prepend 0 -> scale to [-6, 6].
        """
        lo, hi = ANTDistribution.KNOT_RANGE
        raw_shifted = raw - np.max(raw)
        g = np.exp(raw_shifted)
        p = g / np.sum(g)
        c = np.concatenate([[0.0], np.cumsum(p)])
        return lo + (hi - lo) * c

    @staticmethod
    def from_unconstrained(params: np.ndarray, n_points: int, bw_scale: float = 1.0) -> "ANTDistribution":
        """Construct from unconstrained optimiser parameters.

        params layout: [focus, tightness, strength, w_1, ..., w_n]
        Total length: n_points + 3

        focus: centre of knot concentration (in KNOT_RANGE)
        tightness: std of the concentration (smaller = tighter)
        strength: blend between uniform (0) and focused (1)
        w_i: bump-basis softmax weights for y-values
        """
        from .bump_basis import make_bump_matrix_from_positions, apply_bump_basis
        from .focused_grid import focused_grid

        lo, hi = ANTDistribution.KNOT_RANGE
        n = n_points

        focus = params[0]
        tightness = params[1]
        strength = params[2]
        y_weights = params[3:]

        # Build focused x-grid (n interior + 2 endpoints = n+1 with endpoints)
        x_interior = focused_grid(n - 1, lo, hi, focus, tightness, strength)
        x = np.concatenate([[lo], x_interior, [hi]])

        # Build bump matrix from x-positions (not indices).
        # Each softmax weight corresponds to a gap between adjacent
        # y-values. Use the midpoint of each x-interval as the position.
        x_midpoints = 0.5 * (x[:-1] + x[1:])  # n midpoints for n gaps
        bump = make_bump_matrix_from_positions(x_midpoints, bandwidth_scale=bw_scale)

        softmax_inputs = apply_bump_basis(y_weights, bump)
        y = ANTDistribution._softmax_to_y(softmax_inputs)
        return ANTDistribution.from_knots(x, y)

    @staticmethod
    def identity_params(n_points: int) -> np.ndarray:
        """Initial parameters for h(x) = x (the Gaussian copula case).

        focus=0 (centre), tightness=3, strength=0 (uniform), equal y-weights.
        """
        params = np.zeros(n_points + 3)
        params[0] = 0.0   # focus at centre
        params[1] = 3.0   # wide tightness
        params[2] = 0.0   # no concentration
        return params

    def _to_raw(self, x):
        return self._mu + self._sigma * x

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        raw = self._to_raw(x)
        hx = np.asarray(self._h_raw(raw), dtype=float)
        hp = np.asarray(self._h_deriv_raw(raw), dtype=float) * self._sigma
        return norm_pdf(hx) * np.maximum(hp, 0.0)

    def cdf(self, x):
        return norm_cdf(np.asarray(self._h_raw(self._to_raw(x)), dtype=float))

    def ppf(self, p):
        return (np.asarray(self._h_inv_raw(norm_ppf(p)), dtype=float) - self._mu) / self._sigma

    def negentropy(self):
        """Negentropy of the standardised distribution.

        H_N - H(X) where H_N = 0.5 * ln(2*pi*e) is the entropy of the
        standard normal. Computed by direct integration of -f ln f.
        """
        h_n = 0.5 * np.log(2 * np.pi * np.e)
        x = np.linspace(-20, 20, 4000)
        fx = np.asarray(self.pdf(x), dtype=float)
        integrand = np.where(fx > 1e-300, -fx * np.log(fx), 0.0)
        h_x = np.trapezoid(integrand, x)
        return h_n - h_x

    @property
    def mu(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma
