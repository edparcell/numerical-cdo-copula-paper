"""One-factor copula models."""

from abc import ABC, abstractmethod

import numpy as np

from .distributions import Distribution, Normal, ANTDistribution
from .mathutils import norm_cdf, norm_ppf, gauss_hermite_points
from .steffen import SteffenInterpolator


class CopulaModel(ABC):
    @abstractmethod
    def quadrature_points(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (market_factor_values, weights) for integration.

        Weights should sum to 1 (i.e., include the 1/sqrt(pi) factor).
        """
        ...

    @abstractmethod
    def conditional_default_prob(self, p_unconditional: float, m: float) -> float:
        """P(default | M = m) given unconditional default probability p."""
        ...

    @property
    @abstractmethod
    def rho(self) -> float: ...


class GaussianCopula(CopulaModel):
    def __init__(self, rho: float):
        self._rho = rho
        self._sqrt_rho = np.sqrt(rho)
        self._sqrt_1mrho = np.sqrt(1.0 - rho)

    @property
    def rho(self) -> float:
        return self._rho

    def quadrature_points(self, n: int = 40) -> tuple[np.ndarray, np.ndarray]:
        nodes, weights = gauss_hermite_points(n)
        m_values = np.sqrt(2) * nodes
        w = weights / np.sqrt(np.pi)
        return m_values, w

    def conditional_default_prob(self, p_unconditional: float, m: float) -> float:
        return norm_cdf(
            (norm_ppf(p_unconditional) - self._sqrt_rho * m) / self._sqrt_1mrho
        )


# Gauss-Hermite nodes for the convolution integral (fixed)
_CONV_N = 100


class ANTCopula(CopulaModel):
    def __init__(self, rho: float, dist_m: Distribution, dist_eps: Distribution):
        self._rho = rho
        self._sqrt_rho = np.sqrt(rho)
        self._sqrt_1mrho = np.sqrt(1.0 - rho)
        self._dist_m = dist_m
        self._dist_eps = dist_eps

        # Precompute F_A and F_A^{-1} on a grid
        self._build_fa_grid()

    def _build_fa_grid(self):
        """Compute F_A on a grid via Gauss-Hermite, Steffen-interpolate."""
        sqrt_rho = self._sqrt_rho
        sqrt_1mrho = self._sqrt_1mrho

        # Gauss-Hermite nodes for integration over M (change of variable)
        nodes, weights = gauss_hermite_points(_CONV_N)
        u_nodes = np.sqrt(2) * nodes
        m_nodes = self._dist_m.ppf(norm_cdf(u_nodes))
        w_nodes = weights / np.sqrt(np.pi)

        # Evaluate F_A on a grid
        x_grid = np.linspace(-15, 15, 2001)
        fa_grid = np.empty_like(x_grid)

        for i, x in enumerate(x_grid):
            z_vals = (x - sqrt_rho * m_nodes) / sqrt_1mrho
            fe_vals = self._dist_eps.cdf(z_vals)
            fa_grid[i] = np.sum(w_nodes * fe_vals)

        fa_grid = np.clip(fa_grid, 0.0, 1.0)

        self._fa_interp = SteffenInterpolator(x_grid, fa_grid)

        # For the inverse, find the region where F_A is strictly increasing
        mask = (fa_grid > 1e-12) & (fa_grid < 1 - 1e-12)
        fa_masked = fa_grid[mask]
        x_masked = x_grid[mask]
        # Remove any remaining duplicates
        unique_mask = np.concatenate([[True], np.diff(fa_masked) > 1e-15])
        fa_masked = fa_masked[unique_mask]
        x_masked = x_masked[unique_mask]
        if len(fa_masked) < 2:
            # Fallback: linear between extremes
            fa_masked = np.array([1e-12, 1 - 1e-12])
            x_masked = np.array([-7.0, 7.0])
        self._fa_inv_interp = SteffenInterpolator(fa_masked, x_masked)

    def _fa_inv(self, p: float) -> float:
        return self._fa_inv_interp(p)

    @property
    def rho(self) -> float:
        return self._rho

    def quadrature_points(self, n: int = 40) -> tuple[np.ndarray, np.ndarray]:
        nodes, weights = gauss_hermite_points(n)
        u = np.sqrt(2) * nodes
        m_values = self._dist_m.ppf(norm_cdf(u))
        w = weights / np.sqrt(np.pi)
        return m_values, w

    def conditional_default_prob(self, p_unconditional: float, m: float) -> float:
        c = self._fa_inv(p_unconditional)
        z = (c - self._sqrt_rho * m) / self._sqrt_1mrho
        return self._dist_eps.cdf(z)
