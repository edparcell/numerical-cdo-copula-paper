"""ANT copula using chebpy for convolution and inverse CDF."""

import numpy as np

from .copula import CopulaModel
from .distributions import ANTDistribution
from .mathutils import norm_cdf, norm_ppf, gauss_hermite_points


class ANTCopulaCheb(CopulaModel):
    """ANT copula with chebpy-based F_A computation.

    Uses chebpy for:
    - Representing scaled densities as Chebyshev polynomials
    - Convolution via Hale-Townsend algorithm
    - cumsum for F_A
    - roots() for F_A^{-1}
    """

    def __init__(self, rho: float, dist_m: ANTDistribution, dist_eps: ANTDistribution):
        self._rho = rho
        self._sqrt_rho = np.sqrt(rho)
        self._sqrt_1mrho = np.sqrt(1.0 - rho)
        self._dist_m = dist_m
        self._dist_eps = dist_eps

        self._build_fa_cheb()

    def _build_fa_cheb(self):
        """Build F_A and F_A^{-1} using chebpy convolution."""
        from chebpy import chebfun
        import warnings

        sqrt_rho = self._sqrt_rho
        sqrt_1mrho = self._sqrt_1mrho

        # Domains: ±6 standard deviations of each scaled variable
        domain_m = 6.0 * sqrt_rho
        domain_eps = 6.0 * sqrt_1mrho

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Scaled densities as chebfuns
            self._f_scaled_m = chebfun(
                lambda x: self._dist_m.pdf(x / sqrt_rho) / sqrt_rho,
                [-domain_m, domain_m],
            )
            self._f_scaled_eps = chebfun(
                lambda x: self._dist_eps.pdf(x / sqrt_1mrho) / sqrt_1mrho,
                [-domain_eps, domain_eps],
            )

            # Convolve
            self._f_a = self._f_scaled_m.conv(self._f_scaled_eps)

            # CDF via cumulative sum
            self._F_a = self._f_a.cumsum()

            # Normalise
            fa_max = float(self._F_a(self._F_a.domain[-1]))
            if fa_max > 0 and abs(fa_max - 1.0) > 1e-10:
                self._F_a = self._F_a * (1.0 / fa_max)

    def _fa_inv(self, p: float) -> float:
        """F_A^{-1}(p) via chebpy root-finding."""
        p = float(np.clip(p, 1e-12, 1 - 1e-12))
        roots = (self._F_a - p).roots()
        if len(roots) == 0:
            # Fallback
            return self._F_a.domain[0] if p < 0.5 else self._F_a.domain[-1]
        return float(roots[0])

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
