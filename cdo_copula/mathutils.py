"""Pure numpy replacements for scipy.stats.norm and scipy.special functions.

Avoids scipy import overhead in multiprocessing workers.
"""

import numpy as np
from math import sqrt, pi, log

_SQRT2 = sqrt(2.0)
_SQRT2PI = sqrt(2.0 * pi)
_LOG_SQRT2PI = log(_SQRT2PI)

_GH_CACHE = {}


def norm_pdf(x):
    """Standard normal PDF."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / _SQRT2PI


def norm_cdf(x):
    """Standard normal CDF using the error function."""
    from math import erf as _erf
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    result = np.array([0.5 * (1.0 + _erf(float(xi) / _SQRT2)) for xi in x])
    return float(result[0]) if scalar else result


def norm_ppf(p):
    """Standard normal inverse CDF (percent-point function).

    Uses rational approximation (Beasley-Springer-Moro algorithm).
    Accurate to ~1e-9 over (0, 1).
    """
    p = np.asarray(p, dtype=float)
    scalar = p.ndim == 0
    p = np.atleast_1d(np.clip(p, 1e-15, 1 - 1e-15))
    result = np.empty_like(p)

    # Coefficients for rational approximation
    a = np.array([
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00,
    ])
    b = np.array([
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01,
    ])
    c = np.array([
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00,
    ])
    d = np.array([
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00,
    ])

    p_low = 0.02425
    p_high = 1 - p_low

    # Central region
    mask_central = (p_low <= p) & (p <= p_high)
    q = p[mask_central] - 0.5
    r = q * q
    num = ((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]
    den = ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0
    result[mask_central] = q * num / den

    # Lower tail
    mask_low = p < p_low
    q = np.sqrt(-2.0 * np.log(p[mask_low]))
    num = ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
    den = (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0
    result[mask_low] = num / den

    # Upper tail
    mask_high = p > p_high
    q = np.sqrt(-2.0 * np.log(1.0 - p[mask_high]))
    num = ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
    den = (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0
    result[mask_high] = -num / den

    return float(result[0]) if scalar else result


def gauss_hermite_points(n: int = 40):
    """Return (nodes, weights) for Gauss-Hermite quadrature.

    Computed via eigenvalue method (Golub-Welsch algorithm), cached.
    """
    if n in _GH_CACHE:
        return _GH_CACHE[n]
    i = np.arange(1, n)
    beta = np.sqrt(i / 2.0)
    T = np.diag(beta, -1) + np.diag(beta, 1)
    nodes, V = np.linalg.eigh(T)
    weights = V[0, :] ** 2 * np.sqrt(np.pi)
    idx = np.argsort(nodes)
    result = (nodes[idx], weights[idx])
    _GH_CACHE[n] = result
    return result


def gauss_hermite_expect(func, n: int = 40):
    """Compute E[func(Z)] where Z ~ N(0,1) using Gauss-Hermite quadrature."""
    nodes, weights = gauss_hermite_points(n)
    u = _SQRT2 * nodes
    return np.sum(weights * func(u)) / np.sqrt(np.pi)
