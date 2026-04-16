"""Focused x-grid: mixture of uniform + truncated normal for knot placement."""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import truncnorm


def focused_grid(n: int, left: float, right: float,
                 focus: float, tightness: float, strength: float) -> np.ndarray:
    """Generate n interior knot positions concentrated around focus.

    Uses the inverse CDF of a mixture distribution:
      F(x) = strength * F_truncnorm(x) + (1 - strength) * F_uniform(x)

    where F_truncnorm is a truncated normal with mean=focus and
    std=tightness on [left, right], and F_uniform is uniform on
    [left, right].

    At strength=0: equally spaced (uniform).
    At strength=1: concentrated around focus.

    Returns n points strictly inside (left, right).
    """
    if tightness < 1e-6:
        tightness = 1e-6

    # Truncated normal parameters
    a_tn = (left - focus) / tightness
    b_tn = (right - focus) / tightness
    tn = truncnorm(a_tn, b_tn, loc=focus, scale=tightness)

    # Mixture CDF
    def mixture_cdf(x):
        uniform_part = (x - left) / (right - left)
        normal_part = tn.cdf(x)
        return (1 - strength) * uniform_part + strength * normal_part

    # Evaluate at n equally-spaced quantile levels in (0, 1)
    quantiles = np.linspace(0, 1, n + 2)[1:-1]  # n interior points

    # Inverse CDF via root-finding
    points = np.empty(n)
    for i, q in enumerate(quantiles):
        points[i] = brentq(lambda x: mixture_cdf(x) - q, left, right, xtol=1e-10)

    return points
