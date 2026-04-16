"""Bump basis: maps raw parameters to softmax inputs via a banded kernel.

Instead of each raw parameter controlling a single softmax input,
each parameter creates a smooth bump that moves several nearby inputs.
This gives gradient-based optimizers better sensitivity.
"""

import numpy as np


def make_bump_matrix_from_positions(x: np.ndarray, bandwidth_scale: float = 1.0) -> np.ndarray:
    """Create an n x n bump basis matrix from x-positions.

    Entry (i, j) = exp(-0.5 * (x_i - x_j)^2 / bandwidth^2).
    Symmetric, positive definite. The kernel width is in x-space,
    so the coupling between knots depends on their actual distance,
    not their index distance.

    Base bandwidth is 0.5 * average spacing, multiplied by bandwidth_scale.
    At scale=1, couples ~2-3 neighbours. Higher scale = broader coupling.
    """
    avg_spacing = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 1.0
    bandwidth = 0.5 * avg_spacing * bandwidth_scale
    bandwidth = max(bandwidth, 1e-6)  # avoid division by zero
    diff = x[:, None] - x[None, :]
    return np.exp(-0.5 * diff**2 / bandwidth**2)


def apply_bump_basis(raw: np.ndarray, bump_matrix: np.ndarray) -> np.ndarray:
    """Map raw parameters through the bump basis to get softmax inputs."""
    return bump_matrix @ raw
