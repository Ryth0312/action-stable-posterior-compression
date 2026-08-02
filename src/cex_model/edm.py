"""Equilibrium Dispersive Model (EDM) spatial discretization matrices."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import inv

from cex_model.column import ColumnParameters


@dataclass
class EDMMatrices:
    """Finite-element discretization operators for EDM.

    All arrays are a deterministic function of the column geometry and are shared
    (by reference) from a process-wide cache — treat them as READ-ONLY. ``T`` is
    the precomputed transport operator ``N @ A_C`` (constant per geometry), reused
    by both the RHS and the analytic Jacobian.
    """

    M: np.ndarray  # mass matrix
    N: np.ndarray  # inverse mass matrix
    A_C: np.ndarray  # convection + dispersion operator
    h: float  # element size, mm
    T: np.ndarray = None  # type: ignore[assignment]  # N @ A_C (transport operator)


@lru_cache(maxsize=64)
def _edm_matrices_cached(
    grid_size: int, length: float, velocity: float, epsbed: float, dax: float
) -> EDMMatrices:
    """Build (and cache) M, N, A_C, T from primitive geometry keys.

    Cached on the exact set of column attributes that affect the result, so the
    dense ``inv(M)`` and ``N @ A_C`` are computed once per geometry and reused
    across the many simulations in fitting / optimization (lossless).
    """
    gs = grid_size
    h = length / (gs - 1)

    M = (
        h / 6 * np.diag(np.ones(gs - 1), 1)
        + 2 * h / 3 * np.diag(np.ones(gs))
        + h / 6 * np.diag(np.ones(gs - 1), -1)
    )
    M[0, 0] = h / 3
    M[-1, -1] = h / 3

    N = inv(M)

    C = (
        0.5 * np.diag(np.ones(gs - 1), 1)
        - 0.5 * np.diag(np.ones(gs - 1), -1)
    )
    C[0, 0] = -0.5
    C[-1, -1] = 0.5

    A = (
        -1 / h * np.diag(np.ones(gs - 1), 1)
        + 2 / h * np.diag(np.ones(gs))
        - 1 / h * np.diag(np.ones(gs - 1), -1)
    )
    A[0, 0] = 1 / h
    A[-1, -1] = 1 / h

    A_C = -velocity / epsbed * C - dax * A
    T = N @ A_C

    return EDMMatrices(M=M, N=N, A_C=A_C, h=h, T=T)


def generate_edm_matrices(column: ColumnParameters) -> EDMMatrices:
    """Build M, N, A_C, T matrices (matches MATLAB ``generate_matrix.m``).

    Spatial discretization: linear finite elements on ``grid_size`` nodes.
    Delegates to a cache keyed by the geometry attributes that determine the
    result.
    """
    return _edm_matrices_cached(
        column.grid_size, column.length, column.velocity, column.epsbed, column.dax
    )
