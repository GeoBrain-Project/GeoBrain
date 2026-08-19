"""
Arbitrary-order finite-difference coefficients.

Unlike GeoBrain's current kernel (which hard-codes the 4th-order staggered taps
``c1 = 9/8, c2 = -1/24``), these are solved from the defining linear system for
**any** even order, matching a configurable ``fd_order``.

Staggered first derivative
--------------------------
Approximate ``f'(x)`` from samples at the half-points ``x ± (k - 1/2)h``::

    f'(x) ≈ (1/h) Σ_{k=1}^{N} c_k [ f(x + (k-1/2)h) − f(x − (k-1/2)h) ]

Taylor-matching the odd derivatives gives, for m = 1..N::

    Σ_{k=1}^{N} c_k (k - 1/2)^(2m-1) = (1/2) · δ_{m,1}

Solving this ``N×N`` system yields the staggered coefficients. For ``N = 2``
(4th order) it reproduces ``[9/8, -1/24]``; for ``N = 4`` (8th order)
``[1225/1024, -245/3072, 49/5120, -5/7168]``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations
from geobrain.core import GeoBrainError

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray


@lru_cache(maxsize=None)
def staggered_first_derivative_coeffs(order: int) -> tuple[float, ...]:
    """Return the staggered first-derivative taps ``(c_1, ..., c_N)``.

    Args:
        order: spatial accuracy order (even, ``>= 2``); ``N = order // 2``.

    Returns:
        A tuple of ``N`` coefficients applied to the antisymmetric pairs
        ``f(x+(k-1/2)h) - f(x-(k-1/2)h)``.
    """
    if order < 2 or order % 2 != 0:
        raise GeoBrainError(f"order must be a positive even integer, got {order}")
    n = order // 2
    # Vandermonde-like system A c = b with A[m, k] = (k - 1/2)^(2m-1).
    k: NDArray[np.float64] = np.arange(1, n + 1, dtype=np.float64)
    powers = 2 * np.arange(1, n + 1, dtype=np.float64) - 1.0  # 1, 3, 5, ...
    a = (k[None, :] - 0.5) ** powers[:, None]
    b: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
    b[0] = 0.5
    c = np.linalg.solve(a, b)
    return tuple(float(v) for v in c)


@lru_cache(maxsize=None)
def centered_first_derivative_coeffs(order: int) -> tuple[float, ...]:
    """Return centered (non-staggered) first-derivative taps ``(c_1, ..., c_N)``.

    Approximates ``f'(x) ≈ (1/h) Σ_{k=1}^{N} c_k [f(x+kh) − f(x−kh)]`` with
    ``Σ_k c_k k^(2m-1) = (1/2) δ_{m,1}``. Provided for completeness / centered
    schemes; the staggered taps are what the velocity-stress kernel uses.
    """
    if order < 2 or order % 2 != 0:
        raise GeoBrainError(f"order must be a positive even integer, got {order}")
    n = order // 2
    k: NDArray[np.float64] = np.arange(1, n + 1, dtype=np.float64)
    powers = 2 * np.arange(1, n + 1, dtype=np.float64) - 1.0
    a = k[None, :] ** powers[:, None]
    b: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
    b[0] = 0.5
    c = np.linalg.solve(a, b)
    return tuple(float(v) for v in c)
