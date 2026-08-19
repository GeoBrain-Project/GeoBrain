"""
GSLIB-flavoured numeric primitives used by the back-transform path.

Faithful ports of:

- ``gauinv`` (GSLIB inverse standard-normal CDF, polynomial form)
- ``gcum``   (GSLIB standard-normal CDF, polynomial form)
- ``powint`` (power-law interpolation between two pivots)
- ``locate`` (1-based bracket index in a monotone table)
- ``backtr_value`` (single-value back-transform with tail handling,
  ``ltail/utail`` ∈ {1: linear, 2: power, 4: hyperbolic (upper only)})

Kept private to ``transform/`` because the rest of the package has no need
for GSLIB primitives, they're an implementation detail of the
normal-score machinery.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math

import numpy as np

from ...frames._arrays import FloatArray, as_float_array

__all__ = ["gauinv", "gcum", "powint", "locate", "backtr_value"]

_EPSLON = 1.0e-20
_LIM = 1.0e-10
_P = (
    -0.322232431088, -1.0, -0.342242088547,
    -0.0204231210245, -0.0000453642210148,
)
_Q = (
    0.0993484626060, 0.588581570495, 0.531103462366,
    0.103537752850, 0.0038560700634,
)


def gauinv(p: float) -> tuple[float, int]:
    """
    GSLIB inverse standard-normal CDF (polynomial approximation).

    Returns ``(z, ierr)`` where ``ierr=1`` means ``p`` was clipped to
    ``±1e10`` (out of range).
    """
    p = float(p)
    if p < _LIM:
        return -1.0e10, 1
    if p > (1.0 - _LIM):
        return 1.0e10, 1
    if p == 0.5:
        return 0.0, 0

    pp = 1.0 - p if p > 0.5 else p
    y = math.sqrt(math.log(1.0 / (pp * pp)))
    p0, p1, p2, p3, p4 = _P
    q0, q1, q2, q3, q4 = _Q
    xp = y + (
        ((((p4 * y + p3) * y + p2) * y + p1) * y) + p0
    ) / (
        ((((q4 * y + q3) * y + q2) * y + q1) * y) + q0
    )
    if p == pp:
        xp = -xp
    return float(xp), 0


def gcum(x: float) -> float:
    """GSLIB standard-normal CDF (polynomial approximation)."""
    x = float(x)
    z = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * z)
    val = t * (
        0.31938153
        + t * (-0.356563782
        + t * (1.781477937
        + t * (-1.821255978
        + t * 1.330274429)))
    )
    e2 = math.exp(-z * z / 2.0) * 0.3989422803 if z <= 6.0 else 0.0
    val = 1.0 - e2 * val
    return val if x >= 0.0 else 1.0 - val


def powint(
    xlow: float, xhigh: float,
    ylow: float, yhigh: float,
    xval: float, power: float,
) -> float:
    """
    Power-law interpolation between two pivots.

    ``y = ylow + (yhigh − ylow) · ((xval − xlow) / (xhigh − xlow))^power``,
    with the midpoint as a fall-back when the bracket is degenerate.
    """
    if abs(xhigh - xlow) < _EPSLON:
        return (ylow + yhigh) / 2.0
    return ylow + (yhigh - ylow) * (
        ((xval - xlow) / (xhigh - xlow)) ** power
    )


def locate(arr: FloatArray, val: float) -> int:
    """
    Return ``j`` such that ``arr[j] <= val < arr[j+1]``.

    Uses ``np.searchsorted`` for performance, equivalent to the GSLIB
    ``locate`` bisection. Returns ``-1`` when ``val < arr[0]`` and
    ``len(arr) - 1`` when ``val >= arr[-1]``.
    """
    idx = int(np.searchsorted(arr, val, side="right")) - 1
    return idx


def backtr_value(
    ynorm: float,
    vrg_table: FloatArray,
    vr_table: FloatArray,
    zmin: float | None,
    zmax: float | None,
    ltail: int = 1,
    ltpar: float = 1.0,
    utail: int = 1,
    utpar: float = 1.5,
) -> float:
    """
    Back-transform a single normal score to original units.

    ``vrg_table`` and ``vr_table`` are the sorted normal scores and
    sorted original values built by :class:`NormalScore.fit`.

    ``zmin`` / ``zmax`` are *physical* hard bounds. A ``None`` bound
    means "unbounded"; the corresponding tail is extrapolated freely
    (GSLIB ``backtr`` semantics) rather than clamped to the data range.
    For the linear / power lower tail the pivot at ``cdf = 0`` is
    ``zmin``; when ``zmin`` is ``None`` the power tail is anchored at the
    data minimum ``vr[0]`` instead. The symmetric convention holds for
    ``zmax`` and the upper power tail. Clipping at the end is applied
    *only* to finite (user-supplied) bounds.
    """
    vrg = as_float_array(vrg_table)
    vr = as_float_array(vr_table)
    n = vrg.size

    # Pivot for the open end of each power/linear tail. When the user did
    # not supply a hard physical bound (None) the pivot collapses onto the
    # data extreme, so the power tail still extrapolates monotonically.
    lo_pivot = float(vr[0]) if zmin is None else float(zmin)
    hi_pivot = float(vr[-1]) if zmax is None else float(zmax)

    if ynorm <= vrg[0]:
        cdflo = gcum(vrg[0])
        cdfbt = gcum(ynorm)
        if ltail == 1:
            result = powint(0.0, cdflo, lo_pivot, vr[0], cdfbt, 1.0)
        elif ltail == 2:
            cpow = 1.0 / ltpar
            result = powint(0.0, cdflo, lo_pivot, vr[0], cdfbt, cpow)
        else:
            result = float(vr[0])
    elif ynorm >= vrg[-1]:
        cdfhi = gcum(vrg[-1])
        cdfbt = gcum(ynorm)
        if utail == 1:
            result = powint(cdfhi, 1.0, vr[-1], hi_pivot, cdfbt, 1.0)
        elif utail == 2:
            cpow = 1.0 / utpar
            result = powint(cdfhi, 1.0, vr[-1], hi_pivot, cdfbt, cpow)
        elif utail == 4:
            lam = (float(vr[-1]) ** utpar) * (1.0 - gcum(vrg[-1]))
            denom = 1.0 - gcum(ynorm)
            if denom < _EPSLON:
                # gcum saturated to 1: the hyperbolic tail diverges; fall
                # back to the finite upper bound if one was supplied, else
                # the data maximum.
                result = hi_pivot
            else:
                result = (lam / denom) ** (1.0 / utpar)
        else:
            result = float(vr[-1])
    else:
        j = locate(vrg, float(ynorm))
        j = max(min(n - 2, j), 0)
        result = powint(
            float(vrg[j]), float(vrg[j + 1]),
            float(vr[j]), float(vr[j + 1]),
            float(ynorm), 1.0,
        )

    # Clip ONLY to finite, user-supplied physical bounds. A None bound
    # leaves that side unbounded so genuine extrapolation survives.
    if zmin is not None and result < zmin:
        result = float(zmin)
    if zmax is not None and result > zmax:
        result = float(zmax)
    return float(result)
