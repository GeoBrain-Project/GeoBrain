"""
Robust experimental-variogram estimators.

Each
function takes the pairwise increments within a lag bin and returns one
semivariance value (= the per-bin γ contribution). ``matheron``,
``cressie_hawkins`` and ``dowd`` consume the **absolute** increments
``|Δ|``; ``genton`` consumes the **signed** increments ``Vᵢ − Vⱼ``.

Estimators:

- **Cressie-Hawkins**: robust mean of ``|Δ|^{1/2}``, fourth power,
  with finite-sample correction.
- **Dowd**: based on the median of ``|Δ|``.
- **Genton**: based on the Qₙ scale of the *signed* increments.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import cast

import numpy as np

from ...frames._arrays import FloatArray, IntArray, as_float_array

__all__ = ["cressie_hawkins", "dowd", "genton", "matheron"]


def matheron(diffs: FloatArray) -> float:
    """
    Classical Matheron estimator: ``(1/(2N)) · Σ Δᵢ²``.

    ``diffs`` is the vector of absolute pairwise differences in the lag bin.
    """
    n = int(diffs.size)
    if n == 0:
        return 0.0
    return float(np.sum(diffs ** 2) / (2.0 * n))


def cressie_hawkins(diffs: FloatArray, count: int) -> float:
    """
    Cressie-Hawkins robust estimator.

    ``γ̂(h) = [⟨|Δ|^{1/2}⟩]⁴ / (0.457 + 0.494/N + 0.045/N²) / 2``.

    Args:
        diffs: pair differences in one lag bin.
        count: pair count of the bin.
    """
    n = int(count)
    if n == 0:
        return 0.0
    mean_sqrt = float(np.mean(np.abs(diffs) ** 0.5))
    correction = 0.457 + 0.494 / n + 0.045 / (n * n)
    return float(mean_sqrt**4 / correction / 2.0)


def dowd(diffs: FloatArray) -> float:
    """
    Dowd (1984) robust estimator: ``γ̂(h) = ½ · 2.198 · (median|Δ|)²``.

    Canonical Dowd defines ``2γ̂(h) = 2.198 · [median|Δ|]²``; the ``2.198``
    multiplies the (already squared) median absolute increment, so it is
    **not** itself squared. ``diffs`` is the vector of absolute pairwise
    increments in the lag bin.
    """
    if diffs.size == 0:
        return 0.0
    med = float(np.median(np.abs(diffs)))
    return float(0.5 * 2.198 * med ** 2)


def genton(diffs: FloatArray) -> float:
    """
    Genton (1998) robust estimator based on the Qₙ scale of the increments.

    ``diffs`` is the vector of **signed** pairwise increments ``Vᵢ − Vⱼ``
    within the lag bin (not their absolute value). The Qₙ scale of those
    increments is the ``k = C(⌊N/2⌋+1, 2)`` order statistic of the pairwise
    absolute differences ``|Yᵢ − Yⱼ|`` of the increments ``Y``. Because the
    increments are zero-centred for a stationary process, ``Qₙ`` consistently
    estimates ``√(2γ)``, so ``γ̂(h) = ½ · (2.219 · Qₙ-order-stat)²``.

    Note the sign matters: folding the increments to ``|Δ|`` before forming
    the order statistic shrinks their spread and biases ``γ`` low.
    """
    incr = as_float_array(np.asarray(diffs, dtype=np.float64))
    n = int(incr.size)
    if n == 0:
        return 0.0
    if n == 1:
        return float(0.5 * incr[0] ** 2)

    pairwise = as_float_array(np.abs(incr[:, None] - incr[None, :]))
    iu = cast(tuple[IntArray, IntArray], np.triu_indices(n, k=1))
    pw = as_float_array(pairwise[iu])
    k = int(np.floor(n / 2) + 1)
    k = min(k * (k - 1) // 2, len(pw))
    k = max(k, 1)
    pw_sorted = as_float_array(np.sort(pw))
    qn = float(pw_sorted[k - 1]) if len(pw_sorted) > 0 else 0.0
    return float(0.5 * (2.219 * qn) ** 2)
