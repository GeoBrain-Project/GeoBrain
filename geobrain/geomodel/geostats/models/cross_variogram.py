"""
Experimental cross-variogram computation.

Computes the binned cross-semivariance

::

    γᵢⱼ(h) = (1 / 2N(h)) · Σ_pairs (vᵢ(p) − vᵢ(q))·(vⱼ(p) − vⱼ(q))

between two columns of a :class:`GeoFrame`. Pair binning uses the same
lag-tolerance logic as the univariate
:class:`VariogramCalculator` (defaults to half the lag spacing). The
returned object mirrors :class:`ExperimentalVariogram` (``lags``,
``gammas``, ``pairs``).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Callable, cast

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import FloatArray, IntArray, as_bool_array, as_float_array, as_int_array

if TYPE_CHECKING:
    from ...frames import GeoFrame

__all__ = ["CrossVariogramResult", "CrossVariogramCalculator"]


@dataclass(frozen=True)
class CrossVariogramResult:
    """Binned cross-variogram ``γᵢⱼ(h)``.

    Attributes:
        lags: lag-bin centres [m].
        gammas: cross-variogram values per bin.
        pairs: pair counts per bin.
    """

    lags: FloatArray
    gammas: FloatArray
    pairs: IntArray

    def __repr__(self) -> str:
        return f"CrossVariogramResult({self.lags.size} lags)"


class CrossVariogramCalculator:
    """
    Binned cross-variogram calculator.

    Args:
        lag_distance: required lag spacing (no auto-mode).
        n_lags: number of lag bins.
        tolerance: lag tolerance (default ``lag_distance / 2``).
    """

    def __init__(
        self,
        lag_distance: float,
        n_lags: int = 15,
        tolerance: float | None = None,
    ) -> None:
        if lag_distance <= 0:
            raise GeoBrainError(
                "lag_distance must be positive",
                object_name="CrossVariogramCalculator", field="lag_distance",
                expected="> 0", actual=lag_distance,
            )
        if n_lags < 1:
            raise GeoBrainError(
                "n_lags must be >= 1",
                object_name="CrossVariogramCalculator", field="n_lags",
                expected=">= 1", actual=n_lags,
            )
        if tolerance is not None and tolerance <= 0:
            raise GeoBrainError(
                "tolerance must be positive",
                object_name="CrossVariogramCalculator", field="tolerance",
                expected="> 0", actual=tolerance,
            )
        self.lag_distance = float(lag_distance)
        self.n_lags = int(n_lags)
        self.tolerance = tolerance

    def compute(
        self, data: "GeoFrame", column1: str, column2: str
    ) -> CrossVariogramResult:
        for col in (column1, column2):
            if col not in data.columns:
                raise GeoBrainError(
                    f"data column {col!r} is missing",
                    object_name="CrossVariogramCalculator.compute",
                    field="column", expected=f"one of {data.columns}",
                    actual=col,
                )
        coords = as_float_array(data.geometry.coords)
        v1 = as_float_array(data[column1])
        v2 = as_float_array(data[column2])
        n = int(v1.size)
        lag = self.lag_distance
        tol = float(self.tolerance) if self.tolerance is not None else lag * 0.5

        lags = as_float_array(np.zeros(self.n_lags, dtype=np.float64))
        gammas = as_float_array(np.zeros(self.n_lags, dtype=np.float64))
        pairs = as_int_array(np.zeros(self.n_lags, dtype=np.int64))

        if n < 2:
            lags[:] = (np.arange(self.n_lags) + 1) * lag
            return CrossVariogramResult(lags, gammas, pairs)

        pdist = cast(
            Callable[..., object],
            getattr(import_module("scipy.spatial.distance"), "pdist"),
        )
        dist_vec = as_float_array(pdist(coords, metric="euclidean"))
        # Pairwise cross-differences: 0.5 · (v1_i − v1_j) · (v2_i − v2_j).
        # Use a compact loop because pdist on stacked product is awkward.
        diff_product = np.zeros(dist_vec.size, dtype=np.float64)
        k = 0
        for i in range(n):
            d1 = v1[i] - v1[i + 1 :]
            d2 = v2[i] - v2[i + 1 :]
            block = 0.5 * d1 * d2
            diff_product[k : k + block.size] = block
            k += block.size

        # GSLIB overlapping-window binning: a pair contributes to bin k when its
        # distance is within ``tol`` of the bin centre (lag·k). ``tol = lag/2``
        # gives contiguous, non-overlapping bins; ``tol > lag/2`` widens them.
        for lag_idx in range(self.n_lags):
            nominal = (lag_idx + 1) * lag
            mask = as_bool_array(np.abs(dist_vec - nominal) <= tol)
            count = int(np.sum(mask))
            if count == 0:
                continue
            lags[lag_idx] = float(np.sum(dist_vec[mask]) / count)
            gammas[lag_idx] = float(np.sum(diff_product[mask]) / count)
            pairs[lag_idx] = count

        empty = pairs <= 0
        lags[empty] = (np.arange(self.n_lags)[empty] + 1) * lag
        return CrossVariogramResult(lags, gammas, pairs)

    def __repr__(self) -> str:
        return (
            f"CrossVariogramCalculator(lag={self.lag_distance}, "
            f"n_lags={self.n_lags})"
        )
