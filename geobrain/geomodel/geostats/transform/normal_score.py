"""
Normal-score transform and back-transform.

Builds a quantile-matching table between the (optionally
declustered) input distribution and the standard normal, using
GSLIB midpoint cumulative probabilities. Back-transform supports
configurable lower / upper tail extrapolation (``ltail`` / ``utail``
∈ ``{1: linear, 2: power, 4: hyperbolic (upper only)}``).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import FloatArray, as_bool_array, as_float_array
from ._gslib_helpers import backtr_value, gauinv
from .base import Transform

if TYPE_CHECKING:
    from ...frames import GeoFrame


class NormalScore(Transform):
    """
    Quantile-matching transform to a standard normal distribution.

    Args:
        column: input column to transform.
        output_column: name for the transformed column (default
            ``f"{column}_ns"``).
        weights: optional declustering-weight column.
        tmin, tmax: trim limits; values outside this range are
            excluded from the fit table.
        ltail, ltpar: lower-tail extrapolation mode (1=linear,
            2=power-``ltpar``) and parameter.
        utail, utpar: upper-tail mode (1=linear, 2=power-``utpar``,
            4=hyperbolic) and parameter.
        zmin, zmax: hard *physical* lower / upper bounds for the
            back-transform tails. Left ``None`` (the default) they mean
            "unbounded", the configured ``ltail`` / ``utail`` extrapolate
            genuinely beyond the data range (canonical GSLIB ``backtr``
            semantics). Supply finite values only when a real physical
            limit (e.g. porosity ≥ 0) should clamp the result.
    """

    def __init__(
        self,
        column: str,
        *,
        output_column: str | None = None,
        weights: str | None = None,
        tmin: float = -1e21,
        tmax: float = 1e21,
        ltail: int = 1,
        ltpar: float = 1.0,
        utail: int = 1,
        utpar: float = 1.5,
        zmin: float | None = None,
        zmax: float | None = None,
    ) -> None:
        self.column = column
        self.output_column = output_column or f"{column}_ns"
        self.weights = weights
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.ltail = int(ltail)
        self.ltpar = float(ltpar)
        self.utail = int(utail)
        self.utpar = float(utpar)
        self.zmin: float | None = zmin if zmin is None else float(zmin)
        self.zmax: float | None = zmax if zmax is None else float(zmax)
        self._vr_table: FloatArray | None = None
        self._vrg_table: FloatArray | None = None
        self._fitted = False

    # ------------------------------------------------------------------

    def fit(self, data: "GeoFrame") -> "NormalScore":
        if self.column not in data.columns:
            raise GeoBrainError(
                f"column {self.column!r} is missing",
                object_name="NormalScore.fit", field="column",
                expected=f"one of {data.columns}", actual=self.column,
            )

        values = as_float_array(data[self.column])
        if self.weights is not None:
            if self.weights not in data.columns:
                raise GeoBrainError(
                    f"weights column {self.weights!r} is missing",
                    object_name="NormalScore.fit", field="weights",
                    expected=f"one of {data.columns}", actual=self.weights,
                )
            wt = as_float_array(data[self.weights])
        else:
            wt = None

        mask = as_bool_array((values >= self.tmin) & (values <= self.tmax))
        valid = as_float_array(values[mask].copy())
        if valid.size == 0:
            raise GeoBrainError(
                "no values passed the trim limits",
                object_name="NormalScore.fit", field="values",
                expected=">= 1 valid value",
                actual=f"0 of {values.size}",
            )

        if wt is not None:
            valid_wt = as_float_array(wt[mask].copy())
        else:
            valid_wt = as_float_array(np.ones(valid.size, dtype=np.float64))

        order = np.argsort(valid, kind="mergesort")
        vr = as_float_array(valid[order])
        sorted_wt = as_float_array(valid_wt[order])

        twt = float(np.sum(sorted_wt))
        if twt <= 0.0:
            raise GeoBrainError(
                "weights sum to zero",
                object_name="NormalScore.fit", field="weights",
                expected="positive sum", actual=twt,
            )

        n = vr.size
        vrg = np.zeros(n, dtype=np.float64)
        oldcp = 0.0
        cp = 0.0
        for i in range(n):
            cp += float(sorted_wt[i]) / twt
            midpoint = (cp + oldcp) / 2.0
            oldcp = cp
            vrg[i], _ = gauinv(midpoint)

        self._vr_table = vr
        self._vrg_table = as_float_array(vrg)
        # Do NOT default zmin/zmax to the data extent. Leaving them None
        # means "unbounded" so the configured lower/upper tails genuinely
        # extrapolate beyond the data range (canonical GSLIB ``backtr``
        # semantics). Defaulting to vr[0]/vr[-1] would collapse the tail
        # pivots and clamp every extreme back-transform to the data min/max.
        self._fitted = True
        return self

    # ------------------------------------------------------------------

    def transform(self, data: "GeoFrame") -> "GeoFrame":
        if not self._fitted or self._vr_table is None or self._vrg_table is None:
            raise GeoBrainError(
                "NormalScore.transform called before fit()",
                object_name="NormalScore.transform", field="_fitted",
                expected="True", actual=self._fitted,
            )
        vr = self._vr_table
        vrg = self._vrg_table

        values = as_float_array(data[self.column])
        n = values.size
        ns = np.zeros(n, dtype=np.float64)

        for i in range(n):
            v = float(values[i])
            # Bracket the run of table entries equal to ``v`` with a
            # left/right searchsorted pair. When ``v`` exactly matches one
            # or more tied entries (lo < hi) the correct normal score is the
            # mid-rank Gaussian value, i.e. the MEAN of the matching ``vrg``
            # entries, not the lowest tie's score.
            lo = int(np.searchsorted(vr, v, side="left"))
            hi = int(np.searchsorted(vr, v, side="right"))
            if hi > lo:
                ns[i] = float(np.mean(vrg[lo:hi]))
            elif lo == 0:
                ns[i] = float(vrg[0])
            elif lo >= vr.size:
                ns[i] = float(vrg[-1])
            else:
                v0 = float(vr[lo - 1])
                v1 = float(vr[lo])
                g0 = float(vrg[lo - 1])
                g1 = float(vrg[lo])
                if abs(v1 - v0) < 1e-20:
                    ns[i] = (g0 + g1) / 2.0
                else:
                    ns[i] = g0 + (g1 - g0) * (v - v0) / (v1 - v0)

        out = data.copy()
        out[self.output_column] = ns
        return out

    # ------------------------------------------------------------------

    def inverse_transform(self, data: "GeoFrame") -> "GeoFrame":
        if not self._fitted or self._vr_table is None or self._vrg_table is None:
            raise GeoBrainError(
                "NormalScore.inverse_transform called before fit()",
                object_name="NormalScore.inverse_transform",
                field="_fitted", expected="True", actual=self._fitted,
            )
        if self.output_column not in data.columns:
            raise GeoBrainError(
                f"normal-score column {self.output_column!r} is missing",
                object_name="NormalScore.inverse_transform", field="output_column",
                expected=f"one of {data.columns}", actual=self.output_column,
            )

        # zmin/zmax may legitimately be None, that means "unbounded /
        # extrapolate" rather than clamp to the data extent.
        ns_values = as_float_array(data[self.output_column])
        n = ns_values.size
        original = np.zeros(n, dtype=np.float64)
        for i in range(n):
            original[i] = backtr_value(
                float(ns_values[i]),
                self._vrg_table, self._vr_table,
                self.zmin, self.zmax,
                self.ltail, self.ltpar, self.utail, self.utpar,
            )

        out = data.copy()
        out[self.column] = original
        return out

    # ------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def vr_table(self) -> FloatArray | None:
        return self._vr_table

    @property
    def vrg_table(self) -> FloatArray | None:
        return self._vrg_table

    def __repr__(self) -> str:
        return f"NormalScore({self.column!r})"


__all__ = ["NormalScore"]
