"""
Post-process indicator-kriging probabilities into E-type estimates.

Reads the ``prob_0``, ``prob_1``, … columns produced by
:class:`IndicatorKriging` / :class:`SoftIndicatorKriging`, builds a
full conditional CDF at each location by sandwiching the discrete
``F(tₖ)`` between ``F(zmin) = 0`` and ``F(zmax) = 1``, applies the
order-relation correction, and reports the **E-type** estimate
``E[Z] = zmin + ∫(1 − F(z)) dz`` plus its **variance**
``Var[Z] = E[Z²] − E[Z]²``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import FloatArray, as_float_array

if TYPE_CHECKING:
    from ...frames import GeoFrame

__all__ = ["PostIndicatorKriging"]


def _order_relation_correction(p: FloatArray) -> FloatArray:
    out = as_float_array(p.copy())
    for k in range(1, out.size):
        if out[k] < out[k - 1]:
            out[k] = out[k - 1]
    return as_float_array(np.clip(out, 0.0, 1.0))


def _integrate_etype(z: FloatArray, f: FloatArray) -> float:
    """``E[Z] = zmin + ∫(1 − F(z)) dz`` via trapezoidal rule."""
    total = 0.0
    for k in range(z.size - 1):
        dz = float(z[k + 1] - z[k])
        avg_f = float(0.5 * (f[k] + f[k + 1]))
        total += dz * (1.0 - avg_f)
    return float(z[0] + total)


def _integrate_ez2(z: FloatArray, f: FloatArray) -> float:
    """``E[Z²] = z₀² + 2 ∫ z·(1 − F(z)) dz`` via trapezoidal rule."""
    total = 0.0
    for k in range(z.size - 1):
        dz = float(z[k + 1] - z[k])
        z_mid = float(0.5 * (z[k] + z[k + 1]))
        avg_f = float(0.5 * (f[k] + f[k + 1]))
        total += dz * 2.0 * z_mid * (1.0 - avg_f)
    return float(z[0] * z[0] + total)


class PostIndicatorKriging:
    """
    Post-process IK probability columns into ``estimate`` + ``variance``.

    Args:
        thresholds: cutoff values matching the ``prob_k`` columns.
        zmin / zmax: lower / upper extrapolation bounds. Default:
            ``thresholds[0] − (thresholds[1] − thresholds[0])`` /
            ``thresholds[-1] + (thresholds[-1] − thresholds[-2])``.
        prob_prefix: column-name prefix ``f"{prefix}_{k}"`` (default
            ``"prob"``: matches :class:`IndicatorKriging`).
    """

    def __init__(
        self,
        thresholds: list[float],
        *,
        zmin: float | None = None,
        zmax: float | None = None,
        prob_prefix: str = "prob",
    ) -> None:
        if not thresholds:
            raise GeoBrainError(
                "thresholds must be non-empty",
                object_name="PostIndicatorKriging", field="thresholds",
                expected="length >= 1", actual=0,
            )
        sorted_thr = sorted(float(t) for t in thresholds)
        if sorted_thr != [float(t) for t in thresholds]:
            raise GeoBrainError(
                "thresholds must be sorted ascending",
                object_name="PostIndicatorKriging", field="thresholds",
                expected="ascending", actual=thresholds,
            )
        self.thresholds = sorted_thr
        ncut = len(self.thresholds)
        if zmin is not None:
            self.zmin = float(zmin)
        elif ncut >= 2:
            self.zmin = self.thresholds[0] - (self.thresholds[1] - self.thresholds[0])
        else:
            self.zmin = self.thresholds[0] - 1.0
        if zmax is not None:
            self.zmax = float(zmax)
        elif ncut >= 2:
            self.zmax = self.thresholds[-1] + (self.thresholds[-1] - self.thresholds[-2])
        else:
            self.zmax = self.thresholds[0] + 1.0
        if not (self.zmin < self.thresholds[0]):
            raise GeoBrainError(
                "zmin must be strictly less than the smallest threshold",
                object_name="PostIndicatorKriging", field="zmin",
                expected=f"< {self.thresholds[0]}", actual=self.zmin,
            )
        if not (self.zmax > self.thresholds[-1]):
            raise GeoBrainError(
                "zmax must be strictly greater than the largest threshold",
                object_name="PostIndicatorKriging", field="zmax",
                expected=f"> {self.thresholds[-1]}", actual=self.zmax,
            )
        self.prob_prefix = prob_prefix

    # ------------------------------------------------------------------

    def __call__(self, data: "GeoFrame") -> "GeoFrame":
        from ...frames import GeoFrame

        if not isinstance(data, GeoFrame):
            raise GeoBrainError(
                "data must be a GeoFrame",
                object_name="PostIndicatorKriging", field="data",
                expected="GeoFrame", actual=type(data).__name__,
            )
        ncut = len(self.thresholds)
        n = len(data)
        probs = np.zeros((ncut, n), dtype=np.float64)
        for k in range(ncut):
            col = f"{self.prob_prefix}_{k}"
            if col not in data.columns:
                raise GeoBrainError(
                    f"required column {col!r} is missing",
                    object_name="PostIndicatorKriging.__call__",
                    field="data.columns",
                    expected=f"one of {data.columns}", actual=col,
                )
            probs[k] = as_float_array(data[col])

        estimates = np.full(n, np.nan, dtype=np.float64)
        variances = np.full(n, np.nan, dtype=np.float64)

        z_thr = np.array(self.thresholds, dtype=np.float64)
        z_nodes = np.concatenate([[self.zmin], z_thr, [self.zmax]])

        for i in range(n):
            p = probs[:, i]
            if np.isnan(p).all():
                continue
            p = np.where(np.isnan(p), 0.5, p)
            p = _order_relation_correction(as_float_array(p))
            f_nodes = as_float_array(np.concatenate([[0.0], p, [1.0]]))
            etype = _integrate_etype(z_nodes, f_nodes)
            ez2 = _integrate_ez2(z_nodes, f_nodes)
            estimates[i] = etype
            variances[i] = max(ez2 - etype * etype, 0.0)

        from ...frames import ColumnRole, GeoFrame as _GeoFrame
        out = _GeoFrame(data.geometry)
        # Preserve original probability columns. Note: the emitted prob_k
        # columns are passed through RAW (un-order-relation-corrected); only
        # ``estimate`` / ``variance`` use the order-relation-corrected CDF.
        for k in range(ncut):
            out[f"{self.prob_prefix}_{k}"] = probs[k]
        out["estimate"] = as_float_array(estimates)
        out["variance"] = as_float_array(variances)
        out.set_role("estimate", ColumnRole.ESTIMATE)
        out.set_role("variance", ColumnRole.VARIANCE)
        return out

    def __repr__(self) -> str:
        return (
            f"PostIndicatorKriging(thresholds={self.thresholds}, "
            f"zmin={self.zmin}, zmax={self.zmax})"
        )
