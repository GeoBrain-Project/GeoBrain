"""
Post-process a list of simulation realisations.

Aggregates an ensemble of realisations (one :class:`GeoFrame` per
realisation, all sharing the same geometry and containing a chosen
column) into per-cell summary statistics: mean, variance, requested
percentiles, and optional probabilities of exceeding cutoffs.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import as_float_array

if TYPE_CHECKING:
    from ...frames import Geometry, GeoFrame

__all__ = ["PostSimulation"]


def _same_geometry(a: "Geometry", b: "Geometry") -> bool:
    """
    True if two geometries occupy the same locations.

    Geometry / GeoGrid / GeoPoints define no ``__eq__``, so an
    identity check is tried first (cheap, covers the common
    shared-geometry case) and falls back to a value-based comparison:
    same concrete type and same ``coords`` (matching shape, all close).
    Comparing ``coords`` catches a permuted/misaligned ensemble whose
    members would otherwise be stacked positionally onto the wrong cells.
    """
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    ca, cb = a.coords, b.coords
    if ca.shape != cb.shape:
        return False
    return bool(np.allclose(ca, cb))


class PostSimulation:
    """
    Ensemble statistics from a list of realisations.

    Args:
        percentiles: percentiles to compute (default ``[10, 50, 90]``).
        column: simulation column name in each realisation (default
            ``"simulation"``: matches SGSIM/LUSIM/DSSIM/SISIM/
            DirectSampling output).
        exceed_cutoffs: optional cutoffs ``cⱼ``; emits one column
            ``p_gt_cⱼ`` per cutoff with the across-realisation fraction
            of cells whose value exceeded ``cⱼ``.
    """

    def __init__(
        self,
        *,
        percentiles: Sequence[float] | None = None,
        column: str = "simulation",
        exceed_cutoffs: Sequence[float] | None = None,
    ) -> None:
        pct = list(percentiles) if percentiles is not None else [10.0, 50.0, 90.0]
        for p in pct:
            if not 0.0 <= float(p) <= 100.0:
                raise GeoBrainError(
                    "percentiles must lie in [0, 100]",
                    object_name="PostSimulation", field="percentiles",
                    expected="[0, 100]", actual=p,
                )
        self.percentiles = [float(p) for p in pct]
        self.column = column
        self.exceed_cutoffs = (
            None if exceed_cutoffs is None else [float(c) for c in exceed_cutoffs]
        )

    # ------------------------------------------------------------------

    def __call__(
        self,
        realisations: Sequence["GeoFrame"],
    ) -> "GeoFrame":
        from ...frames import ColumnRole, GeoFrame as _GeoFrame

        if len(realisations) == 0:
            raise GeoBrainError(
                "realisations list must be non-empty",
                object_name="PostSimulation", field="realisations",
                expected="length >= 1", actual=0,
            )
        ncells = len(realisations[0])
        first_geom = realisations[0].geometry
        nsim = len(realisations)
        stack = np.zeros((nsim, ncells), dtype=np.float64)
        for i, r in enumerate(realisations):
            if len(r) != ncells:
                raise GeoBrainError(
                    "all realisations must share the same length",
                    object_name="PostSimulation", field="realisations",
                    expected=f"length {ncells}", actual=len(r),
                )
            if not _same_geometry(r.geometry, first_geom):
                raise GeoBrainError(
                    "all realisations must share the same geometry "
                    "(same locations); misaligned ensembles would be "
                    "stacked positionally onto the wrong cells",
                    object_name="PostSimulation", field="realisations",
                    expected=f"geometry of realisation 0 ({first_geom!r})",
                    actual=f"realisation {i} ({r.geometry!r})",
                )
            if self.column not in r.columns:
                raise GeoBrainError(
                    f"realisation {i} is missing column {self.column!r}",
                    object_name="PostSimulation", field=self.column,
                    expected=f"one of {r.columns}", actual=None,
                )
            stack[i, :] = as_float_array(r[self.column])

        out = _GeoFrame(first_geom)
        out["mean"] = as_float_array(np.mean(stack, axis=0))
        out["variance"] = as_float_array(np.var(stack, axis=0, ddof=0))
        out.set_role("mean", ColumnRole.ESTIMATE)
        out.set_role("variance", ColumnRole.VARIANCE)

        for pct in self.percentiles:
            key = f"p{int(round(pct))}"
            out[key] = as_float_array(np.percentile(stack, pct, axis=0))

        if self.exceed_cutoffs is not None:
            for c in self.exceed_cutoffs:
                key = f"p_gt_{c:g}"
                out[key] = as_float_array(np.mean(stack > c, axis=0))

        return out

    def __repr__(self) -> str:
        return (
            f"PostSimulation(column={self.column!r}, "
            f"percentiles={self.percentiles})"
        )
