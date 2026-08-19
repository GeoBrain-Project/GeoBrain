"""
Cell-based declustering.

Scans a
range of cell sizes, overlays a regular grid on the data, assigns each
sample weight ``1 / count_in_its_cell`` (with averaging over several
origin offsets), and picks the cell size that minimises (or maximises)
the weighted mean.

The output is a single weight column (default ``"decluster_weight"``)
that can be plugged into :class:`NormalScore(weights=...)`,
:class:`VariogramCalculator(...)` or any downstream estimator that
accepts per-sample weights.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import FloatArray, as_float_array, as_int_array, zeros_float
from .base import Transform

if TYPE_CHECKING:
    from ...frames import GeoFrame

__all__ = ["Decluster"]


def _cell_weights(
    x: FloatArray, y: FloatArray, z: FloatArray,
    xsize: float, ysize: float, zsize: float,
    xo: float, yo: float, zo: float,
) -> FloatArray:
    """Normalised cell-declustering weights for one origin offset."""
    ncellx = max(int((float(np.max(x)) - (xo - xsize)) / max(xsize, 1e-10)) + 1, 1)
    ncelly = max(int((float(np.max(y)) - (yo - ysize)) / max(ysize, 1e-10)) + 1, 1)

    icx = np.floor((x - xo) / max(xsize, 1e-10)).astype(np.int64) + 1
    icy = np.floor((y - yo) / max(ysize, 1e-10)).astype(np.int64) + 1
    icz = np.floor((z - zo) / max(zsize, 1e-10)).astype(np.int64) + 1

    cell_id = icx + (icy - 1) * ncellx + (icz - 1) * ncellx * ncelly

    unique_result = cast(
        tuple[object, object, object],
        np.unique(cell_id, return_inverse=True, return_counts=True),
    )
    inverse = as_int_array(unique_result[1])
    counts = as_float_array(unique_result[2])
    weights = as_float_array(1.0 / counts[inverse])

    sumw = float(np.sum(weights))
    if sumw > 0:
        weights = as_float_array(weights / sumw)
    return weights


class Decluster(Transform):
    """
    Cell-based declustering for preferentially-sampled data.

    Args:
        column: column whose declustered mean drives cell-size selection.
        min_size / max_size: cell-size search bounds. Default: extent/50,
            extent/2.
        n_sizes: number of cell sizes tested.
        target: ``"min"`` (default, usual for positively skewed grade
            data) or ``"max"``.
        n_offsets: number of random origin offsets averaged per cell size.
        output_column: name of the weight column (default
            ``"decluster_weight"``).
    """

    def __init__(
        self,
        column: str,
        *,
        min_size: float | None = None,
        max_size: float | None = None,
        n_sizes: int = 20,
        target: str = "min",
        n_offsets: int = 5,
        output_column: str = "decluster_weight",
    ) -> None:
        if target not in ("min", "max"):
            raise GeoBrainError(
                "target must be 'min' or 'max'",
                object_name="Decluster", field="target",
                expected="'min' or 'max'", actual=target,
            )
        if n_sizes < 1:
            raise GeoBrainError(
                "n_sizes must be >= 1",
                object_name="Decluster", field="n_sizes",
                expected=">= 1", actual=n_sizes,
            )
        if n_offsets < 1:
            raise GeoBrainError(
                "n_offsets must be >= 1",
                object_name="Decluster", field="n_offsets",
                expected=">= 1", actual=n_offsets,
            )
        self.column = column
        self.min_size = min_size
        self.max_size = max_size
        self.n_sizes = int(n_sizes)
        self.target = target
        self.n_offsets = int(n_offsets)
        self.output_column = output_column

        self._optimal_weights: FloatArray | None = None
        self._optimal_cell_size: float | None = None
        self._declustered_mean: float | None = None
        self._original_mean: float | None = None
        self._fitted = False

    # ------------------------------------------------------------------

    def fit(self, data: "GeoFrame") -> "Decluster":
        if self.column not in data.columns:
            raise GeoBrainError(
                f"column {self.column!r} is missing",
                object_name="Decluster.fit", field="column",
                expected=f"one of {data.columns}", actual=self.column,
            )

        coords = as_float_array(data.geometry.coords)
        values = as_float_array(data[self.column])
        n = values.size
        ndim = data.geometry.ndim

        x = as_float_array(coords[:, 0])
        y = as_float_array(coords[:, 1])
        z = as_float_array(coords[:, 2] if ndim >= 3 else np.zeros(n, dtype=np.float64))

        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        zmin, zmax = float(np.min(z)), float(np.max(z))

        extent_x = max(xmax - xmin, 1e-10)
        extent_y = max(ymax - ymin, 1e-10)
        extent_z = max(zmax - zmin, 1e-10) if ndim >= 3 else 1.0

        cmin = self.min_size if self.min_size is not None else extent_x / 50.0
        cmax = self.max_size if self.max_size is not None else extent_x / 2.0
        if cmin <= 0 or cmax <= 0 or cmin > cmax:
            raise GeoBrainError(
                "cell-size bounds must satisfy 0 < min_size <= max_size",
                object_name="Decluster.fit", field="min/max_size",
                expected="0 < min <= max", actual=(cmin, cmax),
            )

        anisy = extent_y / extent_x if extent_x > 0 else 1.0
        anisz = extent_z / extent_x if (ndim >= 3 and extent_x > 0) else 0.0

        cell_sizes = (
            as_float_array([cmin])
            if self.n_sizes <= 1
            else as_float_array(np.linspace(cmin, cmax, self.n_sizes))
        )

        search_min = self.target == "min"
        self._original_mean = float(np.mean(values))

        xo1 = xmin - 0.01
        yo1 = ymin - 0.01
        zo1 = zmin - 0.01 if ndim >= 3 else 0.0

        best_weights = as_float_array(np.ones(n, dtype=np.float64) / n)
        best_mean = self._original_mean
        best_cs = float(cell_sizes[0])
        first = True
        noff = max(self.n_offsets, 1)

        for cs in cell_sizes:
            xcs = float(cs)
            ycs = xcs * anisy
            zcs = xcs * anisz if ndim >= 3 else 1.0

            wt_accum = zeros_float(n)
            xfac = min(xcs / float(noff), 0.5 * extent_x) if extent_x > 0 else 0.0
            yfac = min(ycs / float(noff), 0.5 * extent_y) if extent_y > 0 else 0.0
            zfac = (
                min(zcs / float(noff), 0.5 * extent_z)
                if (ndim >= 3 and extent_z > 0) else 0.0
            )

            for kp in range(noff):
                xo = xo1 - float(kp) * xfac
                yo = yo1 - float(kp) * yfac
                zo = zo1 - float(kp) * zfac
                wts = _cell_weights(x, y, z, xcs, ycs, zcs, xo, yo, zo)
                wt_accum = as_float_array(wt_accum + wts)

            sumw = float(np.sum(wt_accum))
            if sumw <= 0:
                continue
            vrcr = float(np.sum(wt_accum * values) / sumw)

            update = first or (vrcr < best_mean if search_min else vrcr > best_mean)
            if update:
                best_mean = vrcr
                best_weights = as_float_array(wt_accum.copy())
                best_cs = xcs
                first = False

        # GSLIB convention: normalise weights so they sum to n_samples.
        sumw = float(np.sum(best_weights))
        if sumw > 0:
            best_weights = as_float_array(best_weights * (float(n) / sumw))

        self._optimal_weights = best_weights
        self._optimal_cell_size = best_cs
        self._declustered_mean = best_mean
        self._fitted = True
        return self

    def transform(self, data: "GeoFrame") -> "GeoFrame":
        if not self._fitted or self._optimal_weights is None:
            raise GeoBrainError(
                "Decluster.transform called before fit()",
                object_name="Decluster.transform", field="_fitted",
                expected="True", actual=self._fitted,
            )
        out = data.copy()
        out[self.output_column] = self._optimal_weights.copy()
        return out

    # ------------------------------------------------------------------

    @property
    def optimal_cell_size(self) -> float | None:
        return self._optimal_cell_size

    @property
    def declustered_mean(self) -> float | None:
        return self._declustered_mean

    @property
    def original_mean(self) -> float | None:
        return self._original_mean

    @property
    def weights(self) -> FloatArray | None:
        """Fitted weight vector (sums to ``n_samples`` after fit)."""
        return self._optimal_weights

    def __repr__(self) -> str:
        return f"Decluster({self.column!r})"
