"""MAPSCleaning: Moving-Average Post-Simulation filter for categorical realisations.

Iteratively replaces each cell of a categorical realisation with the
weighted majority vote of its rectangular neighbourhood. Per-category
"dynamic factors" boost under-represented categories toward target
proportions on each pass.

Use this to smooth out short-range artefacts (e.g. patchy noise) from
:class:`SISIM` / :class:`DirectSampling` / :class:`PlurigaussianSim`
output while approximately preserving the target proportion vector.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Union

import numpy as np

from ....core import GeoBrainError
from ...frames import gslib_grid_layout
from ...frames._arrays import FloatArray, as_float_array, ones_float, zeros_float

if TYPE_CHECKING:
    from ...frames import GeoFrame

__all__ = ["MAPSCleaning"]

_Category = Union[int, float]
_WindowHalf = Union[int, Sequence[int]]


class MAPSCleaning:
    """
    Categorical realisation cleaner.

    Args:
        column: column name carrying the categorical values.
        categories: distinct category values that appear in the
            realisation.
        target_proportions: target proportion per category (must sum
            to 1). ``None`` → uniform.
        n_iterations: number of cleaning passes (default 3).
        window_half: half-size of the voting window. Scalar → applied
            to ``(x, y)`` with ``z=0`` for 2-D. 3-tuple → explicit
            ``(whx, why, whz)``.
        min_cluster_size: reserved for cluster-based post-filter
            (currently unused; kept for API symmetry).
    """

    def __init__(
        self,
        column: str,
        categories: Sequence[_Category],
        *,
        target_proportions: Sequence[float] | None = None,
        n_iterations: int = 3,
        window_half: _WindowHalf = 1,
        min_cluster_size: int = 4,
    ) -> None:
        if not categories:
            raise GeoBrainError(
                "categories must be non-empty",
                object_name="MAPSCleaning",
                field="categories",
                expected="length >= 1",
                actual=0,
            )
        if n_iterations < 1:
            raise GeoBrainError(
                "n_iterations must be >= 1",
                object_name="MAPSCleaning",
                field="n_iterations",
                expected=">= 1",
                actual=n_iterations,
            )
        self.column = column
        self.categories = list(categories)
        self.n_categories = len(self.categories)
        self._cat_to_idx: dict[_Category, int] = {c: i for i, c in enumerate(self.categories)}
        if target_proportions is not None:
            t = as_float_array(target_proportions)
            if t.size != self.n_categories:
                raise GeoBrainError(
                    "target_proportions length must match categories",
                    object_name="MAPSCleaning",
                    field="target_proportions",
                    expected=f"length {self.n_categories}",
                    actual=t.size,
                )
            total = float(np.sum(t))
            self.target = as_float_array(t / total if total > 0 else t)
        else:
            self.target = as_float_array(np.ones(self.n_categories) / self.n_categories)
        self.n_iterations = int(n_iterations)
        self.min_cluster_size = int(min_cluster_size)

        if isinstance(window_half, int):
            self.whalf: tuple[int, int, int] = (window_half, window_half, 0)
        else:
            wh = tuple(int(v) for v in window_half)
            if len(wh) != 3:
                raise GeoBrainError(
                    "window_half must be an int or a 3-element sequence",
                    object_name="MAPSCleaning",
                    field="window_half",
                    expected="int or (whx, why, whz)",
                    actual=wh,
                )
            self.whalf = wh  # type: ignore[assignment]

    # ------------------------------------------------------------------

    def __call__(self, data: "GeoFrame") -> "GeoFrame":
        from ...frames import GeoFrame, GeoGrid

        if not isinstance(data, GeoFrame):
            raise GeoBrainError(
                "data must be a GeoFrame",
                object_name="MAPSCleaning",
                field="data",
                expected="GeoFrame",
                actual=type(data).__name__,
            )
        if not isinstance(data.geometry, GeoGrid):
            raise GeoBrainError(
                "MAPSCleaning requires a GeoGrid geometry",
                object_name="MAPSCleaning",
                field="data.geometry",
                expected="GeoGrid",
                actual=type(data.geometry).__name__,
            )
        if self.column not in data.columns:
            raise GeoBrainError(
                f"column {self.column!r} is missing",
                object_name="MAPSCleaning",
                field="column",
                expected=f"one of {data.columns}",
                actual=self.column,
            )

        grid = data.geometry
        nx, ny, nz = gslib_grid_layout(grid).shape
        # This algorithm explicitly consumes the external three-axis layout;
        # the live 2-D grid and its public arrays remain two-dimensional.
        field = data[self.column].reshape((nx, ny, nz), order="F").copy()
        whx, why, whz = self.whalf

        for _ in range(self.n_iterations):
            current = self._compute_proportions(field)
            facact = ones_float(self.n_categories)
            for k in range(self.n_categories):
                if current[k] > 1.0e-10:
                    facact[k] = self.target[k] / current[k]
                else:
                    facact[k] = 1.0e6
            new_field = as_float_array(field.copy())
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        votes = zeros_float(self.n_categories)
                        for dk in range(-whz, whz + 1):
                            jz = iz + dk
                            if not (0 <= jz < nz):
                                continue
                            for dj in range(-why, why + 1):
                                jy = iy + dj
                                if not (0 <= jy < ny):
                                    continue
                                for di in range(-whx, whx + 1):
                                    jx = ix + di
                                    if not (0 <= jx < nx):
                                        continue
                                    if di == 0 and dj == 0 and dk == 0:
                                        continue
                                    cat = float(field[jx, jy, jz])
                                    ci = self._cat_to_idx.get(cat)
                                    if ci is None:
                                        # Try the int-coerced form (data
                                        # may carry float-encoded integers).
                                        ci = self._cat_to_idx.get(int(cat))
                                    if ci is not None:
                                        votes[ci] += 1.0
                        adjusted = as_float_array(votes * facact)
                        best = int(np.argmax(adjusted))
                        if adjusted[best] > 0:
                            new_field[ix, iy, iz] = float(self.categories[best])
            field = new_field

        out = data.copy()
        # GeoFrame owns and flattens the explicit adapter array back into its
        # declared-rank x-fastest public property contract.
        out[self.column] = field
        return out

    # ------------------------------------------------------------------

    def _compute_proportions(self, field: FloatArray) -> FloatArray:
        total = float(field.size)
        out = zeros_float(self.n_categories)
        for k, cat in enumerate(self.categories):
            out[k] = float(np.sum(field == cat)) / total
        return out

    def __repr__(self) -> str:
        return (
            f"MAPSCleaning({self.column!r}, "
            f"categories={self.categories}, "
            f"n_iterations={self.n_iterations}, "
            f"window_half={self.whalf})"
        )
