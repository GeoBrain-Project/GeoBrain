"""
Geometry abstract base class for spatial data containers.

Numpy-based; ``coords`` is an ``np.ndarray`` (geomodel/ is the numpy island, see
the project's feedback memory ``feedback-geomodel-numpy-island``).

Every concrete geometry exposes ``coords``, ``ndim``, ``npoints``,
``bounds``, ``centroid``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import cast

import numpy as np

from ..errors import GeomodelContractError
from ._arrays import FloatArray


def _canonical_finite_mean(
    values: FloatArray,
    *,
    axis: int,
    object_name: str,
    field: str,
) -> FloatArray | np.float64:
    """Return a permutation-invariant scaled mean of finite float64 values."""
    array = np.asarray(values, dtype=np.float64)
    if not -array.ndim <= axis < array.ndim:
        raise GeomodelContractError(
            "mean axis is invalid for the supplied values",
            object_name=object_name,
            field=field,
            expected=f"axis of an array with rank {array.ndim}",
            actual=axis,
        )
    moved = np.moveaxis(array, axis, 0)
    count = int(moved.shape[0])
    if count == 0:
        raise GeomodelContractError(
            "mean is undefined for empty values",
            object_name=object_name,
            field=field,
            expected="at least one finite value",
            actual=0,
        )

    flat = moved.reshape(count, -1)
    result = np.empty(flat.shape[1], dtype=np.float64)
    for column_index in range(flat.shape[1]):
        # Numeric sorting gives every permutation one fixed accumulation order.
        # Canonicalize signed zero because equality sorting otherwise preserves
        # its input order even though -0.0 and +0.0 are the same observation.
        column = [0.0 if float(value) == 0.0 else float(value) for value in flat[:, column_index]]
        column.sort()
        scale = max(abs(value) for value in column)
        if scale == 0.0:
            candidate = 0.0
        else:
            weighted_values: list[float] = []
            start = 0
            while start < count:
                stop = start + 1
                while stop < count and column[stop] == column[start]:
                    stop += 1
                frequency = (stop - start) / count
                weighted_values.append((column[start] / scale) * frequency)
                start = stop
            candidate = math.fsum(weighted_values) * scale
        if not math.isfinite(candidate):
            raise GeomodelContractError(
                "finite values produced a non-finite mean",
                object_name=object_name,
                field=field,
                expected="finite permutation-invariant mean",
                actual=str(candidate),
            )
        result[column_index] = candidate

    output_shape = moved.shape[1:]
    if not output_shape:
        return np.float64(result[0])
    return cast(FloatArray, result.reshape(output_shape))


class Geometry(ABC):
    """
    Abstract base for spatial geometries.

    Every concrete geometry provides a ``(npoints, ndim)``
    coordinate array. ``ndim`` is 2 or 3. The shared properties
    (``ndim``, ``npoints``, ``bounds``, ``centroid``) are derived
    from ``coords`` and need no overrides.
    """

    @property
    @abstractmethod
    def coords(self) -> FloatArray:
        """``(npoints, ndim)`` float64 array of coordinates."""

    @property
    def ndim(self) -> int:
        return int(self.coords.shape[1])

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Canonical public coordinate names in declared order."""
        return ("x", "y") if self.ndim == 2 else ("x", "y", "z")

    @property
    def coordinate_unit(self) -> str:
        """Canonical public spatial unit."""
        return "m"

    @property
    def npoints(self) -> int:
        return int(self.coords.shape[0])

    @property
    def bounds(self) -> tuple[float, ...]:
        """
        Axis-aligned bounding box.

        Returns a flat tuple::

            2-D: (x_min, y_min, x_max, y_max)
            3-D: (x_min, y_min, z_min, x_max, y_max, z_max)
        """
        if self.npoints == 0:
            raise GeomodelContractError(
                "bounds are undefined for an empty geometry",
                object_name=type(self).__name__,
                field="coords",
                expected=">= 1 point",
                actual=0,
            )
        mins = np.min(self.coords, axis=0)
        maxs = np.max(self.coords, axis=0)
        return tuple(mins.tolist()) + tuple(maxs.tolist())

    @property
    def centroid(self) -> FloatArray:
        if self.npoints == 0:
            raise GeomodelContractError(
                "centroid is undefined for an empty geometry",
                object_name=type(self).__name__,
                field="coords",
                expected=">= 1 point",
                actual=0,
            )
        return cast(
            FloatArray,
            _canonical_finite_mean(
                self.coords,
                axis=0,
                object_name=type(self).__name__,
                field="coords",
            ),
        )


__all__ = ["Geometry"]
