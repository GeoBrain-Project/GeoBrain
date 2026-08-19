"""Declared-dimensional regular cell grid with metre lower-corner geometry.

Public geometry is two- or three-dimensional. Property storage keeps GSLIB's
x-fastest flat order, while a two-dimensional grid never fabricates a z axis.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, cast

import numpy as np

from ..errors import GeomodelContractError
from ._arrays import FloatArray, column_stack_float
from .geometry import Geometry


def _items(value: Iterable[Any], *, field: str) -> tuple[Any, ...]:
    try:
        return tuple(value)
    except TypeError as exc:
        raise GeomodelContractError(
            "GeoGrid dimension values must be iterable",
            object_name="GeoGrid",
            field=field,
            expected="tuple of length 2 or 3",
            actual=value,
        ) from exc


def _shape_tuple(value: Iterable[Any]) -> tuple[int, ...]:
    raw = _items(value, field="shape")
    if len(raw) not in (2, 3) or any(
        isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in raw
    ):
        raise GeomodelContractError(
            "GeoGrid shape must declare two or three integer dimensions",
            object_name="GeoGrid",
            field="shape",
            expected="2 or 3 positive integers",
            actual=raw,
        )
    result = tuple(int(item) for item in raw)
    if any(item <= 0 for item in result):
        raise GeomodelContractError(
            "GeoGrid shape dimensions must be positive",
            object_name="GeoGrid",
            field="shape",
            expected="all > 0",
            actual=result,
        )
    index_capacity = int(np.iinfo(np.intp).max)
    if any(item > index_capacity for item in result) or math.prod(result) > index_capacity:
        raise GeomodelContractError(
            "GeoGrid shape exceeds the NumPy index capacity",
            object_name="GeoGrid",
            field="shape",
            expected={
                "maximum_axis": index_capacity,
                "maximum_cells": index_capacity,
            },
            actual=result,
        )
    return result


def _float_tuple(
    value: Iterable[Any] | None,
    *,
    field: str,
    ndim: int,
    default: float,
    positive: bool,
) -> tuple[float, ...]:
    raw = (default,) * ndim if value is None else _items(value, field=field)
    try:
        result = tuple(float(item) for item in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeomodelContractError(
            f"GeoGrid {field} values must be numeric",
            object_name="GeoGrid",
            field=field,
            expected=f"{ndim} finite numbers",
            actual=raw,
        ) from exc
    if len(result) != ndim or not all(math.isfinite(item) for item in result):
        raise GeomodelContractError(
            f"GeoGrid {field} must match the declared dimension",
            object_name="GeoGrid",
            field=field,
            expected=f"{ndim} finite numbers",
            actual=result,
        )
    if positive and any(item <= 0.0 for item in result):
        raise GeomodelContractError(
            "GeoGrid spacing values must be positive and finite",
            object_name="GeoGrid",
            field=field,
            expected="all > 0 and finite",
            actual=result,
        )
    return result


def _apply_float_overrides(
    values: list[float],
    overrides: tuple[object | None, ...],
    fields: tuple[str, ...],
) -> None:
    """Apply legacy scalar overrides without leaking raw conversion errors."""
    for axis, (override, field) in enumerate(zip(overrides, fields)):
        if override is None:
            continue
        try:
            values[axis] = float(cast(Any, override))
        except (TypeError, ValueError, OverflowError) as exc:
            raise GeomodelContractError(
                "GeoGrid scalar override must be numeric",
                object_name="GeoGrid",
                field=field,
                expected="finite number",
                actual=override,
            ) from exc


def _validate_derived_geometry(
    shape: tuple[int, ...],
    origin: tuple[float, ...],
    spacing: tuple[float, ...],
    *,
    object_name: str,
) -> None:
    """Reject non-finite bounds or centres without allocating coordinates."""
    axis_names = ("x", "y") if len(shape) == 2 else ("x", "y", "z")
    for axis, (count, lower, width) in enumerate(zip(shape, origin, spacing)):
        try:
            count_float = float(count)
            upper = lower + count_float * width
            first_center = lower + 0.5 * width
            last_center = lower + (count_float - 0.5) * width
        except OverflowError as exc:
            raise GeomodelContractError(
                "GeoGrid derived geometry is not representable in float64",
                object_name=object_name,
                field="bounds",
                expected="finite outer bounds and cell centres",
                actual={
                    "axis": axis_names[axis],
                    "shape": count,
                    "origin": lower,
                    "spacing": width,
                },
            ) from exc
        if not math.isfinite(upper):
            raise GeomodelContractError(
                "GeoGrid outer bound is not finite",
                object_name=object_name,
                field="bounds",
                expected="finite origin + shape * spacing",
                actual={
                    "axis": axis_names[axis],
                    "shape": count,
                    "origin": lower,
                    "spacing": width,
                },
            )
        if upper <= lower:
            raise GeomodelContractError(
                "GeoGrid outer bound is not strictly representable",
                object_name=object_name,
                field="bounds",
                expected="finite origin < origin + shape * spacing",
                actual={
                    "axis": axis_names[axis],
                    "lower": lower,
                    "upper": upper,
                    "shape": count,
                    "spacing": width,
                },
            )
        if (
            not math.isfinite(first_center)
            or not math.isfinite(last_center)
            or not lower < first_center < upper
            or not lower < last_center < upper
        ):
            raise GeomodelContractError(
                "GeoGrid cell centres are not strictly representable inside the bounds",
                object_name=object_name,
                field="coords",
                expected="finite lower < first/last centre < upper",
                actual={
                    "axis": axis_names[axis],
                    "shape": count,
                    "origin": lower,
                    "spacing": width,
                    "first_center": first_center,
                    "last_center": last_center,
                    "upper": upper,
                },
            )
        if count > 1:
            second_center = lower + 1.5 * width
            penultimate_center = lower + (count_float - 1.5) * width
            if not (
                first_center < second_center <= last_center
                and first_center <= penultimate_center < last_center
            ):
                raise GeomodelContractError(
                    "GeoGrid adjacent cell centres are not distinctly representable",
                    object_name=object_name,
                    field="coords",
                    expected="strictly increasing adjacent cell centres",
                    actual={
                        "axis": axis_names[axis],
                        "shape": count,
                        "origin": lower,
                        "spacing": width,
                        "first_center": first_center,
                        "second_center": second_center,
                        "penultimate_center": penultimate_center,
                        "last_center": last_center,
                    },
                )


def _exact_grid_index(value: object, *, field: str) -> int:
    """Normalize one exact Python/NumPy integer without admitting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise GeomodelContractError(
            "grid index must be an exact integer",
            object_name="GeoGrid.ijk_to_index",
            field=field,
            expected="Python or NumPy integer (not bool)",
            actual=value,
        )
    return int(value)


class GeoGrid(Geometry):
    """Two- or three-dimensional regular grid of cell centres.

    ``origin`` is the lower outside cell corner in metres. ``shape``,
    ``origin``, ``spacing``, coordinates, bounds, and axis names all retain the
    declared rank. The explicit ``nx``/``xmn`` form remains available as a
    supported alternative spelling; those values obey the same lower-corner
    convention.

    Args:
        shape / origin / spacing: array-style grid definition.
        nx / ny / nz, xmn / ymn / zmn, xsiz / ysiz / zsiz: GSLIB-style
            per-axis definition (alternative to the array form).
    """

    def __init__(
        self,
        shape: Iterable[Any] | None = None,
        origin: Iterable[Any] | None = None,
        spacing: Iterable[Any] | None = None,
        *,
        nx: int | None = None,
        ny: int | None = None,
        nz: int | None = None,
        xmn: float | None = None,
        ymn: float | None = None,
        zmn: float | None = None,
        xsiz: float | None = None,
        ysiz: float | None = None,
        zsiz: float | None = None,
    ) -> None:
        if shape is None:
            if nx is None or ny is None:
                raise GeomodelContractError(
                    "GeoGrid requires shape or nx and ny",
                    object_name="GeoGrid",
                    field="shape",
                    expected="(nx, ny) or (nx, ny, nz)",
                    actual=None,
                )
            shape = (nx, ny) if nz is None else (nx, ny, nz)
        shape_values = list(_shape_tuple(shape))
        ndim = len(shape_values)
        overrides = (nx, ny) if ndim == 2 else (nx, ny, nz)
        for axis, shape_override in enumerate(overrides):
            if shape_override is not None:
                if (
                    isinstance(shape_override, bool)
                    or not isinstance(shape_override, int)
                    or shape_override <= 0
                ):
                    raise GeomodelContractError(
                        "GeoGrid shape override must be a positive integer",
                        object_name="GeoGrid",
                        field=("nx", "ny", "nz")[axis],
                        expected="positive integer",
                        actual=shape_override,
                    )
                shape_values[axis] = shape_override
        if ndim == 2 and nz is not None:
            raise GeomodelContractError(
                "a 2-D GeoGrid cannot accept a z dimension",
                object_name="GeoGrid",
                field="nz",
                expected=None,
                actual=nz,
            )

        origin_values = list(
            _float_tuple(origin, field="origin", ndim=ndim, default=0.0, positive=False)
        )
        spacing_values = list(
            _float_tuple(spacing, field="spacing", ndim=ndim, default=1.0, positive=True)
        )
        origin_overrides = (xmn, ymn) if ndim == 2 else (xmn, ymn, zmn)
        spacing_overrides = (xsiz, ysiz) if ndim == 2 else (xsiz, ysiz, zsiz)
        if ndim == 2 and (zmn is not None or zsiz is not None):
            raise GeomodelContractError(
                "a 2-D GeoGrid cannot accept z origin or spacing",
                object_name="GeoGrid",
                field="z",
                expected=None,
                actual={"zmn": zmn, "zsiz": zsiz},
            )
        _apply_float_overrides(
            origin_values,
            origin_overrides,
            ("xmn", "ymn") if ndim == 2 else ("xmn", "ymn", "zmn"),
        )
        _apply_float_overrides(
            spacing_values,
            spacing_overrides,
            ("xsiz", "ysiz") if ndim == 2 else ("xsiz", "ysiz", "zsiz"),
        )

        self._shape = _shape_tuple(shape_values)
        self._origin = _float_tuple(
            origin_values,
            field="origin",
            ndim=ndim,
            default=0.0,
            positive=False,
        )
        self._spacing = _float_tuple(
            spacing_values,
            field="spacing",
            ndim=ndim,
            default=1.0,
            positive=True,
        )
        self._validate_derived_geometry()
        self.nx = self._shape[0]
        self.ny = self._shape[1]
        self.xmn = self._origin[0]
        self.ymn = self._origin[1]
        self.xsiz = self._spacing[0]
        self.ysiz = self._spacing[1]
        if ndim == 3:
            self.nz = self._shape[2]
            self.zmn = self._origin[2]
            self.zsiz = self._spacing[2]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def origin(self) -> tuple[float, ...]:
        return self._origin

    @property
    def spacing(self) -> tuple[float, ...]:
        return self._spacing

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def axis_names(self) -> tuple[str, ...]:
        return ("x", "y") if self.ndim == 2 else ("x", "y", "z")

    def _validate_derived_geometry(self, *, object_name: str = "GeoGrid") -> None:
        """Revalidate derived geometry at boundaries that issue identities."""
        _validate_derived_geometry(
            self._shape,
            self._origin,
            self._spacing,
            object_name=object_name,
        )

    @property
    def ncells(self) -> int:
        return math.prod(self._shape)

    @property
    def npoints(self) -> int:
        return self.ncells

    @property
    def coords(self) -> FloatArray:
        """Cell-centre coordinates in x-fastest flat order."""
        axes = [
            lower + (np.arange(count, dtype=np.float64) + 0.5) * width
            for count, lower, width in zip(self._shape, self._origin, self._spacing)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        coordinates = column_stack_float(tuple(axis.ravel(order="F") for axis in mesh))
        coordinates.flags.writeable = False
        return coordinates

    @property
    def bounds(self) -> tuple[float, ...]:
        """Half-open outer cell extents flattened as lower then upper values."""
        upper = tuple(
            lower + count * width
            for lower, count, width in zip(self._origin, self._shape, self._spacing)
        )
        return self._origin + upper

    @property
    def centroid(self) -> FloatArray:
        values: FloatArray = np.asarray(
            [
                lower + 0.5 * count * width
                for lower, count, width in zip(self._origin, self._shape, self._spacing)
            ],
            dtype=np.float64,
        )
        values.flags.writeable = False
        return values

    def ijk_to_index(self, i: int, j: int, k: int | None = None) -> int:
        """Convert declared-dimensional grid indices to x-fastest order."""
        i_index = _exact_grid_index(i, field="i")
        j_index = _exact_grid_index(j, field="j")
        if self.ndim == 2:
            if k is not None:
                raise GeomodelContractError(
                    "a 2-D GeoGrid has no z index",
                    object_name="GeoGrid.ijk_to_index",
                    field="k",
                    expected=None,
                    actual=k,
                )
            if not (0 <= i_index < self.nx and 0 <= j_index < self.ny):
                raise GeomodelContractError(
                    "grid index is out of range",
                    object_name="GeoGrid.ijk_to_index",
                    field="ij",
                    expected=f"0<=i<{self.nx}, 0<=j<{self.ny}",
                    actual=(i_index, j_index),
                )
            return i_index + j_index * self.nx

        z_index = 0 if k is None else _exact_grid_index(k, field="k")
        if not (0 <= i_index < self.nx and 0 <= j_index < self.ny and 0 <= z_index < self.nz):
            raise GeomodelContractError(
                "grid index is out of range",
                object_name="GeoGrid.ijk_to_index",
                field="ijk",
                expected=f"0<=i<{self.nx}, 0<=j<{self.ny}, 0<=k<{self.nz}",
                actual=(i_index, j_index, z_index),
            )
        return i_index + j_index * self.nx + z_index * self.nx * self.ny

    def index_to_ijk(self, index: int) -> tuple[int, ...]:
        """Convert one flat index to a tuple with the grid's declared rank."""
        if isinstance(index, bool) or not isinstance(index, int) or not (0 <= index < self.ncells):
            raise GeomodelContractError(
                "flat grid index is out of range",
                object_name="GeoGrid.index_to_ijk",
                field="index",
                expected=f"0 <= index < {self.ncells}",
                actual=index,
            )
        layer = self.nx * self.ny
        k = index // layer
        remainder = index % layer
        j = remainder // self.nx
        i = remainder % self.nx
        return (i, j) if self.ndim == 2 else (i, j, k)

    def __repr__(self) -> str:
        dimensions = "x".join(str(value) for value in self._shape)
        return f"GeoGrid({dimensions}, origin={self._origin}, spacing={self._spacing})"


__all__ = ["GeoGrid"]
