"""Sparse-occupancy exact neighbourhoods for regular cell-centre grids.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from ..frames.geo_grid import GeoGrid
from ..errors import GeomodelContractError, GeomodelResourceError
from .contracts import NeighbourhoodSelection, NeighbourhoodSpec
from .exhaustive import (
    _distance_squared,
    _finalize,
    _owned_source_ids,
    _target,
)

IndexTuple = tuple[int, ...]
FloatArray = np.ndarray[tuple[Any, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[Any, ...], np.dtype[np.int64]]
BoolArray = np.ndarray[tuple[Any, ...], np.dtype[np.bool_]]


class RegularGridNeighbourhood:
    """Exact occupied-cell queries using cached conservative grid offsets.

    Args:
        shape / origin_m / spacing_m: the regular grid definition.
        source_ids: ids returned by selections.
        occupied: optional occupancy mask of grid nodes.
    """

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        origin_m: tuple[float, ...],
        spacing_m: tuple[float, ...],
        source_ids: object,
        occupied: object | None = None,
    ) -> None:
        try:
            raw_shape = tuple(shape)
        except Exception as exc:
            raise GeomodelContractError(
                "regular-grid shape must be an iterable dimension tuple",
                object_name=type(self).__name__,
                field="shape",
                expected="2 or 3 positive exact integers",
                actual=type(shape).__name__,
            ) from exc
        if len(raw_shape) not in (2, 3) or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
            for value in raw_shape
        ):
            raise GeomodelContractError(
                "regular-grid shape must declare two or three positive dimensions",
                object_name=type(self).__name__,
                field="shape",
                expected="2 or 3 positive exact integers",
                actual=raw_shape,
            )
        canonical_shape = tuple(int(value) for value in raw_shape)
        canonical_grid = GeoGrid(
            shape=canonical_shape,
            origin=origin_m,
            spacing=spacing_m,
        )
        self._shape = canonical_grid.shape
        self._ndim = canonical_grid.ndim
        self._origin_m = canonical_grid.origin
        self._spacing_m = canonical_grid.spacing
        count = math.prod(canonical_shape)
        if occupied is None:
            mask: BoolArray = np.ones(count, dtype=np.bool_)
        else:
            try:
                mask = np.asarray(occupied)
            except Exception as exc:
                raise GeomodelContractError(
                    "regular-grid occupancy cannot be converted to an array",
                    object_name=type(self).__name__,
                    field="occupied",
                    expected=f"bool {canonical_shape} or ({count},)",
                    actual=type(occupied).__name__,
                ) from exc
            if mask.shape == canonical_shape:
                mask = mask.ravel(order="F")
            if mask.shape != (count,) or mask.dtype != np.bool_:
                raise GeomodelContractError(
                    "regular-grid occupancy must be a matching bool array",
                    object_name=type(self).__name__,
                    field="occupied",
                    expected=f"bool {canonical_shape} or ({count},)",
                    actual={"shape": tuple(mask.shape), "dtype": str(mask.dtype)},
                )
            mask = np.array(mask, dtype=np.bool_, copy=True)
        occupied_flat = np.flatnonzero(mask)
        try:
            raw_ids = np.asarray(source_ids)
        except Exception as exc:
            raise GeomodelContractError(
                "regular-grid source ids cannot be converted to an array",
                object_name=type(self).__name__,
                field="source_ids",
                expected=f"({count},) or occupied integer ids",
                actual=type(source_ids).__name__,
            ) from exc
        if raw_ids.ndim != 1:
            raise GeomodelContractError(
                "regular-grid source ids must be one-dimensional",
                object_name=type(self).__name__,
                field="source_ids",
                expected=f"({count},) or ({occupied_flat.size},) integer ids",
                actual={"shape": tuple(raw_ids.shape), "dtype": str(raw_ids.dtype)},
            )
        if raw_ids.size == count:
            selected_raw_ids = raw_ids[occupied_flat]
        elif raw_ids.size == occupied_flat.size:
            selected_raw_ids = raw_ids
        else:
            raise GeomodelContractError(
                "regular-grid source ids do not match the grid or occupancy",
                object_name=type(self).__name__,
                field="source_ids",
                expected=f"{count} grid ids or {occupied_flat.size} occupied ids",
                actual=int(raw_ids.size),
            )
        ids = _owned_source_ids(selected_raw_ids, count=occupied_flat.size)
        occupied_indices = np.asarray(
            np.unravel_index(occupied_flat, canonical_shape, order="F"),
            dtype=np.int64,
        ).T
        self._occupied_by_index: dict[IndexTuple, int] = {
            tuple(int(value) for value in index): int(source_id)
            for source_id, index in zip(ids.tolist(), occupied_indices.tolist())
        }
        self._offset_cache: dict[NeighbourhoodSpec, tuple[IndexTuple, ...]] = {}

    @classmethod
    def from_grid(
        cls,
        grid: GeoGrid,
        source_ids: object,
        *,
        occupied: object | None = None,
    ) -> RegularGridNeighbourhood:
        if not isinstance(grid, GeoGrid):
            raise GeomodelContractError(
                "regular-grid neighbourhood requires GeoGrid",
                object_name=cls.__name__,
                field="grid",
                expected="GeoGrid",
                actual=type(grid).__name__,
            )
        grid._validate_derived_geometry(object_name=cls.__name__)
        return cls(
            shape=grid.shape,
            origin_m=grid.origin,
            spacing_m=grid.spacing,
            source_ids=source_ids,
            occupied=occupied,
        )

    @property
    def source_count(self) -> int:
        return len(self._occupied_by_index)

    def _target_index(self, target: FloatArray) -> IndexTuple:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            index_float = (target - np.asarray(self._origin_m, dtype=np.float64)) / np.asarray(
                self._spacing_m, dtype=np.float64
            ) - 0.5
        if not bool(np.isfinite(index_float).all()):
            raise GeomodelContractError(
                "regular-grid query target cannot be resolved to a cell centre",
                object_name=type(self).__name__,
                field="target_m",
                expected="declared grid cell centre",
                actual=target.tolist(),
            )
        nearest = np.rint(index_float)
        index = tuple(int(value) for value in nearest.tolist())
        if any(value < 0 or value >= self._shape[axis] for axis, value in enumerate(index)):
            raise GeomodelContractError(
                "regular-grid query target is outside the grid",
                object_name=type(self).__name__,
                field="target_m",
                expected="cell centre inside grid",
                actual=target.tolist(),
            )
        reconstructed = np.asarray(
            [
                self._origin_m[axis] + (index[axis] + 0.5) * self._spacing_m[axis]
                for axis in range(self._ndim)
            ],
            dtype=np.float64,
        )
        if not np.array_equal(target, reconstructed):
            raise GeomodelContractError(
                "regular-grid query target must be an exact cell centre",
                object_name=type(self).__name__,
                field="target_m",
                expected="declared grid cell centre",
                actual=target.tolist(),
            )
        return index

    def _offsets(self, spec: NeighbourhoodSpec) -> tuple[IndexTuple, ...]:
        cached = self._offset_cache.get(spec)
        if cached is not None:
            return cached
        maximum_radius = max(spec.radii_m)
        half_width_values: list[int] = []
        for axis, spacing in enumerate(self._spacing_m):
            try:
                ratio = maximum_radius / spacing
            except OverflowError:
                ratio = math.inf
            maximum_grid_offset = self._shape[axis] - 1
            if not math.isfinite(ratio) or ratio >= maximum_grid_offset:
                half_width_values.append(maximum_grid_offset)
            else:
                # One additional cell makes the table conservative when
                # independently rounded cell centres straddle a declared-
                # spacing ULP boundary.  Offsets beyond the finite grid can
                # never identify an occupied cell, so clip them before the
                # resource estimate and Cartesian product.
                half_width_values.append(min(int(math.ceil(ratio)) + 1, maximum_grid_offset))
        half_widths = tuple(half_width_values)
        candidate_count = math.prod(2 * width + 1 for width in half_widths)
        if candidate_count > 10_000_000:
            raise GeomodelResourceError(
                "regular-grid neighbourhood offset table exceeds the deterministic limit",
                object_name=type(self).__name__,
                field="radii_m",
                expected="at most 10,000,000 conservative offsets",
                actual=candidate_count,
            )
        offsets = np.asarray(
            list(itertools.product(*(range(-width, width + 1) for width in half_widths))),
            dtype=np.int64,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            physical = offsets.astype(np.float64) * np.asarray(
                self._spacing_m,
                dtype=np.float64,
            )
        zero: FloatArray = np.zeros(self._ndim, dtype=np.float64)
        distances = _distance_squared(physical, zero, spec)
        # Ideal ``offset * spacing`` distances are suitable for deterministic
        # ordering only.  Filtering by them can drop a real cell whose two
        # independently rounded centres are inside the exact query boundary.
        # The shared finalizer performs the authoritative inclusion test.
        retained_offsets = offsets
        retained_distances = distances
        # The offset tuple is the deterministic final tie breaker here; stable
        # source id remains the final query tie breaker in `_finalize`.
        lex_keys: list[IntArray | FloatArray] = [
            retained_offsets[:, axis] for axis in reversed(range(self._ndim))
        ]
        lex_keys.append(retained_distances)
        order = np.lexsort(tuple(lex_keys[:-1]) + (lex_keys[-1],))
        result = tuple(
            tuple(int(value) for value in retained_offsets[position].tolist())
            for position in order.tolist()
        )
        self._offset_cache[spec] = result
        return result

    def query(self, target_m: object, spec: NeighbourhoodSpec) -> NeighbourhoodSelection:
        if not isinstance(spec, NeighbourhoodSpec):
            raise GeomodelContractError(
                "query requires NeighbourhoodSpec",
                object_name=type(self).__name__,
                field="spec",
                expected="NeighbourhoodSpec",
                actual=type(spec).__name__,
            )
        if spec.ndim != self._ndim:
            raise GeomodelContractError(
                "neighbourhood specification dimension does not match grid",
                object_name=type(self).__name__,
                field="spec.radii_m",
                expected=f"{self._ndim} radii",
                actual=spec.radii_m,
            )
        target = _target(target_m, ndim=self._ndim)
        target_index = self._target_index(target)
        selected_ids: list[int] = []
        selected_coordinates: list[tuple[float, ...]] = []
        for offset in self._offsets(spec):
            index = tuple(target_index[axis] + offset[axis] for axis in range(self._ndim))
            source_id = self._occupied_by_index.get(index)
            if source_id is None:
                continue
            selected_ids.append(source_id)
            selected_coordinates.append(
                tuple(
                    self._origin_m[axis] + (index[axis] + 0.5) * self._spacing_m[axis]
                    for axis in range(self._ndim)
                )
            )
        coordinates = np.asarray(selected_coordinates, dtype=np.float64).reshape(-1, self._ndim)
        ids = np.asarray(selected_ids, dtype=np.int64)
        distances = _distance_squared(coordinates, target, spec)
        return _finalize(
            ids,
            distances,
            distance_checks=int(ids.size),
            spec=spec,
        )


__all__ = ["RegularGridNeighbourhood"]
