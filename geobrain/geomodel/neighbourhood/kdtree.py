"""Exact post-filtered static and append-only KD-tree neighbourhoods.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
import threading
from typing import Any, cast

import numpy as np

from ..errors import GeomodelContractError
from .contracts import NeighbourhoodSelection, NeighbourhoodSpec
from .exhaustive import (
    _distance_squared,
    _finalize,
    _owned_coordinates,
    _owned_source_ids,
    _target,
)

FloatArray = np.ndarray[tuple[Any, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[Any, ...], np.dtype[np.int64]]


def _batch_size(value: object, *, object_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GeomodelContractError(
            "KD-tree rebuild batch size must be a positive exact integer",
            object_name=object_name,
            field="rebuild_batch_size",
            expected="positive non-boolean int",
            actual=value,
        )
    return value


def _tree(coordinates_m: FloatArray) -> Any:
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]

    return cKDTree(coordinates_m)


def _candidate_indices(tree: Any, target_m: FloatArray, radius_m: float) -> IntArray:
    # ``math.nextafter(max_float, inf)`` returns ``inf`` without emitting the
    # NumPy overflow warning that would otherwise violate warnings-as-errors.
    # An infinite query radius is the only conservative successor at that
    # boundary and the shared exact post-filter remains authoritative.
    conservative_radius = math.nextafter(radius_m, math.inf)
    values = tree.query_ball_point(target_m, conservative_radius, p=np.inf)
    return cast(IntArray, np.asarray(values, dtype=np.int64))


class StaticKDTreeNeighbourhood:
    """Immutable conservative KD-tree with exact ellipsoid post-filtering.

    Args:
        coordinates_m: source coordinates [m].
        source_ids: ids returned by selections.
    """

    def __init__(self, coordinates_m: object, source_ids: object) -> None:
        coordinates = _owned_coordinates(coordinates_m)
        ids = _owned_source_ids(source_ids, count=coordinates.shape[0])
        self._coordinates_m = coordinates
        self._source_ids = ids
        self._ndim = int(coordinates.shape[1])
        self._tree = _tree(coordinates)

    @classmethod
    def from_arrays(
        cls,
        coordinates_m: object,
        source_ids: object,
        *,
        rebuild_batch_size: int | None = None,
    ) -> StaticKDTreeNeighbourhood:
        if rebuild_batch_size is not None:
            _batch_size(rebuild_batch_size, object_name=cls.__name__)
        return cls(coordinates_m, source_ids)

    @property
    def source_count(self) -> int:
        return int(self._source_ids.size)

    def query(self, target_m: object, spec: NeighbourhoodSpec) -> NeighbourhoodSelection:
        if not isinstance(spec, NeighbourhoodSpec):
            raise GeomodelContractError(
                "query requires NeighbourhoodSpec",
                object_name=type(self).__name__,
                field="spec",
                expected="NeighbourhoodSpec",
                actual=type(spec).__name__,
            )
        target = _target(target_m, ndim=self._ndim)
        if spec.ndim != self._ndim:
            raise GeomodelContractError(
                "neighbourhood specification dimension does not match coordinates",
                object_name=type(self).__name__,
                field="spec.radii_m",
                expected=f"{self._ndim} radii",
                actual=spec.radii_m,
            )
        candidate_indices = _candidate_indices(self._tree, target, max(spec.radii_m))
        candidate_coordinates = self._coordinates_m[candidate_indices]
        candidate_ids = self._source_ids[candidate_indices]
        distances = _distance_squared(candidate_coordinates, target, spec)
        return _finalize(
            candidate_ids,
            distances,
            distance_checks=int(candidate_indices.size),
            spec=spec,
        )


class DynamicKDTreeNeighbourhood:
    """Append-only exact index with deterministic fixed-batch rebuilds.

    Args:
        coordinates_m: source coordinates [m].
        source_ids: ids returned by selections.
        rebuild_batch_size: insertions between k-d tree rebuilds.
    """

    def __init__(
        self,
        coordinates_m: object,
        source_ids: object,
        *,
        rebuild_batch_size: int,
    ) -> None:
        coordinates = _owned_coordinates(coordinates_m)
        ids = _owned_source_ids(source_ids, count=coordinates.shape[0])
        self._tree_coordinates_m = coordinates
        self._tree_source_ids = ids
        self._ndim = int(coordinates.shape[1])
        self._tree = _tree(coordinates)
        self._rebuild_batch_size = _batch_size(
            rebuild_batch_size,
            object_name=type(self).__name__,
        )
        self._buffer_coordinates: list[FloatArray] = []
        self._buffer_source_ids: list[int] = []
        self._all_source_ids = {int(value) for value in ids.tolist()}
        self._index_rebuilds = 0
        self._lock = threading.RLock()

    @classmethod
    def from_arrays(
        cls,
        coordinates_m: object,
        source_ids: object,
        *,
        rebuild_batch_size: int,
    ) -> DynamicKDTreeNeighbourhood:
        return cls(
            coordinates_m,
            source_ids,
            rebuild_batch_size=rebuild_batch_size,
        )

    @property
    def source_count(self) -> int:
        with self._lock:
            return len(self._all_source_ids)

    @property
    def buffered_count(self) -> int:
        with self._lock:
            return len(self._buffer_source_ids)

    @property
    def index_rebuilds(self) -> int:
        with self._lock:
            return self._index_rebuilds

    def append(self, source_id: int, coordinate_m: np.ndarray) -> None:
        """Append one new stable id and rebuild on the exact configured batch."""
        if (
            isinstance(source_id, bool)
            or not isinstance(source_id, (int, np.integer))
            or not np.iinfo(np.int64).min <= int(source_id) <= np.iinfo(np.int64).max
        ):
            raise GeomodelContractError(
                "dynamic source id must be an exact int64 integer",
                object_name=type(self).__name__,
                field="source_id",
                expected="unique non-boolean int64",
                actual=source_id,
            )
        resolved_id = int(source_id)
        coordinate = _target(coordinate_m, ndim=self._ndim)
        with self._lock:
            if resolved_id in self._all_source_ids:
                raise GeomodelContractError(
                    "dynamic source id may be appended only once",
                    object_name=type(self).__name__,
                    field="source_id",
                    expected="new unique source id",
                    actual=resolved_id,
                )
            self._all_source_ids.add(resolved_id)
            self._buffer_source_ids.append(resolved_id)
            self._buffer_coordinates.append(coordinate.copy())
            if len(self._buffer_source_ids) == self._rebuild_batch_size:
                self._rebuild()

    def _rebuild(self) -> None:
        buffered_coordinates = np.asarray(self._buffer_coordinates, dtype=np.float64)
        buffered_ids = np.asarray(self._buffer_source_ids, dtype=np.int64)
        combined_coordinates = np.concatenate(
            (self._tree_coordinates_m, buffered_coordinates),
            axis=0,
        )
        combined_ids = np.concatenate((self._tree_source_ids, buffered_ids), axis=0)
        self._tree_coordinates_m = _owned_coordinates(combined_coordinates)
        self._tree_source_ids = _owned_source_ids(
            combined_ids,
            count=combined_ids.size,
        )
        self._tree = _tree(self._tree_coordinates_m)
        self._buffer_coordinates.clear()
        self._buffer_source_ids.clear()
        self._index_rebuilds += 1

    def query(self, target_m: object, spec: NeighbourhoodSpec) -> NeighbourhoodSelection:
        if not isinstance(spec, NeighbourhoodSpec):
            raise GeomodelContractError(
                "query requires NeighbourhoodSpec",
                object_name=type(self).__name__,
                field="spec",
                expected="NeighbourhoodSpec",
                actual=type(spec).__name__,
            )
        target = _target(target_m, ndim=self._ndim)
        if spec.ndim != self._ndim:
            raise GeomodelContractError(
                "neighbourhood specification dimension does not match coordinates",
                object_name=type(self).__name__,
                field="spec.radii_m",
                expected=f"{self._ndim} radii",
                actual=spec.radii_m,
            )
        with self._lock:
            tree = self._tree
            tree_coordinates_m = self._tree_coordinates_m
            tree_source_ids = self._tree_source_ids
            buffer_coordinates = np.asarray(self._buffer_coordinates, dtype=np.float64).reshape(
                -1, self._ndim
            )
            buffer_ids = np.asarray(self._buffer_source_ids, dtype=np.int64)
        tree_indices = _candidate_indices(tree, target, max(spec.radii_m))
        coordinates = tree_coordinates_m[tree_indices]
        ids = tree_source_ids[tree_indices]
        if buffer_ids.size:
            coordinates = np.concatenate((coordinates, buffer_coordinates), axis=0)
            ids = np.concatenate((ids, buffer_ids), axis=0)
        distances = _distance_squared(coordinates, target, spec)
        return _finalize(
            np.asarray(ids, dtype=np.int64),
            distances,
            distance_checks=int(ids.size),
            spec=spec,
        )


__all__ = ["DynamicKDTreeNeighbourhood", "StaticKDTreeNeighbourhood"]
