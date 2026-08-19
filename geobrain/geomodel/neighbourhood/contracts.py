"""Validated exact contracts shared by Geomodel neighbourhood backends.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import numpy as np

from ..errors import GeomodelContractError

FloatArray = np.ndarray[tuple[Any, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[Any, ...], np.dtype[np.int64]]
SelectionStatus = Literal["selected", "insufficient"]


def _contract_error(
    message: str,
    *,
    object_name: str,
    field: str,
    expected: object,
    actual: object,
) -> None:
    raise GeomodelContractError(
        message,
        object_name=object_name,
        field=field,
        expected=expected,
        actual=actual,
    )


def _owned_readonly_float(values: object, *, shape_rank: int) -> FloatArray:
    try:
        array = np.array(values, dtype="<f8", copy=True, order="C")
    except Exception as exc:
        raise GeomodelContractError(
            "neighbourhood values must be numeric",
            object_name="NeighbourhoodSelection",
            field="distance_squared",
            expected="finite float64 array",
            actual=type(values).__name__,
        ) from exc
    if array.ndim != shape_rank or not np.isfinite(array).all():
        _contract_error(
            "neighbourhood values have invalid shape or values",
            object_name="NeighbourhoodSelection",
            field="distance_squared",
            expected=f"finite rank-{shape_rank} float64 array",
            actual={"shape": tuple(array.shape), "dtype": str(array.dtype)},
        )
    return cast(
        FloatArray, np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)
    )


def _owned_readonly_int(values: object) -> IntArray:
    try:
        raw = np.asarray(values)
    except Exception as exc:
        raise GeomodelContractError(
            "neighbourhood ids cannot be converted to an array",
            object_name="NeighbourhoodSelection",
            field="ids",
            expected="rank-1 int64-compatible array",
            actual=type(values).__name__,
        ) from exc
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        _contract_error(
            "neighbourhood ids must be a one-dimensional integer array",
            object_name="NeighbourhoodSelection",
            field="ids",
            expected="rank-1 int64-compatible array",
            actual={"shape": tuple(raw.shape), "dtype": str(raw.dtype)},
        )
    if raw.dtype.kind == "u" and raw.size and int(raw.max()) > np.iinfo(np.int64).max:
        _contract_error(
            "neighbourhood id exceeds int64",
            object_name="NeighbourhoodSelection",
            field="ids",
            expected="int64 range",
            actual=int(raw.max()),
        )
    array = np.array(raw, dtype="<i8", copy=True, order="C")
    return cast(IntArray, np.frombuffer(array.tobytes(order="C"), dtype="<i8"))


@dataclass(frozen=True, slots=True)
class NeighbourhoodSpec:
    """Dimension-matched ellipsoidal search and deterministic limits.

    Attributes:
        radii_m: search ellipse radii [m].
        angles_deg: ellipse rotation angles [deg].
        min_neighbors / max_neighbors: selection bounds.
        include_radius: include-all radius [m].
    """

    radii_m: tuple[float, ...]
    angles_deg: tuple[float, ...]
    min_neighbors: int = 1
    max_neighbors: int = 32
    include_radius: bool = True

    def __post_init__(self) -> None:
        object_name = type(self).__name__
        try:
            raw_radii = tuple(self.radii_m)
            raw_angles = tuple(self.angles_deg)
            if any(isinstance(value, (bool, np.bool_)) for value in raw_radii + raw_angles):
                raise TypeError("boolean geometry")
            radii = tuple(float(value) for value in raw_radii)
            angles = tuple(float(value) for value in raw_angles)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GeomodelContractError(
                "neighbourhood geometry must be numeric",
                object_name=object_name,
                field="radii_m/angles_deg",
                expected="finite numeric tuples",
                actual={"radii_m": self.radii_m, "angles_deg": self.angles_deg},
            ) from exc
        if len(radii) not in (2, 3) or any(
            not math.isfinite(value) or value <= 0.0 for value in radii
        ):
            _contract_error(
                "neighbourhood radii must declare positive finite 2-D or 3-D metre axes",
                object_name=object_name,
                field="radii_m",
                expected="2 or 3 positive finite values",
                actual=radii,
            )
        expected_angles = 1 if len(radii) == 2 else 3
        if len(angles) != expected_angles or any(not math.isfinite(value) for value in angles):
            _contract_error(
                "neighbourhood angles must match the declared dimension",
                object_name=object_name,
                field="angles_deg",
                expected=f"{expected_angles} finite values",
                actual=angles,
            )
        if (
            isinstance(self.min_neighbors, (bool, np.bool_))
            or not isinstance(self.min_neighbors, (int, np.integer))
            or int(self.min_neighbors) < 0
        ):
            _contract_error(
                "minimum neighbour count must be an exact non-negative integer",
                object_name=object_name,
                field="min_neighbors",
                expected="non-boolean int >= 0",
                actual=self.min_neighbors,
            )
        if (
            isinstance(self.max_neighbors, (bool, np.bool_))
            or not isinstance(self.max_neighbors, (int, np.integer))
            or int(self.max_neighbors) < 1
            or int(self.max_neighbors) < int(self.min_neighbors)
        ):
            _contract_error(
                "maximum neighbour count must be an exact positive bound",
                object_name=object_name,
                field="max_neighbors",
                expected="int >= max(1, min_neighbors)",
                actual=self.max_neighbors,
            )
        if type(self.include_radius) is not bool:
            _contract_error(
                "radius inclusion selector must be bool",
                object_name=object_name,
                field="include_radius",
                expected="bool",
                actual=self.include_radius,
            )
        object.__setattr__(self, "radii_m", radii)
        object.__setattr__(self, "angles_deg", angles)
        object.__setattr__(self, "min_neighbors", int(self.min_neighbors))
        object.__setattr__(self, "max_neighbors", int(self.max_neighbors))

    @property
    def ndim(self) -> int:
        return len(self.radii_m)

    def to_dict(self) -> dict[str, object]:
        return {
            "radii_m": list(self.radii_m),
            "angles_deg": list(self.angles_deg),
            "min_neighbors": self.min_neighbors,
            "max_neighbors": self.max_neighbors,
            "include_radius": self.include_radius,
        }


@dataclass(frozen=True, slots=True)
class NeighbourhoodSelection:
    """Owned immutable deterministic query result.

    Attributes:
        ids: selected neighbour ids.
        distance_squared: squared distances of the selection.
        distance_checks: work counter of the search.
        status: selection status flag.
    """

    ids: IntArray
    distance_squared: FloatArray
    distance_checks: int
    status: SelectionStatus

    def __post_init__(self) -> None:
        ids = _owned_readonly_int(self.ids)
        distances = _owned_readonly_float(self.distance_squared, shape_rank=1)
        if ids.shape != distances.shape:
            _contract_error(
                "neighbourhood result arrays must align",
                object_name=type(self).__name__,
                field="ids/distance_squared",
                expected="equal one-dimensional shapes",
                actual={"ids": tuple(ids.shape), "distance_squared": tuple(distances.shape)},
            )
        if bool(np.any(distances < 0.0)):
            _contract_error(
                "anisotropic squared distances cannot be negative",
                object_name=type(self).__name__,
                field="distance_squared",
                expected=">= 0",
                actual=float(np.min(distances)),
            )
        if np.unique(ids).size != ids.size:
            _contract_error(
                "neighbourhood result ids must be unique",
                object_name=type(self).__name__,
                field="ids",
                expected="unique stable source ids",
                actual=ids.tolist(),
            )
        canonical_order = np.lexsort((ids, distances))
        if not np.array_equal(canonical_order, np.arange(ids.size, dtype=np.int64)):
            _contract_error(
                "neighbourhood results must use canonical distance/id order",
                object_name=type(self).__name__,
                field="ids/distance_squared",
                expected="sorted by (distance_squared, source_id)",
                actual={"ids": ids.tolist(), "distance_squared": distances.tolist()},
            )
        if (
            isinstance(self.distance_checks, (bool, np.bool_))
            or not isinstance(self.distance_checks, (int, np.integer))
            or int(self.distance_checks) < ids.size
        ):
            _contract_error(
                "distance-check count must include every returned candidate",
                object_name=type(self).__name__,
                field="distance_checks",
                expected=f"integer >= {ids.size}",
                actual=self.distance_checks,
            )
        if self.status not in ("selected", "insufficient"):
            _contract_error(
                "neighbourhood status is invalid",
                object_name=type(self).__name__,
                field="status",
                expected="'selected' or 'insufficient'",
                actual=self.status,
            )
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "distance_squared", distances)
        object.__setattr__(self, "distance_checks", int(self.distance_checks))


class NeighbourhoodBackend(Protocol):
    def query(
        self,
        target_m: FloatArray,
        spec: NeighbourhoodSpec,
    ) -> NeighbourhoodSelection: ...


__all__ = [
    "NeighbourhoodBackend",
    "NeighbourhoodSelection",
    "NeighbourhoodSpec",
]
