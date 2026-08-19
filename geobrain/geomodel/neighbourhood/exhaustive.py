"""Exact exhaustive Geomodel neighbourhood oracle and shared finalizer.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from typing import Any, Literal, cast

import numpy as np

from ..errors import GeomodelContractError
from .contracts import NeighbourhoodSelection, NeighbourhoodSpec

FloatArray = np.ndarray[tuple[Any, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[Any, ...], np.dtype[np.int64]]


def _owned_coordinates(values: object) -> FloatArray:
    try:
        array = np.array(values, dtype="<f8", order="C", copy=True)
    except Exception as exc:
        raise GeomodelContractError(
            "neighbourhood coordinates must be numeric",
            object_name="NeighbourhoodBackend",
            field="coordinates_m",
            expected="finite (n, 2|3) float64 array",
            actual=type(values).__name__,
        ) from exc
    if array.ndim != 2 or array.shape[1] not in (2, 3) or not np.isfinite(array).all():
        raise GeomodelContractError(
            "neighbourhood coordinates have invalid shape or values",
            object_name="NeighbourhoodBackend",
            field="coordinates_m",
            expected="finite (n, 2|3) float64 array",
            actual={"shape": tuple(array.shape), "dtype": str(array.dtype)},
        )
    return cast(
        FloatArray, np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)
    )


def _owned_source_ids(values: object, *, count: int) -> IntArray:
    try:
        raw = np.asarray(values)
    except Exception as exc:
        raise GeomodelContractError(
            "source ids cannot be converted to an array",
            object_name="NeighbourhoodBackend",
            field="source_ids",
            expected=f"({count},) int64-compatible",
            actual=type(values).__name__,
        ) from exc
    if raw.ndim != 1 or raw.shape[0] != count or raw.dtype.kind not in "iu":
        raise GeomodelContractError(
            "source ids must be a matching one-dimensional integer array",
            object_name="NeighbourhoodBackend",
            field="source_ids",
            expected=f"({count},) int64-compatible",
            actual={"shape": tuple(raw.shape), "dtype": str(raw.dtype)},
        )
    if raw.dtype.kind == "u" and raw.size and int(raw.max()) > np.iinfo(np.int64).max:
        raise GeomodelContractError(
            "source id exceeds int64",
            object_name="NeighbourhoodBackend",
            field="source_ids",
            expected="int64 range",
            actual=int(raw.max()),
        )
    array = np.array(raw, dtype="<i8", order="C", copy=True)
    if np.unique(array).size != count:
        raise GeomodelContractError(
            "source ids must be globally unique",
            object_name="NeighbourhoodBackend",
            field="source_ids",
            expected="unique stable ids",
            actual=array.tolist(),
        )
    return cast(IntArray, np.frombuffer(array.tobytes(order="C"), dtype="<i8"))


def _target(target_m: object, *, ndim: int) -> FloatArray:
    try:
        target = np.array(target_m, dtype="<f8", order="C", copy=True)
    except Exception as exc:
        raise GeomodelContractError(
            "neighbourhood target must be numeric",
            object_name="NeighbourhoodBackend.query",
            field="target_m",
            expected=f"finite ({ndim},) float64 coordinate",
            actual=target_m,
        ) from exc
    if target.shape != (ndim,) or not np.isfinite(target).all():
        raise GeomodelContractError(
            "neighbourhood target has invalid shape or values",
            object_name="NeighbourhoodBackend.query",
            field="target_m",
            expected=f"finite ({ndim},) float64 coordinate",
            actual={"shape": tuple(target.shape), "values": target.tolist()},
        )
    return cast(FloatArray, target)


def _rotation_transform(spec: NeighbourhoodSpec) -> FloatArray:
    """Return the orthonormal GSLIB principal-axis rotation rows."""
    if spec.ndim == 2:
        azimuth = math.radians(90.0 - spec.angles_deg[0])
        cosine = math.cos(azimuth)
        sine = math.sin(azimuth)
        return cast(
            FloatArray,
            np.asarray(
                [
                    [cosine, sine],
                    [-sine, cosine],
                ],
                dtype=np.float64,
            ),
        )

    azimuth, dip, plunge = spec.angles_deg
    if 0.0 <= azimuth < 270.0:
        alpha = math.radians(90.0 - azimuth)
    else:
        alpha = math.radians(450.0 - azimuth)
    beta = math.radians(-dip)
    theta = math.radians(plunge)
    sin_a, cos_a = math.sin(alpha), math.cos(alpha)
    sin_b, cos_b = math.sin(beta), math.cos(beta)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    return cast(
        FloatArray,
        np.asarray(
            [
                [cos_b * cos_a, cos_b * sin_a, -sin_b],
                [
                    -cos_t * sin_a + sin_t * sin_b * cos_a,
                    cos_t * cos_a + sin_t * sin_b * sin_a,
                    sin_t * cos_b,
                ],
                [
                    sin_t * sin_a + cos_t * sin_b * cos_a,
                    -sin_t * cos_a + cos_t * sin_b * sin_a,
                    cos_t * cos_b,
                ],
            ],
            dtype=np.float64,
        ),
    )


def _distance_squared(
    coordinates_m: FloatArray,
    target_m: FloatArray,
    spec: NeighbourhoodSpec,
) -> FloatArray:
    """Compute one canonical float64 normalized anisotropic distance path."""
    if coordinates_m.shape[1] != spec.ndim:
        raise GeomodelContractError(
            "neighbourhood specification dimension does not match coordinates",
            object_name="NeighbourhoodBackend.query",
            field="spec.radii_m",
            expected=f"{coordinates_m.shape[1]} radii",
            actual=spec.radii_m,
        )
    transform = _rotation_transform(spec)
    output: FloatArray = np.full(coordinates_m.shape[0], np.inf, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        deltas = coordinates_m - target_m
    finite = np.isfinite(deltas).all(axis=1)
    if bool(np.any(finite)):
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            finite_deltas = deltas[finite]
            rotated = np.empty_like(finite_deltas)
            # Fixed scalar accumulation order makes each candidate's bits
            # independent of backend candidate-batch shape and BLAS dispatch.
            for principal_axis in range(spec.ndim):
                component = finite_deltas[:, 0] * transform[principal_axis, 0]
                for coordinate_axis in range(1, spec.ndim):
                    component = component + (
                        finite_deltas[:, coordinate_axis]
                        * transform[principal_axis, coordinate_axis]
                    )
                rotated[:, principal_axis] = component
            normalized = rotated / np.asarray(spec.radii_m, dtype=np.float64)
            values = normalized[:, 0] * normalized[:, 0]
            for principal_axis in range(1, spec.ndim):
                values = values + normalized[:, principal_axis] * normalized[:, principal_axis]
        values = np.where(np.isfinite(values), values, np.inf)
        output[finite] = values
    return output


def _finalize(
    candidate_ids: IntArray,
    distance_squared: FloatArray,
    *,
    distance_checks: int,
    spec: NeighbourhoodSpec,
) -> NeighbourhoodSelection:
    inside = distance_squared <= 1.0 if spec.include_radius else distance_squared < 1.0
    ids = candidate_ids[inside]
    distances = distance_squared[inside]
    order = np.lexsort((ids, distances))
    selected = order[: spec.max_neighbors]
    ids_out = np.asarray(ids[selected], dtype=np.int64)
    distances_out = np.asarray(distances[selected], dtype=np.float64)
    status: Literal["selected", "insufficient"] = (
        "selected" if ids_out.size >= spec.min_neighbors else "insufficient"
    )
    return NeighbourhoodSelection(
        ids=ids_out,
        distance_squared=distances_out,
        distance_checks=distance_checks,
        status=status,
    )


class ExhaustiveNeighbourhood:
    """Small-problem oracle that checks every owned source coordinate.

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

    @classmethod
    def from_arrays(
        cls,
        coordinates_m: object,
        source_ids: object,
        *,
        rebuild_batch_size: int | None = None,
    ) -> ExhaustiveNeighbourhood:
        if rebuild_batch_size is not None and (
            isinstance(rebuild_batch_size, bool)
            or not isinstance(rebuild_batch_size, int)
            or rebuild_batch_size < 1
        ):
            raise GeomodelContractError(
                "rebuild batch size must be a positive integer",
                object_name=cls.__name__,
                field="rebuild_batch_size",
                expected="positive non-boolean int",
                actual=rebuild_batch_size,
            )
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
        distances = _distance_squared(self._coordinates_m, target, spec)
        return _finalize(
            self._source_ids,
            distances,
            distance_checks=self.source_count,
            spec=spec,
        )


__all__ = ["ExhaustiveNeighbourhood"]
