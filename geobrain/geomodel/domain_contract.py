"""Canonical dimension-aware metre-domain records and fingerprints.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from .errors import GeomodelContractError

DomainDimension = Literal[2, 3]
StorageOrder = Literal["x-fastest", "points"]


def _invalid(field: str, expected: object, actual: object) -> NoReturn:
    raise GeomodelContractError(
        "invalid Geomodel domain contract",
        object_name="DomainContract",
        field=field,
        expected=expected,
        actual=actual,
    )


def _finite_float_tuple(value: object, *, field: str, ndim: int) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in cast(Iterable[Any], value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeomodelContractError(
            "invalid Geomodel domain contract",
            object_name="DomainContract",
            field=field,
            expected=f"{ndim} finite numbers",
            actual=value,
        ) from exc
    if len(result) != ndim or not all(math.isfinite(item) for item in result):
        _invalid(field, f"{ndim} finite numbers", value)
    return result


@dataclass(frozen=True, slots=True)
class DomainContract:
    """Immutable canonical identity of a point or regular-grid domain.

    Attributes:
        ndim / axis_names: dimensionality and axis labels.
        coordinate_unit: coordinate unit string.
        origin_m / shape / spacing_m: gridded-domain geometry.
        storage_order: array storage order tag.
        fingerprint: content hash identifying the domain.
    """

    ndim: DomainDimension
    axis_names: tuple[str, ...]
    coordinate_unit: Literal["m"]
    origin_m: tuple[float, ...]
    shape: tuple[int, ...] | None
    spacing_m: tuple[float, ...] | None
    storage_order: StorageOrder
    fingerprint: str

    def __post_init__(self) -> None:
        if isinstance(self.ndim, bool) or not isinstance(self.ndim, int) or self.ndim not in (2, 3):
            _invalid("ndim", "2 or 3", self.ndim)
        ndim = int(self.ndim)
        axes = ("x", "y") if ndim == 2 else ("x", "y", "z")
        try:
            supplied_axes = tuple(self.axis_names)
        except TypeError as exc:
            raise GeomodelContractError(
                "invalid Geomodel domain contract",
                object_name="DomainContract",
                field="axis_names",
                expected=axes,
                actual=self.axis_names,
            ) from exc
        if supplied_axes != axes:
            _invalid("axis_names", axes, self.axis_names)
        object.__setattr__(self, "axis_names", axes)
        if self.coordinate_unit != "m":
            _invalid("coordinate_unit", "m", self.coordinate_unit)
        object.__setattr__(
            self,
            "origin_m",
            _finite_float_tuple(self.origin_m, field="origin_m", ndim=ndim),
        )
        if self.storage_order not in ("x-fastest", "points"):
            _invalid("storage_order", "'x-fastest' or 'points'", self.storage_order)

        if self.storage_order == "points":
            if self.shape is not None or self.spacing_m is not None:
                _invalid(
                    "shape",
                    "shape and spacing_m are None for point domains",
                    {"shape": self.shape, "spacing_m": self.spacing_m},
                )
        else:
            if self.shape is None or self.spacing_m is None:
                _invalid(
                    "shape",
                    "shape and spacing_m are present for regular grids",
                    {"shape": self.shape, "spacing_m": self.spacing_m},
                )
            try:
                shape = tuple(self.shape)
            except TypeError as exc:
                raise GeomodelContractError(
                    "invalid Geomodel domain contract",
                    object_name="DomainContract",
                    field="shape",
                    expected=f"{ndim} positive integers",
                    actual=self.shape,
                ) from exc
            if len(shape) != ndim or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape
            ):
                _invalid("shape", f"{ndim} positive integers", self.shape)
            object.__setattr__(self, "shape", shape)
            spacing = _finite_float_tuple(self.spacing_m, field="spacing_m", ndim=ndim)
            if any(item <= 0.0 for item in spacing):
                _invalid("spacing_m", f"{ndim} positive finite numbers", spacing)
            object.__setattr__(self, "spacing_m", spacing)

        fingerprint = self.fingerprint
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            _invalid("fingerprint", "64 lowercase hexadecimal characters", fingerprint)

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-native domain record."""
        return {
            "ndim": self.ndim,
            "axis_names": list(self.axis_names),
            "coordinate_unit": self.coordinate_unit,
            "origin_m": list(self.origin_m),
            "shape": None if self.shape is None else list(self.shape),
            "spacing_m": None if self.spacing_m is None else list(self.spacing_m),
            "storage_order": self.storage_order,
            "fingerprint": self.fingerprint,
        }


def _fingerprint(header: dict[str, object], coordinates: NDArray[np.float64] | None) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )
    digest.update(b"\n")
    if coordinates is not None:
        canonical = np.ascontiguousarray(coordinates, dtype="<f8")
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def domain_contract(domain: object) -> DomainContract:
    """Validate and fingerprint a GeoGrid, GeoPoints, GeoFrame, or coordinate array."""
    from .frames.geo_frame import GeoFrame
    from .frames.geo_grid import GeoGrid
    from .frames.geometry import Geometry

    geometry: object = domain.geometry if isinstance(domain, GeoFrame) else domain
    if isinstance(geometry, GeoGrid):
        geometry._validate_derived_geometry(object_name="domain_contract")
        header: dict[str, object] = {
            "schema": "geobrain.geomodel-domain/1.0",
            "ndim": geometry.ndim,
            "axis_names": list(geometry.axis_names),
            "coordinate_unit": "m",
            "origin_m": list(geometry.origin),
            "shape": list(geometry.shape),
            "spacing_m": list(geometry.spacing),
            "storage_order": "x-fastest",
        }
        return DomainContract(
            ndim=cast(DomainDimension, geometry.ndim),
            axis_names=geometry.axis_names,
            coordinate_unit="m",
            origin_m=geometry.origin,
            shape=geometry.shape,
            spacing_m=geometry.spacing,
            storage_order="x-fastest",
            fingerprint=_fingerprint(header, None),
        )

    coordinates_source = geometry.coords if isinstance(geometry, Geometry) else geometry
    try:
        coordinates = np.array(
            coordinates_source,
            dtype=np.float64,
            copy=True,
            order="C",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeomodelContractError(
            "point-domain coordinates must be numeric",
            object_name="domain_contract",
            field="domain",
            expected="GeoPoints, GeoFrame, or (n, 2|3) array",
            actual=type(domain).__name__,
        ) from exc
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] == 0
        or coordinates.shape[1] not in (2, 3)
        or not np.isfinite(coordinates).all()
    ):
        raise GeomodelContractError(
            "point-domain coordinates are invalid",
            object_name="domain_contract",
            field="domain",
            expected="non-empty finite (n, 2|3) coordinates",
            actual=tuple(coordinates.shape),
        )
    ndim = int(coordinates.shape[1])
    axes = ("x", "y") if ndim == 2 else ("x", "y", "z")
    origin = tuple(float(value) for value in np.min(coordinates, axis=0))
    header = {
        "schema": "geobrain.geomodel-domain/1.0",
        "ndim": ndim,
        "axis_names": list(axes),
        "coordinate_unit": "m",
        "origin_m": list(origin),
        "shape": None,
        "spacing_m": None,
        "storage_order": "points",
        "count": int(coordinates.shape[0]),
        "dtype": "<f8",
        "memory_order": "C",
    }
    return DomainContract(
        ndim=cast(DomainDimension, ndim),
        axis_names=axes,
        coordinate_unit="m",
        origin_m=origin,
        shape=None,
        spacing_m=None,
        storage_order="points",
        fingerprint=_fingerprint(header, coordinates),
    )


__all__ = ["DomainContract", "domain_contract"]
