"""Explicit three-axis adapter records for GSLIB-style array consumers.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal, NoReturn, cast

from ..errors import GeomodelContractError
from .geo_grid import GeoGrid


def _invalid(field: str, expected: object, actual: object) -> NoReturn:
    raise GeomodelContractError(
        "invalid GSLIB grid layout",
        object_name="GslibGridLayout",
        field=field,
        expected=expected,
        actual=actual,
    )


def _triple(value: object, *, field: str) -> tuple[object, object, object]:
    try:
        result = tuple(cast(Iterable[object], value))
    except TypeError as exc:
        raise GeomodelContractError(
            "invalid GSLIB grid layout",
            object_name="GslibGridLayout",
            field=field,
            expected="sequence of length 3",
            actual=value,
        ) from exc
    if len(result) != 3:
        _invalid(field, "sequence of length 3", value)
    return (result[0], result[1], result[2])


@dataclass(frozen=True, slots=True)
class GslibGridLayout:
    """A named external-layout view that may contain a singleton z axis.

    Attributes:
        source_ndim: dimensionality of the source grid.
        shape / origin_m / spacing_m: the GSLIB grid definition.
    """

    source_ndim: Literal[2, 3]
    shape: tuple[int, int, int]
    origin_m: tuple[float, float, float]
    spacing_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if type(self.source_ndim) is not int or self.source_ndim not in (2, 3):
            _invalid("source_ndim", "2 or 3", self.source_ndim)

        raw_shape = _triple(self.shape, field="shape")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in raw_shape
        ):
            _invalid("shape", "three positive integers", self.shape)
        shape = cast(
            tuple[int, int, int],
            tuple(int(cast(Any, item)) for item in raw_shape),
        )
        if self.source_ndim == 2 and shape[2] != 1:
            _invalid("shape", "singleton third axis for a 2-D source", shape)
        object.__setattr__(self, "shape", shape)

        for field_name in ("origin_m", "spacing_m"):
            raw_values = _triple(getattr(self, field_name), field=field_name)
            try:
                values = tuple(float(cast(Any, item)) for item in raw_values)
            except (TypeError, ValueError, OverflowError) as exc:
                raise GeomodelContractError(
                    "invalid GSLIB grid layout",
                    object_name=type(self).__name__,
                    field=field_name,
                    expected="three finite numbers",
                    actual=raw_values,
                ) from exc
            if not all(math.isfinite(item) for item in values):
                _invalid(field_name, "three finite numbers", values)
            if field_name == "spacing_m" and any(item <= 0.0 for item in values):
                _invalid(field_name, "three positive finite numbers", values)
            object.__setattr__(self, field_name, values)

    def to_dict(self) -> dict[str, object]:
        """Return the adapter layout as strict JSON-native values."""
        return {
            "source_ndim": self.source_ndim,
            "shape": list(self.shape),
            "origin_m": list(self.origin_m),
            "spacing_m": list(self.spacing_m),
        }


def gslib_grid_layout(grid: GeoGrid) -> GslibGridLayout:
    """Create an explicit three-axis layout without mutating a live GeoGrid."""
    if not isinstance(grid, GeoGrid):
        raise GeomodelContractError(
            "GSLIB layout requires a GeoGrid",
            object_name="gslib_grid_layout",
            field="grid",
            expected="GeoGrid",
            actual=type(grid).__name__,
        )
    if grid.ndim == 2:
        return GslibGridLayout(
            source_ndim=2,
            shape=(grid.shape[0], grid.shape[1], 1),
            origin_m=(grid.origin[0], grid.origin[1], 0.0),
            spacing_m=(grid.spacing[0], grid.spacing[1], 1.0),
        )
    return GslibGridLayout(
        source_ndim=3,
        shape=cast(tuple[int, int, int], grid.shape),
        origin_m=cast(tuple[float, float, float], grid.origin),
        spacing_m=cast(tuple[float, float, float], grid.spacing),
    )


__all__ = ["GslibGridLayout", "gslib_grid_layout"]
