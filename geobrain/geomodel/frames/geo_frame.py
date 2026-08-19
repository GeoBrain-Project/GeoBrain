"""GeoFrame: geometry + property columns + role tags (numpy).

Key design choices:

- Validation routes through :class:`GeomodelContractError` instead of bare
  ``ValueError`` / ``KeyError``.
- No CRS slot, no pandas interop: both can be added later if
  interchange examples need them.
- Underlying storage stays numpy (`np.ndarray`): see the
  ``feedback-geomodel-numpy-island`` memory.

GeoGrid acceptance: passing a property with the grid's declared shape is
flattened with ``order="F"`` to match
the GSLIB x-fastest layout used by :meth:`GeoGrid.coords`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast

import numpy as np

from ..errors import GeomodelContractError
from ._arrays import BoolArray, FloatArray, as_bool_array, as_float_array
from .geometry import Geometry
from .geo_points import GeoPoints
from .geo_grid import GeoGrid
from .metadata import PropertyMetadata
from .roles import ColumnRole, normalize_role

GridArrayOrder = Literal["C", "F", "A"]


def _readonly_float_copy(values: object) -> FloatArray:
    """Return an owned float64 array backed by immutable bytes."""
    owned = np.array(values, dtype=np.float64, copy=True, order="C")
    return cast(
        FloatArray,
        np.frombuffer(owned.tobytes(order="C"), dtype=np.float64).reshape(owned.shape),
    )


class GeoFrame:
    """
    Spatial table combining geometry, property columns, and role tags.

    Args:
        geometry: a :class:`Geometry` (GeoPoints / GeoGrid).
        properties: mapping ``name -> array-like`` of property columns.
            Each is coerced to ``np.float64``; for a GeoGrid the
            array may use the declared grid shape or flat length ``ncells``.
        roles: optional mapping ``name -> ColumnRole | str``.
        metadata: optional exact metadata for each supplied property.
            Omitted records become explicit dimensionless-continuous
            metadata (the transitional default while remaining callers
            adopt explicit metadata).

    The instance is mutable: ``table["new_col"] = values`` and
    :meth:`set_role` modify in place.
    """

    def __init__(
        self,
        geometry: Geometry,
        properties: Mapping[str, Any] | None = None,
        roles: Mapping[str, ColumnRole | str] | None = None,
        *,
        metadata: Mapping[str, PropertyMetadata] | None = None,
    ) -> None:
        if not isinstance(geometry, Geometry):
            raise GeomodelContractError(
                "GeoFrame requires a Geometry instance",
                object_name="GeoFrame",
                field="geometry",
                expected="Geometry",
                actual=type(geometry).__name__,
            )
        self._geometry: Geometry = geometry
        self._properties: dict[str, FloatArray] = {}
        self._roles: dict[str, ColumnRole] = {}
        self._metadata: dict[str, PropertyMetadata] = {}

        property_values = {} if properties is None else dict(properties)
        metadata_values = {} if metadata is None else dict(metadata)
        unknown_metadata = set(metadata_values) - set(property_values)
        if unknown_metadata:
            raise GeomodelContractError(
                "GeoFrame metadata names an unknown property",
                object_name="GeoFrame",
                field="metadata",
                expected=sorted(property_values),
                actual=sorted(unknown_metadata),
            )
        for name, arr in property_values.items():
            property_metadata = metadata_values.get(
                name,
                PropertyMetadata(name=name, kind="continuous", unit="1"),
            )
            self._set_property(name, arr, property_metadata)
        if roles:
            for name, role in roles.items():
                self.set_role(name, role)

    # ------------------------------------------------------------------
    # Dunder API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.geometry.npoints

    def __contains__(self, key: object) -> bool:
        return key in self._properties

    def __getitem__(self, key: str) -> FloatArray:
        if key not in self._properties:
            raise GeomodelContractError(
                f"column {key!r} is missing",
                object_name="GeoFrame",
                field="key",
                expected=f"one of {list(self._properties)}",
                actual=key,
            )
        return self._properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        metadata = self._metadata.get(
            key,
            PropertyMetadata(name=key, kind="continuous", unit="1"),
        )
        self._set_property(key, value, metadata)

    def __delitem__(self, key: str) -> None:
        if key not in self._properties:
            raise GeomodelContractError(
                f"column {key!r} is missing",
                object_name="GeoFrame",
                field="key",
                expected=f"one of {list(self._properties)}",
                actual=key,
            )
        del self._properties[key]
        self._roles.pop(key, None)
        self._metadata.pop(key, None)

    def __iter__(self) -> Iterable[str]:
        return iter(self._properties)

    # ------------------------------------------------------------------

    @property
    def geometry(self) -> Geometry:
        """Return the geometry whose row count was validated at construction."""
        return self._geometry

    @property
    def columns(self) -> list[str]:
        return list(self._properties.keys())

    @property
    def column_roles(self) -> dict[str, ColumnRole]:
        return dict(self._roles)

    @property
    def property_metadata(self) -> dict[str, PropertyMetadata]:
        """Return an independent mapping of columns to immutable metadata."""
        return dict(self._metadata)

    def metadata_for(self, column: str) -> PropertyMetadata:
        """Return the immutable metadata record for one property column."""
        try:
            return self._metadata[column]
        except KeyError as exc:
            raise GeomodelContractError(
                f"metadata_for: column {column!r} is missing",
                object_name="GeoFrame.metadata_for",
                field="column",
                expected=f"one of {list(self._properties)}",
                actual=column,
            ) from exc

    @property
    def bounds(self) -> tuple[float, ...]:
        return self.geometry.bounds

    @property
    def centroid(self) -> FloatArray:
        return self.geometry.centroid

    # ------------------------------------------------------------------
    # Role management
    # ------------------------------------------------------------------

    def set_role(self, column: str, role: ColumnRole | str | None) -> None:
        if column not in self._properties:
            raise GeomodelContractError(
                f"set_role: column {column!r} is missing",
                object_name="GeoFrame.set_role",
                field="column",
                expected=f"one of {list(self._properties)}",
                actual=column,
            )
        normalized = normalize_role(role)
        if normalized is None:
            self._roles.pop(column, None)
            return
        self._roles[column] = normalized

    def get_role(self, column: str) -> ColumnRole | None:
        return self._roles.get(column)

    def columns_by_role(self, role: ColumnRole | str) -> list[str]:
        normalized = normalize_role(role)
        return [name for name in self.columns if self._roles.get(name) is normalized]

    def clear_role(
        self,
        *,
        column: str | None = None,
        role: ColumnRole | str | None = None,
    ) -> None:
        if column is None and role is None:
            self._roles.clear()
            return
        if column is not None:
            self._roles.pop(column, None)
        if role is not None:
            normalized = normalize_role(role)
            for name in list(self._roles):
                if self._roles[name] is normalized:
                    del self._roles[name]

    # ------------------------------------------------------------------
    # Functional helpers
    # ------------------------------------------------------------------

    def copy(self) -> "GeoFrame":
        return GeoFrame(
            self.geometry,
            self._properties,
            self._roles,
            metadata=self._metadata,
        )

    def select(self, *columns: str) -> "GeoFrame":
        for c in columns:
            if c not in self._properties:
                raise GeomodelContractError(
                    f"select: column {c!r} is missing",
                    object_name="GeoFrame.select",
                    field="column",
                    expected=f"one of {list(self._properties)}",
                    actual=c,
                )
        selected_roles = {c: r for c, r in self._roles.items() if c in columns}
        return GeoFrame(
            self.geometry,
            {c: self._properties[c] for c in columns},
            selected_roles,
            metadata={c: self._metadata[c] for c in columns},
        )

    def where(self, mask: Any) -> "GeoFrame":
        m = self._coerce_mask(mask)
        new_geom = GeoPoints(self.geometry.coords[m])
        return GeoFrame(
            new_geom,
            {name: values[m] for name, values in self._properties.items()},
            self._roles,
            metadata=self._metadata,
        )

    def trim(self, tmin: float, tmax: float, column: str) -> "GeoFrame":
        if column not in self._properties:
            raise GeomodelContractError(
                f"trim: column {column!r} is missing",
                object_name="GeoFrame.trim",
                field="column",
                expected=f"one of {list(self._properties)}",
                actual=column,
            )
        values = self._properties[column]
        return self.where(as_bool_array((values >= tmin) & (values <= tmax)))

    # ------------------------------------------------------------------
    # Grid-array conversion
    # ------------------------------------------------------------------

    def to_numpy(
        self,
        column: str,
        *,
        shaped: bool = False,
        order: GridArrayOrder = "F",
    ) -> FloatArray:
        """Return a property as flat array or (for GeoGrid) shaped."""
        if column not in self._properties:
            raise GeomodelContractError(
                f"to_numpy: column {column!r} is missing",
                object_name="GeoFrame.to_numpy",
                field="column",
                expected=f"one of {list(self._properties)}",
                actual=column,
            )
        if shaped:
            return self.to_grid_array(column, order=order)
        return self._properties[column]

    def to_grid_array(self, column: str, *, order: GridArrayOrder = "F") -> FloatArray:
        """Return a column with its GeoGrid's declared ``(nx, ny[, nz])`` shape."""
        if not isinstance(self.geometry, GeoGrid):
            raise GeomodelContractError(
                "to_grid_array requires a GeoGrid geometry",
                object_name="GeoFrame.to_grid_array",
                field="geometry",
                expected="GeoGrid",
                actual=type(self.geometry).__name__,
            )
        if column not in self._properties:
            raise GeomodelContractError(
                f"to_grid_array: column {column!r} is missing",
                object_name="GeoFrame.to_grid_array",
                field="column",
                expected=f"one of {list(self._properties)}",
                actual=column,
            )
        g = self.geometry
        flat = self._properties[column]
        grid_values = as_float_array(flat.reshape(g.shape, order="F"))
        if order == "F":
            return grid_values
        return as_float_array(np.array(grid_values, dtype=np.float64, order=order, copy=True))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _coerce_property(self, value: Any, name: str) -> FloatArray:
        try:
            arr = as_float_array(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GeomodelContractError(
                f"property {name!r} must be numeric",
                object_name="GeoFrame",
                field=name,
                expected="values coercible to a finite float64 array",
                actual=type(value).__name__,
            ) from exc

        if isinstance(self.geometry, GeoGrid):
            g = self.geometry
            if arr.shape == g.shape:
                return _readonly_float_copy(arr.ravel(order="F"))
            if arr.size == g.ncells:
                return _readonly_float_copy(arr.reshape(-1, order="F"))
            raise GeomodelContractError(
                f"property {name!r} shape does not match GeoGrid",
                object_name="GeoFrame",
                field=name,
                expected=f"{g.shape} or flat length {g.ncells}",
                actual=tuple(arr.shape),
            )

        if arr.ndim != 1:
            raise GeomodelContractError(
                f"property {name!r} must be 1-D for {type(self.geometry).__name__}",
                object_name="GeoFrame",
                field=name,
                expected="1-D array",
                actual=tuple(arr.shape),
            )
        if arr.size != self.geometry.npoints:
            raise GeomodelContractError(
                f"property {name!r} length does not match geometry",
                object_name="GeoFrame",
                field=name,
                expected=f"length {self.geometry.npoints}",
                actual=arr.size,
            )
        return _readonly_float_copy(arr)

    def _set_property(
        self,
        name: str,
        value: Any,
        metadata: PropertyMetadata,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise GeomodelContractError(
                "GeoFrame property name must be a non-empty string",
                object_name="GeoFrame",
                field="properties",
                expected="non-empty string keys",
                actual=name,
            )
        if not isinstance(metadata, PropertyMetadata):
            raise GeomodelContractError(
                "GeoFrame property metadata is invalid",
                object_name="GeoFrame",
                field=f"metadata.{name}",
                expected="PropertyMetadata",
                actual=type(metadata).__name__,
            )
        if metadata.name != name:
            raise GeomodelContractError(
                "GeoFrame property metadata name does not match its column",
                object_name="GeoFrame",
                field=f"metadata.{name}.name",
                expected=name,
                actual=metadata.name,
            )
        values = self._coerce_property(value, name)
        metadata.validate_values(values, object_name="GeoFrame")
        self._properties[name] = values
        self._metadata[name] = metadata

    def _coerce_mask(self, mask: Any) -> BoolArray:
        m = as_bool_array(mask)
        if isinstance(self.geometry, GeoGrid):
            g = self.geometry
            if m.shape == g.shape:
                return as_bool_array(m.ravel(order="F"))
        if m.ndim != 1:
            raise GeomodelContractError(
                "mask must be 1-D",
                object_name="GeoFrame.where",
                field="mask",
                expected="1-D bool array",
                actual=tuple(m.shape),
            )
        if m.size != len(self):
            raise GeomodelContractError(
                "mask length does not match GeoFrame",
                object_name="GeoFrame.where",
                field="mask",
                expected=f"length {len(self)}",
                actual=m.size,
            )
        return m

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cols = ", ".join(self.columns[:5])
        if len(self.columns) > 5:
            cols += f", ... ({len(self.columns)} total)"
        return f"GeoFrame({self.geometry!r}, columns=[{cols}])"


__all__ = ["GeoFrame"]
