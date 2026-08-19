"""Immutable content-addressed cache identities for Geomodel.

The keys in this module digest canonical JSON metadata and owned byte copies.
They never retain caller arrays, mappings, object identities, pointers, or
sampled-value shortcuts.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from .domain_contract import DomainContract
from .frames.metadata import PropertyMetadata
from .errors import GeomodelContractError

if TYPE_CHECKING:
    # ``typing.Self`` is 3.11+; this package still supports 3.10
    # (``requires-python = ">=3.10"``). ``from __future__ import annotations``
    # above keeps every annotation lazy, so a type-checking-only import is
    # enough: the same pattern ``geobrain/bayes/base.py`` uses.
    from typing import Self

FloatArray = np.ndarray[tuple[Any, ...], np.dtype[np.float64]]


def _digest(header: Mapping[str, object], payload: bytes = b"") -> str:
    try:
        encoded = json.dumps(
            dict(header),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise GeomodelContractError(
            "cache-key metadata must be canonical JSON",
            object_name="GeomodelCacheKey",
            field="metadata",
            expected="finite JSON-native values",
            actual=header,
        ) from exc
    hasher = hashlib.sha256()
    hasher.update(encoded)
    hasher.update(b"\n")
    hasher.update(payload)
    return hasher.hexdigest()


def _validate_sha256(value: object, *, object_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GeomodelContractError(
            "cache-key digest must be lowercase SHA-256",
            object_name=object_name,
            field="digest",
            expected="64 lowercase hexadecimal characters",
            actual=value,
        )
    return value


def _canonical_float64_array(
    values: object,
    *,
    object_name: str,
    ndim: int | None = None,
) -> FloatArray:
    try:
        array = np.array(values, dtype="<f8", order="C", copy=True)
    except Exception as exc:
        raise GeomodelContractError(
            "cache-key array must be numeric",
            object_name=object_name,
            field="values",
            expected="finite float64 array",
            actual=type(values).__name__,
        ) from exc
    if (ndim is not None and array.ndim != ndim) or not np.isfinite(array).all():
        raise GeomodelContractError(
            "cache-key array has invalid shape or values",
            object_name=object_name,
            field="values",
            expected=f"finite rank-{ndim} float64 array" if ndim is not None else "finite array",
            actual={"shape": tuple(array.shape), "dtype": str(array.dtype)},
        )
    return cast(FloatArray, array)


def _strict_json_value(
    value: object,
    *,
    object_name: str,
    path: str,
    active: set[int],
) -> object:
    """Own JSON metadata while rejecting key coercions and non-finite values."""
    if isinstance(value, Mapping):
        try:
            items = tuple(value.items())
        except Exception as exc:
            raise GeomodelContractError(
                "cache-key metadata mapping cannot be read",
                object_name=object_name,
                field=path,
                expected="finite JSON mapping with string keys",
                actual=type(value).__name__,
            ) from exc
        invalid_keys = tuple(key for key, _ in items if not isinstance(key, str))
        if invalid_keys:
            raise GeomodelContractError(
                "cache-key metadata keys must be strings",
                object_name=object_name,
                field=path,
                expected="JSON object with string keys",
                actual=invalid_keys[0],
            )
        identity = id(value)
        if identity in active:
            raise GeomodelContractError(
                "cache-key metadata cannot contain a recursive cycle",
                object_name=object_name,
                field=path,
                expected="acyclic finite JSON metadata",
                actual="recursive mapping",
            )
        active.add(identity)
        try:
            return {
                cast(str, key): _strict_json_value(
                    item,
                    object_name=object_name,
                    path=f"{path}.{key}",
                    active=active,
                )
                for key, item in items
            }
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise GeomodelContractError(
                "cache-key metadata cannot contain a recursive cycle",
                object_name=object_name,
                field=path,
                expected="acyclic finite JSON metadata",
                actual="recursive sequence",
            )
        active.add(identity)
        try:
            return [
                _strict_json_value(
                    item,
                    object_name=object_name,
                    path=f"{path}[{index}]",
                    active=active,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(identity)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise GeomodelContractError(
        "cache-key metadata must contain finite JSON-native values",
        object_name=object_name,
        field=path,
        expected="null, string, bool, int, finite float, list, or string-keyed mapping",
        actual=value,
    )


def _training_image_array(values: object, *, object_name: str) -> np.ndarray[Any, Any]:
    try:
        raw = np.asarray(values)
    except Exception as exc:
        raise GeomodelContractError(
            "training-image values cannot be converted to an array",
            object_name=object_name,
            field="values",
            expected="non-empty numeric rank-2 or rank-3 array",
            actual=type(values).__name__,
        ) from exc
    if raw.ndim not in (2, 3) or raw.size == 0 or raw.dtype.kind not in "biuf":
        raise GeomodelContractError(
            "training image has invalid shape or dtype",
            object_name=object_name,
            field="values",
            expected="non-empty numeric rank-2 or rank-3 array",
            actual={"shape": tuple(raw.shape), "dtype": str(raw.dtype)},
        )
    dtype = raw.dtype.newbyteorder("<")
    return cast(
        np.ndarray[Any, Any],
        np.array(raw, dtype=dtype, order="C", copy=True),
    )


def _training_image_mask(
    missing_mask: object | None,
    *,
    shape: tuple[int, ...],
    object_name: str,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    if missing_mask is None:
        return np.zeros(shape, dtype=np.bool_)
    try:
        mask = np.asarray(missing_mask)
    except Exception as exc:
        raise GeomodelContractError(
            "training-image missing mask cannot be converted to an array",
            object_name=object_name,
            field="missing_mask",
            expected=f"bool array with shape {shape}",
            actual=type(missing_mask).__name__,
        ) from exc
    if mask.shape != shape or mask.dtype != np.bool_:
        raise GeomodelContractError(
            "training-image missing mask must match the values",
            object_name=object_name,
            field="missing_mask",
            expected=f"bool array with shape {shape}",
            actual={"shape": tuple(mask.shape), "dtype": str(mask.dtype)},
        )
    return cast(
        np.ndarray[Any, np.dtype[np.bool_]],
        np.array(mask, dtype=np.bool_, order="C", copy=True),
    )


def _validate_training_values(
    values: np.ndarray[Any, Any],
    mask: np.ndarray[Any, np.dtype[np.bool_]],
    property_metadata: PropertyMetadata,
    *,
    object_name: str,
) -> None:
    active = values[~mask]
    if active.dtype.kind == "f" and not np.isfinite(active).all():
        raise GeomodelContractError(
            "training-image values must be finite outside the missing mask",
            object_name=object_name,
            field="values",
            expected="finite unmasked values",
            actual="contains NaN or infinity",
        )
    if property_metadata.kind == "continuous":
        return
    if property_metadata.kind != "categorical":
        raise GeomodelContractError(
            "training-image property kind is unsupported",
            object_name=object_name,
            field="property",
            expected="continuous or categorical PropertyMetadata",
            actual=property_metadata.kind,
        )
    vocabulary = set(property_metadata.category_codes)
    for raw_value in active.tolist():
        value = int(raw_value)
        if raw_value != value or value not in vocabulary:
            raise GeomodelContractError(
                "categorical training image contains a value outside its vocabulary",
                object_name=object_name,
                field=property_metadata.name,
                expected={"vocabulary": list(property_metadata.category_codes)},
                actual=raw_value,
            )


@dataclass(frozen=True, slots=True)
class _DigestKey:
    """Frozen base for exact digest-bearing cache keys."""

    digest: str
    schema: ClassVar[str] = "geobrain.geomodel.cache-key/1.0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "digest",
            _validate_sha256(self.digest, object_name=type(self).__name__),
        )

    def to_dict(self) -> dict[str, str]:
        """Return a strict JSON-safe cache identity."""
        return {"schema": self.schema, "digest": self.digest}


class DomainCacheKey(_DigestKey):
    """Content identity for point or regular-grid domains."""

    schema = "geobrain.geomodel.cache-key/domain/1.0"

    @classmethod
    def from_points(cls, coordinates_m: object, *, coordinate_unit: str) -> Self:
        """Build a key from owned canonical point-coordinate bytes."""
        if coordinate_unit != "m":
            raise GeomodelContractError(
                "Geomodel domain cache keys require metre coordinates",
                object_name=cls.__name__,
                field="coordinate_unit",
                expected="m",
                actual=coordinate_unit,
            )
        coordinates = _canonical_float64_array(
            coordinates_m,
            object_name=cls.__name__,
            ndim=2,
        )
        if coordinates.shape[0] == 0 or coordinates.shape[1] not in (2, 3):
            raise GeomodelContractError(
                "point-domain cache key requires non-empty 2-D or 3-D coordinates",
                object_name=cls.__name__,
                field="coordinates_m",
                expected="non-empty (n, 2|3)",
                actual=tuple(coordinates.shape),
            )
        header: dict[str, object] = {
            "schema": cls.schema,
            "coordinate_unit": "m",
            "shape": list(coordinates.shape),
            "dtype": "<f8",
            "order": "C",
        }
        return cls(_digest(header, coordinates.tobytes(order="C")))

    @classmethod
    def from_domain(cls, domain: DomainContract) -> Self:
        """Build a key from a previously validated domain contract."""
        if not isinstance(domain, DomainContract):
            raise GeomodelContractError(
                "domain cache key requires DomainContract",
                object_name=cls.__name__,
                field="domain",
                expected="DomainContract",
                actual=type(domain).__name__,
            )
        return cls(
            _digest(
                {
                    "schema": cls.schema,
                    "domain": domain.to_dict(),
                }
            )
        )


class CovarianceCacheKey(_DigestKey):
    """Content identity for a covariance model and numerical dtype."""

    schema = "geobrain.geomodel.cache-key/covariance/1.0"

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object], *, dtype: str = "<f8") -> Self:
        if not isinstance(parameters, Mapping):
            raise GeomodelContractError(
                "covariance cache parameters must be a mapping",
                object_name=cls.__name__,
                field="parameters",
                expected="JSON mapping",
                actual=type(parameters).__name__,
            )
        canonical_parameters = _strict_json_value(
            parameters,
            object_name=cls.__name__,
            path="parameters",
            active=set(),
        )
        try:
            canonical_dtype = np.dtype(dtype).newbyteorder("<").str
        except (TypeError, ValueError) as exc:
            raise GeomodelContractError(
                "covariance cache dtype is invalid",
                object_name=cls.__name__,
                field="dtype",
                expected="NumPy dtype string",
                actual=dtype,
            ) from exc
        return cls(
            _digest(
                {
                    "schema": cls.schema,
                    "dtype": canonical_dtype,
                    "parameters": canonical_parameters,
                }
            )
        )


class NeighbourhoodCacheKey(_DigestKey):
    """Content identity for a neighbourhood selection policy."""

    schema = "geobrain.geomodel.cache-key/neighbourhood/1.0"

    @classmethod
    def from_spec(cls, spec: object) -> Self:
        from .neighbourhood.contracts import NeighbourhoodSpec

        if not isinstance(spec, NeighbourhoodSpec):
            raise GeomodelContractError(
                "neighbourhood cache key requires NeighbourhoodSpec",
                object_name=cls.__name__,
                field="spec",
                expected="NeighbourhoodSpec",
                actual=type(spec).__name__,
            )
        return cls(_digest({"schema": cls.schema, "spec": spec.to_dict()}))


class TrainingImageCacheKey(_DigestKey):
    """Content identity for a complete training-image array."""

    schema = "geobrain.geomodel.cache-key/training-image/1.0"

    @classmethod
    def from_array(
        cls,
        values: object,
        *,
        property: PropertyMetadata,
        axis_names: tuple[str, ...],
        missing_mask: object | None = None,
    ) -> Self:
        if not isinstance(property, PropertyMetadata):
            raise GeomodelContractError(
                "training-image cache key requires PropertyMetadata",
                object_name=cls.__name__,
                field="property",
                expected="PropertyMetadata",
                actual=type(property).__name__,
            )
        array = _training_image_array(values, object_name=cls.__name__)
        try:
            axes = tuple(axis_names)
        except Exception as exc:
            raise GeomodelContractError(
                "training-image axes must be iterable",
                object_name=cls.__name__,
                field="axis_names",
                expected=f"{array.ndim} unique axis names",
                actual=axis_names,
            ) from exc
        if (
            len(axes) != array.ndim
            or any(not isinstance(axis, str) or not axis.strip() for axis in axes)
            or len(set(axes)) != len(axes)
            or set(axes) != set(("x", "y") if array.ndim == 2 else ("x", "y", "z"))
        ):
            raise GeomodelContractError(
                "training-image axes must match the declared array rank",
                object_name=cls.__name__,
                field="axis_names",
                expected=("x", "y") if array.ndim == 2 else ("x", "y", "z"),
                actual=axes,
            )
        mask = _training_image_mask(
            missing_mask,
            shape=tuple(array.shape),
            object_name=cls.__name__,
        )
        _validate_training_values(array, mask, property, object_name=cls.__name__)
        header: dict[str, object] = {
            "schema": cls.schema,
            "property": property.to_dict(),
            "axis_names": list(axes),
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "order": "C",
            "values_nbytes": int(array.nbytes),
            "missing_mask_dtype": "|b1",
            "missing_mask_order": "C",
            "missing_mask_nbytes": int(mask.nbytes),
        }
        payload = array.tobytes(order="C") + mask.tobytes(order="C")
        return cls(_digest(header, payload))


class CheckpointCacheKey(_DigestKey):
    """Identity for a checksum-verified model artifact."""

    schema = "geobrain.geomodel.cache-key/checkpoint/1.0"

    @classmethod
    def from_sha256(cls, sha256: str) -> Self:
        checksum = _validate_sha256(sha256, object_name=cls.__name__)
        return cls(_digest({"schema": cls.schema, "artifact_sha256": checksum}))


__all__ = [
    "CheckpointCacheKey",
    "CovarianceCacheKey",
    "DomainCacheKey",
    "NeighbourhoodCacheKey",
    "TrainingImageCacheKey",
]
