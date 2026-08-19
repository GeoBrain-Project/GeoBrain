"""Owned hard/soft conditioning with explicit outside and collision policies.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import copy
import builtins
import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .frames.geo_frame import GeoFrame
from .frames.geo_grid import GeoGrid
from .frames.geometry import Geometry, _canonical_finite_mean
from .frames.metadata import PropertyMetadata
from .domain_contract import domain_contract
from .errors import GeomodelContractError

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
ObservationArray: TypeAlias = FloatArray | np.ma.MaskedArray


class _ReadonlyMaskedArray(np.ma.MaskedArray):
    """Masked float array whose public mask cannot be detached for mutation."""

    def unshare_mask(self) -> "_ReadonlyMaskedArray":
        raise ValueError("conditioning observation mask is read-only")


def _owned_float_array(value: object, *, field_name: str) -> FloatArray:
    try:
        result = np.array(value, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeomodelContractError(
            "conditioning values must be numeric",
            object_name="ConditioningSet",
            field=field_name,
            expected="finite float64 array",
            actual=type(value).__name__,
        ) from exc
    if not np.isfinite(result).all():
        raise GeomodelContractError(
            "conditioning values must be finite",
            object_name="ConditioningSet",
            field=field_name,
            expected="finite float64 array",
            actual="contains NaN or infinity",
        )
    # A writeable flag on an owning ndarray can be restored by callers. Back
    # the public array with immutable bytes so the read-only contract also
    # holds for derived views and cannot be bypassed through ``setflags``.
    readonly = np.frombuffer(result.tobytes(order="C"), dtype=np.float64).reshape(result.shape)
    return cast(FloatArray, readonly)


def _owned_observation_array(value: object, *, field_name: str) -> ObservationArray:
    """Own finite observations while preserving an explicit missing-value mask."""
    try:
        masked: np.ma.MaskedArray = np.ma.asarray(value, dtype=np.float64)
        data = np.array(np.ma.getdata(masked), dtype=np.float64, copy=True, order="C")
        mask = np.array(np.ma.getmaskarray(masked), dtype=np.bool_, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeomodelContractError(
            "conditioning values must be numeric",
            object_name="ConditioningSet",
            field=field_name,
            expected="finite float64 array with an optional boolean mask",
            actual=type(value).__name__,
        ) from exc
    if not np.isfinite(data[~mask]).all():
        raise GeomodelContractError(
            "conditioning values must be finite where observations are present",
            object_name="ConditioningSet",
            field=field_name,
            expected="finite unmasked float64 values",
            actual="contains unmasked NaN or infinity",
        )
    readonly_data = np.frombuffer(data.tobytes(order="C"), dtype=np.float64).reshape(data.shape)
    if not np.any(mask):
        return cast(FloatArray, readonly_data)
    readonly_mask = np.frombuffer(mask.tobytes(order="C"), dtype=np.bool_).reshape(mask.shape)
    return _ReadonlyMaskedArray(readonly_data, mask=readonly_mask, copy=False)


def _working_observation_array(value: ObservationArray) -> ObservationArray:
    """Return a mutable normalization copy without dropping a missing-value mask."""
    if isinstance(value, np.ma.MaskedArray):
        return np.ma.array(
            np.ma.getdata(value),
            mask=np.ma.getmaskarray(value),
            dtype=np.float64,
            copy=True,
        )
    return np.array(value, dtype=np.float64, copy=True, order="C")


def _overflow_safe_mean(values: FloatArray, *, axis: int) -> FloatArray | np.float64:
    """Compute one canonical finite mean without overflow or order drift."""
    return _canonical_finite_mean(
        values,
        axis=axis,
        object_name="ConditioningSet",
        field="observations",
    )


def _canonical_precision_mean(
    values: FloatArray,
    precision: FloatArray,
) -> tuple[FloatArray, float]:
    """Combine soft rows in one exact, permutation-independent order."""
    order = sorted(
        range(int(values.shape[0])),
        key=lambda index: (
            tuple(0.0 if float(value) == 0.0 else float(value) for value in values[index]),
            float(precision[index]),
        ),
    )
    ordered_precision = [float(precision[index]) for index in order]
    try:
        total_precision = math.fsum(ordered_precision)
    except OverflowError as exc:
        raise GeomodelContractError(
            "combined soft precision is not finite",
            object_name="normalize_conditioning",
            field="soft_precision",
            expected="finite summed precision",
            actual=ordered_precision,
        ) from exc
    if not math.isfinite(total_precision):
        raise GeomodelContractError(
            "combined soft precision is not finite",
            object_name="normalize_conditioning",
            field="soft_precision",
            expected="finite summed precision",
            actual=ordered_precision,
        )

    maximum_precision = max(ordered_precision)
    scaled_precision = [value / maximum_precision for value in ordered_precision]
    scaled_total = math.fsum(scaled_precision)
    combined = np.empty(values.shape[1], dtype=np.float64)
    for column in range(int(values.shape[1])):
        numerator = math.fsum(
            float(values[index, column]) * weight
            for index, weight in zip(order, scaled_precision, strict=True)
        )
        probability = numerator / scaled_total
        if not math.isfinite(probability):
            raise GeomodelContractError(
                "combined soft probability is not finite",
                object_name="normalize_conditioning",
                field="soft_probabilities",
                expected="finite precision-weighted probability",
                actual=str(probability),
            )
        combined[column] = probability
    return cast(FloatArray, combined), total_precision


def _presence_mask(value: ObservationArray | None, *, rows: int) -> NDArray[np.bool_]:
    """Return which aligned observation rows contain real values."""
    if value is None:
        return np.zeros(rows, dtype=np.bool_)
    mask = np.ma.getmaskarray(value)
    if value.ndim == 1:
        return ~mask
    return cast(NDArray[np.bool_], ~np.all(mask, axis=1))


@dataclass(frozen=True, slots=True)
class ConditioningPolicy:
    """Explicit rules for outside observations and coincident soft data.

    Attributes:
        outside: policy for data outside the domain.
        soft_collision: policy when hard and soft data collide.
        coordinate_tolerance_m: coincidence tolerance [m].
        value_tolerance: duplicate-value tolerance.
    """

    outside: Literal["error", "discard"] = "error"
    soft_collision: Literal["error", "precision_mean"] = "error"
    coordinate_tolerance_m: float = 1.0e-10
    value_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.outside not in ("error", "discard"):
            raise GeomodelContractError(
                "invalid conditioning outside policy",
                object_name=type(self).__name__,
                field="outside",
                expected="'error' or 'discard'",
                actual=self.outside,
            )
        if self.soft_collision not in ("error", "precision_mean"):
            raise GeomodelContractError(
                "invalid soft-conditioning collision policy",
                object_name=type(self).__name__,
                field="soft_collision",
                expected="'error' or 'precision_mean'",
                actual=self.soft_collision,
            )
        for field_name in ("coordinate_tolerance_m", "value_tolerance"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise GeomodelContractError(
                    "conditioning tolerance must be numeric",
                    object_name=type(self).__name__,
                    field=field_name,
                    expected="finite non-negative number",
                    actual=value,
                )
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < 0.0:
                raise GeomodelContractError(
                    "conditioning tolerance must be finite and non-negative",
                    object_name=type(self).__name__,
                    field=field_name,
                    expected=">= 0 and finite",
                    actual=value,
                )
            object.__setattr__(self, field_name, normalized)

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-native policy record."""
        return {
            "outside": self.outside,
            "soft_collision": self.soft_collision,
            "coordinate_tolerance_m": self.coordinate_tolerance_m,
            "value_tolerance": self.value_tolerance,
        }


@dataclass(frozen=True, slots=True)
class ConditioningSet:
    """Owned hard data and categorical probabilities with optional precision.

    Attributes:
        coordinates_m: ``(n, dim)`` data locations [m].
        hard_values: exact conditioning values.
        soft_probabilities: probabilistic conditioning table.
        property: the conditioned :class:`PropertyMetadata`.
        policy: the :class:`ConditioningPolicy` applied.
        soft_precision: precision weight of soft rows.
    """

    coordinates_m: FloatArray
    hard_values: ObservationArray | None
    soft_probabilities: ObservationArray | None
    property: PropertyMetadata
    policy: ConditioningPolicy = field(default_factory=ConditioningPolicy)
    soft_precision: ObservationArray | None = None
    _diagnostics: dict[str, object] = field(
        init=False,
        repr=False,
        compare=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        coordinates = _owned_float_array(self.coordinates_m, field_name="coordinates_m")
        if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
            raise GeomodelContractError(
                "conditioning coordinates have an invalid dimension",
                object_name=type(self).__name__,
                field="coordinates_m",
                expected="(n, 2) or (n, 3)",
                actual=tuple(coordinates.shape),
            )
        object.__setattr__(self, "coordinates_m", coordinates)
        if not isinstance(self.property, PropertyMetadata):
            raise GeomodelContractError(
                "conditioning property metadata is invalid",
                object_name=type(self).__name__,
                field="property",
                expected="PropertyMetadata",
                actual=type(self.property).__name__,
            )
        if not isinstance(self.policy, ConditioningPolicy):
            raise GeomodelContractError(
                "conditioning policy is invalid",
                object_name=type(self).__name__,
                field="policy",
                expected="ConditioningPolicy",
                actual=type(self.policy).__name__,
            )

        count = int(coordinates.shape[0])
        hard: ObservationArray | None = None
        if self.hard_values is not None:
            hard = _owned_observation_array(self.hard_values, field_name="hard_values")
            if hard.ndim != 1 or hard.shape[0] != count:
                raise GeomodelContractError(
                    "hard conditioning values do not match coordinates",
                    object_name=type(self).__name__,
                    field="hard_values",
                    expected=f"shape ({count},)",
                    actual=tuple(hard.shape),
                )
            hard_present = _presence_mask(hard, rows=count)
            self.property.validate_values(
                np.asarray(np.ma.getdata(hard)[hard_present], dtype=np.float64),
                object_name=type(self).__name__,
            )
        object.__setattr__(self, "hard_values", hard)

        soft: ObservationArray | None = None
        if self.soft_probabilities is not None:
            if self.property.kind != "categorical":
                raise GeomodelContractError(
                    "soft conditioning requires a complete categorical support vocabulary",
                    object_name=type(self).__name__,
                    field="soft_probabilities",
                    expected="categorical property with one probability per declared category",
                    actual=self.property.kind,
                )
            soft = _owned_observation_array(
                self.soft_probabilities,
                field_name="soft_probabilities",
            )
            if soft.ndim == 1:
                soft = cast(ObservationArray, soft.reshape(-1, 1))
            if soft.ndim != 2 or soft.shape[0] != count:
                raise GeomodelContractError(
                    "soft conditioning probabilities do not match coordinates",
                    object_name=type(self).__name__,
                    field="soft_probabilities",
                    expected=f"shape ({count}, k)",
                    actual=tuple(soft.shape),
                )
            soft_mask = np.ma.getmaskarray(soft)
            row_all_missing = np.all(soft_mask, axis=1)
            row_all_present = ~np.any(soft_mask, axis=1)
            if not np.all(row_all_missing | row_all_present):
                raise GeomodelContractError(
                    "soft conditioning probabilities must use complete rows",
                    object_name=type(self).__name__,
                    field="soft_probabilities",
                    expected="every row is entirely present or entirely missing",
                    actual=soft_mask.tolist(),
                )
            soft_data = np.asarray(np.ma.getdata(soft), dtype=np.float64)
            soft_present = row_all_present
            present_values = soft_data[soft_present]
            if np.any((present_values < 0.0) | (present_values > 1.0)):
                raise GeomodelContractError(
                    "soft conditioning probabilities must lie in [0, 1]",
                    object_name=type(self).__name__,
                    field="soft_probabilities",
                    expected="all values in [0, 1]",
                    actual="outside [0, 1]",
                )
            if self.property.kind == "categorical":
                category_count = len(self.property.categories)
                if soft.shape[1] != category_count:
                    raise GeomodelContractError(
                        "soft categorical probabilities do not match the vocabulary",
                        object_name=type(self).__name__,
                        field="soft_probabilities",
                        expected=f"{category_count} columns",
                        actual=int(soft.shape[1]),
                    )
                if not np.allclose(
                    np.sum(present_values, axis=1),
                    1.0,
                    rtol=0.0,
                    atol=self.policy.value_tolerance,
                ):
                    raise GeomodelContractError(
                        "soft categorical probabilities must sum to one",
                        object_name=type(self).__name__,
                        field="soft_probabilities",
                        expected="row sum == 1",
                        actual=np.sum(present_values, axis=1).tolist(),
                    )
        object.__setattr__(self, "soft_probabilities", soft)

        precision: ObservationArray | None = None
        if self.soft_precision is not None:
            if soft is None:
                raise GeomodelContractError(
                    "soft precision requires soft probabilities",
                    object_name=type(self).__name__,
                    field="soft_precision",
                    expected="one positive precision for every present soft row",
                    actual="soft_probabilities is None",
                )
            precision = _owned_observation_array(
                self.soft_precision,
                field_name="soft_precision",
            )
            if precision.ndim != 1 or precision.shape[0] != count:
                raise GeomodelContractError(
                    "soft precision does not match conditioning coordinates",
                    object_name=type(self).__name__,
                    field="soft_precision",
                    expected=f"shape ({count},)",
                    actual=tuple(precision.shape),
                )
            precision_present = _presence_mask(precision, rows=count)
            soft_present = _presence_mask(soft, rows=count)
            if not np.array_equal(precision_present, soft_present):
                raise GeomodelContractError(
                    "soft precision presence must match soft probability rows",
                    object_name=type(self).__name__,
                    field="soft_precision",
                    expected={"present_rows": np.flatnonzero(soft_present).astype(int).tolist()},
                    actual={"present_rows": np.flatnonzero(precision_present).astype(int).tolist()},
                )
            precision_values = np.asarray(
                np.ma.getdata(precision)[precision_present],
                dtype=np.float64,
            )
            if np.any(precision_values <= 0.0):
                raise GeomodelContractError(
                    "soft precision must be strictly positive",
                    object_name=type(self).__name__,
                    field="soft_precision",
                    expected="finite values > 0 for every present soft row",
                    actual=precision_values.tolist(),
                )
        object.__setattr__(self, "soft_precision", precision)

        if count > 0 and hard is None and soft is None:
            raise GeomodelContractError(
                "conditioning coordinates have no hard or soft observations",
                object_name=type(self).__name__,
                field="hard_values",
                expected="hard_values or soft_probabilities",
                actual=None,
            )
        hard_present = _presence_mask(hard, rows=count)
        soft_present = _presence_mask(soft, rows=count)
        if count > 0 and not np.all(hard_present | soft_present):
            missing: list[int] = np.flatnonzero(~(hard_present | soft_present)).astype(int).tolist()
            raise GeomodelContractError(
                "conditioning coordinates must carry a hard or soft observation",
                object_name=type(self).__name__,
                field="coordinates_m",
                expected="at least one present observation per coordinate",
                actual={"missing_indices": missing},
            )
        object.__setattr__(self, "_diagnostics", {"input_count": count})

    @builtins.property
    def diagnostics(self) -> dict[str, object]:
        """Return an independent JSON-native normalization diagnosis."""
        return copy.deepcopy(self._diagnostics)

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-native conditioning record."""
        return {
            "coordinates_m": self.coordinates_m.tolist(),
            "hard_values": None if self.hard_values is None else self.hard_values.tolist(),
            "soft_probabilities": (
                None if self.soft_probabilities is None else self.soft_probabilities.tolist()
            ),
            "soft_precision": (
                None if self.soft_precision is None else self.soft_precision.tolist()
            ),
            "property": self.property.to_dict(),
            "policy": self.policy.to_dict(),
            "diagnostics": self.diagnostics,
        }


def _grid_membership(
    coordinates: FloatArray,
    grid: GeoGrid,
) -> tuple[NDArray[np.bool_], IntArray]:
    lower = np.asarray(grid.origin, dtype=np.float64)
    spacing = np.asarray(grid.spacing, dtype=np.float64)
    upper = lower + np.asarray(grid.shape, dtype=np.float64) * spacing
    valid = ~np.any((coordinates < lower) | (coordinates >= upper), axis=1)
    indices: IntArray = np.full(coordinates.shape, -1, dtype=np.int64)
    if np.any(valid):
        indices[valid] = np.floor((coordinates[valid] - lower) / spacing).astype(np.int64)
    return valid, indices


def _property_is_compatible(actual: PropertyMetadata, expected: PropertyMetadata) -> bool:
    return bool(actual == expected)


def _normalization_policy(
    data: ConditioningSet,
    requested: ConditioningPolicy,
) -> ConditioningPolicy:
    default = ConditioningPolicy()
    if requested == default and data.policy != default:
        return data.policy
    return requested


def _nonnegative_sum_within(left: float, right: float, limit: float) -> bool:
    """Compare ``left + right <= limit`` exactly without forming the sum."""
    left_numerator, left_denominator = left.as_integer_ratio()
    right_numerator, right_denominator = right.as_integer_ratio()
    limit_numerator, limit_denominator = limit.as_integer_ratio()
    return (
        left_numerator * right_denominator + right_numerator * left_denominator
    ) * limit_denominator <= limit_numerator * left_denominator * right_denominator


def _axis_distance_within(left: float, right: float, tolerance: float) -> bool:
    """Test one finite coordinate distance without overflowing subtraction."""
    if (left < 0.0) == (right < 0.0):
        return abs(left - right) <= tolerance
    return _nonnegative_sum_within(abs(left), abs(right), tolerance)


def _coordinates_within(
    left: FloatArray,
    right: FloatArray,
    tolerance: float,
) -> bool:
    """Return whether two finite points satisfy the inclusive L-infinity bound."""
    return all(
        _axis_distance_within(float(left_value), float(right_value), tolerance)
        for left_value, right_value in zip(left, right)
    )


def _sorted_groups(coordinates: FloatArray, tolerance: float) -> list[NDArray[np.int64]]:
    if coordinates.shape[0] == 0:
        return []
    keys = tuple(coordinates[:, axis] for axis in reversed(range(coordinates.shape[1])))
    order = np.lexsort(keys)
    parent = np.arange(coordinates.shape[0], dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    def bucket_key(index: int) -> tuple[int | float, ...]:
        row = coordinates[index]
        if tolerance == 0.0:
            return tuple(float(value) for value in row)

        def floor_ratio(value: float) -> int:
            scaled = value / tolerance
            if math.isfinite(scaled):
                return math.floor(scaled)
            value_numerator, value_denominator = value.as_integer_ratio()
            tolerance_numerator, tolerance_denominator = tolerance.as_integer_ratio()
            return (value_numerator * tolerance_denominator) // (
                value_denominator * tolerance_numerator
            )

        return tuple(floor_ratio(float(value)) for value in row)

    buckets: dict[tuple[int | float, ...], list[int]] = {}
    for original_index in order.tolist():
        buckets.setdefault(bucket_key(original_index), []).append(original_index)

    sorted_keys = sorted(buckets)
    for candidates in buckets.values():
        for candidate in candidates[1:]:
            union(candidates[0], candidate)

    if tolerance > 0.0:
        from scipy.spatial import cKDTree  # type: ignore[import-untyped]

        offsets = tuple(product((-1, 0, 1), repeat=coordinates.shape[1]))
        trees: dict[tuple[int | float, ...], Any] = {}
        maximum_float = float(np.finfo(np.float64).max)
        inclusive_radius = (
            math.inf if tolerance == maximum_float else float(np.nextafter(tolerance, math.inf))
        )
        for key in sorted_keys:
            key_ids = np.asarray(buckets[key], dtype=np.int64)
            for offset in offsets:
                if not any(offset):
                    continue
                neighbour_key = tuple(int(key[axis]) + offset[axis] for axis in range(len(key)))
                if neighbour_key <= key or neighbour_key not in buckets:
                    continue
                neighbour_ids = np.asarray(buckets[neighbour_key], dtype=np.int64)
                pair_count = int(key_ids.size * neighbour_ids.size)
                if pair_count <= 4096:
                    connected = any(
                        _coordinates_within(
                            coordinates[left_index],
                            coordinates[right_index],
                            tolerance,
                        )
                        for left_index in key_ids.tolist()
                        for right_index in neighbour_ids.tolist()
                    )
                else:
                    tree = trees.get(key)
                    if tree is None:
                        tree = cKDTree(coordinates[key_ids])
                        trees[key] = tree
                    connected = False
                    for neighbour_id in neighbour_ids.tolist():
                        candidate_indices = (
                            range(int(key_ids.size))
                            if math.isinf(inclusive_radius)
                            else tree.query_ball_point(
                                coordinates[neighbour_id],
                                r=inclusive_radius,
                                p=np.inf,
                                workers=1,
                                return_sorted=True,
                            )
                        )
                        if any(
                            _coordinates_within(
                                coordinates[key_ids[int(candidate_index)]],
                                coordinates[neighbour_id],
                                tolerance,
                            )
                            for candidate_index in candidate_indices
                        ):
                            connected = True
                            break
                if connected:
                    union(buckets[key][0], buckets[neighbour_key][0])

    grouped: dict[int, list[int]] = {}
    for original_index in order.tolist():
        grouped.setdefault(find(original_index), []).append(original_index)
    return [np.asarray(group, dtype=np.int64) for group in grouped.values()]


def normalize_conditioning(
    data: GeoFrame | ConditioningSet | None,
    domain: Geometry | GeoFrame | np.ndarray,
    property: PropertyMetadata,
    policy: ConditioningPolicy = ConditioningPolicy(),
) -> ConditioningSet:
    """Normalize hard/soft data without silent clipping, discard, or collisions.

    Args:
        data: conditioning input (frame / set / ``None``).
        domain: simulation domain the data must fall inside.
        property: the conditioned property.
        policy: conflict-handling policy.
    """
    if not isinstance(property, PropertyMetadata):
        raise GeomodelContractError(
            "conditioning property is invalid",
            object_name="normalize_conditioning",
            field="property",
            expected="PropertyMetadata",
            actual=type(property).__name__,
        )
    geometry: object = domain.geometry if isinstance(domain, GeoFrame) else domain
    if isinstance(geometry, Geometry):
        ndim = geometry.ndim
    else:
        ndim = domain_contract(geometry).ndim
    if ndim not in (2, 3):
        raise GeomodelContractError(
            "conditioning domain has an invalid dimension",
            object_name="normalize_conditioning",
            field="domain",
            expected="2-D or 3-D geometry",
            actual=ndim,
        )

    if data is None:
        result = ConditioningSet(
            np.empty((0, ndim), dtype=np.float64),
            None,
            None,
            property,
            policy,
        )
        object.__setattr__(
            result,
            "_diagnostics",
            {
                "input_count": 0,
                "output_count": 0,
                "discarded_count": 0,
                "discarded_indices": [],
                "discarded_coordinates_m": [],
                "deduplicated_count": 0,
                "combined_soft_count": 0,
                "superseded_soft_count": 0,
                "superseded_soft_indices": [],
                "cell_indices": [],
            },
        )
        return result

    if isinstance(data, GeoFrame):
        if property.name not in data:
            raise GeomodelContractError(
                "conditioning property is missing from GeoFrame",
                object_name="normalize_conditioning",
                field="property",
                expected=property.name,
                actual=data.columns,
            )
        if not _property_is_compatible(data.metadata_for(property.name), property):
            raise GeomodelContractError(
                "conditioning property metadata does not match the request",
                object_name="normalize_conditioning",
                field="property",
                expected=property.to_dict(),
                actual=data.metadata_for(property.name).to_dict(),
            )
        source = ConditioningSet(
            data.geometry.coords,
            data[property.name],
            None,
            property,
            policy,
        )
    elif isinstance(data, ConditioningSet):
        if not _property_is_compatible(data.property, property):
            raise GeomodelContractError(
                "conditioning property does not match the request",
                object_name="normalize_conditioning",
                field="property",
                expected=property.to_dict(),
                actual=data.property.to_dict(),
            )
        source = data
        policy = _normalization_policy(data, policy)
    else:
        raise GeomodelContractError(
            "conditioning data has an unsupported type",
            object_name="normalize_conditioning",
            field="data",
            expected="GeoFrame, ConditioningSet, or None",
            actual=type(data).__name__,
        )

    coordinates = np.array(source.coordinates_m, dtype=np.float64, copy=True, order="C")
    if coordinates.shape[1] != ndim:
        raise GeomodelContractError(
            "conditioning coordinate dimension does not match the domain",
            object_name="normalize_conditioning",
            field="coordinates_m",
            expected=ndim,
            actual=int(coordinates.shape[1]),
        )
    hard = None if source.hard_values is None else _working_observation_array(source.hard_values)
    soft = (
        None
        if source.soft_probabilities is None
        else _working_observation_array(source.soft_probabilities)
    )
    precision = (
        None if source.soft_precision is None else _working_observation_array(source.soft_precision)
    )
    input_count = int(coordinates.shape[0])
    source_indices: IntArray = cast(
        IntArray,
        np.arange(input_count, dtype=np.int64),
    )
    discarded_indices: list[int] = []
    discarded_coordinates: list[list[float]] = []
    cell_indices: IntArray | None = None
    if isinstance(geometry, GeoGrid):
        valid, all_cell_indices = _grid_membership(coordinates, geometry)
        if not np.all(valid):
            first = int(np.flatnonzero(~valid)[0])
            if policy.outside == "error":
                raise GeomodelContractError(
                    "conditioning coordinate is outside the half-open grid extent",
                    object_name="normalize_conditioning",
                    field="coordinates_m",
                    expected={
                        "lower": list(geometry.origin),
                        "upper_exclusive": list(geometry.bounds[geometry.ndim :]),
                    },
                    actual=coordinates[first].tolist(),
                )
            discarded_indices = np.flatnonzero(~valid).astype(int).tolist()
            discarded_coordinates = coordinates[~valid].tolist()
            coordinates = coordinates[valid]
            hard = None if hard is None else hard[valid]
            soft = None if soft is None else soft[valid]
            precision = None if precision is None else precision[valid]
            source_indices = source_indices[valid]
            all_cell_indices = all_cell_indices[valid]
        cell_indices = all_cell_indices

    groups = _sorted_groups(coordinates, policy.coordinate_tolerance_m)
    output_coordinates: list[FloatArray] = []
    output_hard: list[float] = []
    output_hard_present: list[bool] = []
    output_soft: list[FloatArray] = []
    output_soft_present: list[bool] = []
    output_precision: list[float] = []
    output_precision_present: list[bool] = []
    output_cells: list[IntArray] = []
    combined_soft_count = 0
    superseded_soft_indices: list[int] = []
    for group in groups:
        output_coordinates.append(cast(FloatArray, _overflow_safe_mean(coordinates[group], axis=0)))
        if cell_indices is not None:
            if not np.all(cell_indices[group] == cell_indices[group[0]]):
                raise GeomodelContractError(
                    "coincident conditioning tolerance crosses grid-cell boundaries",
                    object_name="normalize_conditioning",
                    field="coordinate_tolerance_m",
                    expected="each collision group maps to one cell",
                    actual=cell_indices[group].tolist(),
                )
            output_cells.append(cell_indices[group[0]])
        group_hard: float | None = None
        if hard is not None:
            hard_present = _presence_mask(hard, rows=hard.shape[0])
            hard_group = group[hard_present[group]]
            if hard_group.size > 0:
                hard_values = np.asarray(np.ma.getdata(hard)[hard_group], dtype=np.float64)
                hard_conflicts = (
                    not np.all(hard_values == hard_values[0])
                    if property.kind == "categorical"
                    else not math.isclose(
                        float(np.min(hard_values)),
                        float(np.max(hard_values)),
                        rel_tol=0.0,
                        abs_tol=policy.value_tolerance,
                    )
                )
                if hard_conflicts:
                    raise GeomodelContractError(
                        "conflicting hard conditioning values at coincident coordinates",
                        object_name="normalize_conditioning",
                        field="hard_values",
                        expected=(
                            "identical categorical codes"
                            if property.kind == "categorical"
                            else f"difference <= {policy.value_tolerance}"
                        ),
                        actual=hard_values.tolist(),
                    )
                group_hard = (
                    float(hard_values[0])
                    if property.kind == "categorical"
                    else float(_overflow_safe_mean(hard_values, axis=0))
                )
        output_hard.append(0.0 if group_hard is None else group_hard)
        output_hard_present.append(group_hard is not None)

        group_soft: FloatArray | None = None
        group_precision: float | None = None
        soft_group: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        if soft is not None:
            soft_present = _presence_mask(soft, rows=soft.shape[0])
            soft_group = group[soft_present[group]]
            if soft_group.size > 0:
                if soft_group.size > 1 and policy.soft_collision == "error":
                    raise GeomodelContractError(
                        "coincident soft observations require an explicit combination policy",
                        object_name="normalize_conditioning",
                        field="soft_collision",
                        expected="precision_mean",
                        actual="error",
                    )
                soft_values = np.asarray(
                    np.ma.getdata(soft)[soft_group],
                    dtype=np.float64,
                )
                if soft_group.size > 1:
                    if precision is None:
                        raise GeomodelContractError(
                            "precision-weighted soft combination requires explicit precisions",
                            object_name="normalize_conditioning",
                            field="soft_precision",
                            expected="one positive precision for every colliding soft row",
                            actual=None,
                        )
                    precision_values = np.asarray(
                        np.ma.getdata(precision)[soft_group],
                        dtype=np.float64,
                    )
                    group_soft, group_precision = _canonical_precision_mean(
                        soft_values,
                        precision_values,
                    )
                else:
                    group_soft = cast(FloatArray, soft_values[0])
                    if precision is not None:
                        group_precision = float(np.ma.getdata(precision)[soft_group[0]])
                if soft_group.size > 1:
                    combined_soft_count += int(soft_group.size) - 1
                if group_hard is not None:
                    if property.kind == "categorical":
                        code_to_column = {
                            category.code: column
                            for column, category in enumerate(property.categories)
                        }
                        column = code_to_column[int(group_hard)]
                        incompatible_rows = soft_values[:, column] <= 0.0
                        if np.any(incompatible_rows):
                            raise GeomodelContractError(
                                "hard and soft conditioning support is incompatible",
                                object_name="normalize_conditioning",
                                field="soft_probabilities",
                                expected=f"positive support for hard category {int(group_hard)}",
                                actual={
                                    "rows": source_indices[soft_group[incompatible_rows]]
                                    .astype(int)
                                    .tolist(),
                                    "probabilities": soft_values[incompatible_rows].tolist(),
                                },
                            )
                    superseded_soft_indices.extend(source_indices[soft_group].astype(int).tolist())
                    group_soft = None
                    group_precision = None
        soft_columns = 0 if soft is None else int(soft.shape[1])
        output_soft.append(
            np.zeros(soft_columns, dtype=np.float64) if group_soft is None else group_soft
        )
        output_soft_present.append(group_soft is not None)
        output_precision.append(0.0 if group_precision is None else group_precision)
        output_precision_present.append(group_precision is not None)

    normalized_coordinates = np.asarray(output_coordinates, dtype=np.float64).reshape(-1, ndim)
    normalized_hard: ObservationArray | None = None
    if any(output_hard_present):
        hard_data = np.asarray(output_hard, dtype=np.float64)
        hard_mask = ~np.asarray(output_hard_present, dtype=np.bool_)
        normalized_hard = np.ma.array(hard_data, mask=hard_mask) if np.any(hard_mask) else hard_data
    normalized_soft = None
    if soft is not None and any(output_soft_present):
        soft_data = np.asarray(output_soft, dtype=np.float64).reshape(len(groups), soft.shape[1])
        soft_row_mask = ~np.asarray(output_soft_present, dtype=np.bool_)
        normalized_soft = (
            np.ma.array(
                soft_data,
                mask=np.repeat(soft_row_mask[:, None], soft.shape[1], axis=1),
            )
            if np.any(soft_row_mask)
            else soft_data
        )
    normalized_precision: ObservationArray | None = None
    if precision is not None and any(output_precision_present):
        precision_data = np.asarray(output_precision, dtype=np.float64)
        precision_mask = ~np.asarray(output_precision_present, dtype=np.bool_)
        normalized_precision = (
            np.ma.array(precision_data, mask=precision_mask)
            if np.any(precision_mask)
            else precision_data
        )
    result = ConditioningSet(
        normalized_coordinates,
        normalized_hard,
        normalized_soft,
        property,
        policy,
        soft_precision=normalized_precision,
    )
    object.__setattr__(
        result,
        "_diagnostics",
        {
            "input_count": input_count,
            "output_count": len(groups),
            "discarded_count": len(discarded_indices),
            "discarded_indices": discarded_indices,
            "discarded_coordinates_m": discarded_coordinates,
            "deduplicated_count": int(coordinates.shape[0]) - len(groups),
            "combined_soft_count": combined_soft_count,
            "superseded_soft_count": len(superseded_soft_indices),
            "superseded_soft_indices": superseded_soft_indices,
            "cell_indices": [cell.astype(int).tolist() for cell in output_cells],
        },
    )
    return result


__all__ = ["ConditioningPolicy", "ConditioningSet", "normalize_conditioning"]
