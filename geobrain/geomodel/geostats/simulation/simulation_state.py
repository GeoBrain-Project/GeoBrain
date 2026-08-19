"""Preallocated append-only state for sequential simulation realizations.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ...errors import GeomodelContractError, GeomodelResourceError
from ...neighbourhood import DynamicKDTreeNeighbourhood, NeighbourhoodSelection, NeighbourhoodSpec

FloatArray = np.ndarray[tuple[Any, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[Any, ...], np.dtype[np.int64]]


@dataclass(slots=True)
class SimulationStateAccounting:
    """Monotonic operation counts for one realization state."""

    distance_checks: int = 0
    index_queries: int = 0
    index_rebuilds: int = 0
    append_writes: int = 0
    pool_rebuilds: int = 0
    candidate_comparisons: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return JSON-safe accounting without exposing mutable state."""
        return {
            "distance_checks": self.distance_checks,
            "index_queries": self.index_queries,
            "index_rebuilds": self.index_rebuilds,
            "append_writes": self.append_writes,
            "pool_rebuilds": self.pool_rebuilds,
            "candidate_comparisons": self.candidate_comparisons,
        }


def _targets(targets_m: object) -> FloatArray:
    try:
        targets = np.array(targets_m, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeomodelContractError(
            "simulation targets must be finite metre coordinates",
            object_name="SequentialSimulationState",
            field="targets_m",
            expected="finite (n, 2) or (n, 3) float64 coordinates",
            actual=type(targets_m).__name__,
        ) from exc
    if targets.ndim != 2 or targets.shape[0] < 1 or targets.shape[1] not in (2, 3):
        raise GeomodelContractError(
            "simulation targets must declare non-empty 2-D or 3-D coordinates",
            object_name="SequentialSimulationState",
            field="targets_m",
            expected="non-empty (n, 2) or (n, 3) coordinates",
            actual=tuple(targets.shape),
        )
    if not np.isfinite(targets).all():
        raise GeomodelContractError(
            "simulation targets must be finite",
            object_name="SequentialSimulationState",
            field="targets_m",
            expected="finite coordinates",
            actual="non-finite coordinate",
        )
    return cast(FloatArray, targets)


def _path(path: object, *, count: int) -> IntArray:
    values = np.asarray(path)
    if values.ndim != 1 or values.size != count or values.dtype.kind not in "iu":
        raise GeomodelContractError(
            "simulation path must be an integer permutation of target nodes",
            object_name="SequentialSimulationState",
            field="path",
            expected=f"permutation of 0..{count - 1}",
            actual={"shape": tuple(values.shape), "dtype": str(values.dtype)},
        )
    resolved = np.array(values, dtype=np.int64, copy=True, order="C")
    if not np.array_equal(np.sort(resolved), np.arange(count, dtype=np.int64)):
        raise GeomodelContractError(
            "simulation path must be an exact permutation",
            object_name="SequentialSimulationState",
            field="path",
            expected=f"permutation of 0..{count - 1}",
            actual=resolved.tolist(),
        )
    return cast(IntArray, resolved)


def _initial_nodes(
    initial: object,
    targets_m: FloatArray,
) -> tuple[FloatArray, IntArray, dict[int, float]]:
    """Normalise direct node values or a ConditioningSet-like coordinate/value pair."""
    ndim = targets_m.shape[1]
    if initial is None:
        return (
            cast(FloatArray, np.empty((0, ndim), dtype=np.float64)),
            cast(IntArray, np.empty(0, dtype=np.int64)),
            {},
        )
    if isinstance(initial, Mapping):
        node_values: dict[int, float] = {}
        for raw_node, raw_value in initial.items():
            if isinstance(raw_node, bool) or not isinstance(raw_node, (int, np.integer)):
                raise GeomodelContractError(
                    "initial node ids must be exact integers",
                    object_name="SequentialSimulationState",
                    field="initial",
                    expected="mapping[int, finite float]",
                    actual=raw_node,
                )
            node = int(raw_node)
            value = float(raw_value)
            if not 0 <= node < targets_m.shape[0] or not np.isfinite(value):
                raise GeomodelContractError(
                    "initial node values must be finite target nodes",
                    object_name="SequentialSimulationState",
                    field="initial",
                    expected="in-range node and finite value",
                    actual={"node": node, "value": raw_value},
                )
            node_values[node] = value
        ordered = np.asarray(sorted(node_values), dtype=np.int64)
        return targets_m[ordered].copy(), ordered, node_values
    if isinstance(initial, tuple) and len(initial) == 2:
        coordinates_raw, values_raw = initial
    elif hasattr(initial, "coordinates_m") and hasattr(initial, "hard_values"):
        coordinates_raw = getattr(initial, "coordinates_m")
        values_raw = getattr(initial, "hard_values")
    else:
        raise GeomodelContractError(
            "initial conditioning must be None, node-value mapping, or coordinate/value pair",
            object_name="SequentialSimulationState",
            field="initial",
            expected="None, Mapping[int, float], ConditioningSet, or (coordinates, values)",
            actual=type(initial).__name__,
        )
    coordinates = np.asarray(coordinates_raw, dtype=np.float64)
    conditioning_values = np.asarray(values_raw, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != ndim
        or conditioning_values.ndim != 1
        or conditioning_values.shape[0] != coordinates.shape[0]
        or not np.isfinite(coordinates).all()
        or not np.isfinite(conditioning_values).all()
    ):
        raise GeomodelContractError(
            "initial conditioning must contain aligned finite coordinates and values",
            object_name="SequentialSimulationState",
            field="initial",
            expected=f"(n, {ndim}) coordinates with n finite values",
            actual={
                "coordinates": tuple(coordinates.shape),
                "values": tuple(conditioning_values.shape),
            },
        )
    source_ids = -np.arange(1, coordinates.shape[0] + 1, dtype=np.int64)
    return (
        cast(FloatArray, np.array(coordinates, dtype=np.float64, copy=True, order="C")),
        cast(IntArray, source_ids),
        {
            int(source_id): float(value)
            for source_id, value in zip(source_ids, conditioning_values, strict=True)
        },
    )


class SequentialSimulationState:
    """Fixed-size target state with an append-only exact dynamic index."""

    __slots__ = (
        "targets_m",
        "path",
        "values",
        "known",
        "index",
        "accounting",
        "_known_count",
        "_initial_values",
    )

    def __init__(
        self,
        targets_m: FloatArray,
        path: IntArray,
        values: FloatArray,
        known: np.ndarray,
        index: DynamicKDTreeNeighbourhood,
        initial_values: dict[int, float],
        known_count: int,
    ) -> None:
        self.targets_m = targets_m
        self.path = path
        self.values = values
        self.known = known
        self.index = index
        self.accounting = SimulationStateAccounting(index_rebuilds=index.index_rebuilds)
        self._initial_values = initial_values
        self._known_count = known_count

    @classmethod
    def create(
        cls,
        targets_m: object,
        path: object,
        initial: object,
        *,
        budget_bytes: int | None = None,
        rebuild_batch_size: int = 32,
    ) -> "SequentialSimulationState":
        """Allocate each target array once and seed the dynamic index once."""
        targets = _targets(targets_m)
        target_path = _path(path, count=targets.shape[0])
        if isinstance(budget_bytes, (bool, np.bool_)) or (
            budget_bytes is not None and (not isinstance(budget_bytes, (int, np.integer)) or budget_bytes < 1)
        ):
            raise GeomodelContractError(
                "state budget must be a positive exact integer or None",
                object_name=cls.__name__,
                field="budget_bytes",
                expected="positive int or None",
                actual=budget_bytes,
            )
        required_bytes = int(targets.nbytes + target_path.nbytes + targets.shape[0] * 9)
        if budget_bytes is not None and required_bytes > int(budget_bytes):
            raise GeomodelResourceError(
                "simulation state allocation exceeds the configured resource budget",
                object_name=cls.__name__,
                field="budget_bytes",
                expected=f">= {required_bytes}",
                actual=int(budget_bytes),
            )
        initial_coordinates, initial_ids, initial_values = _initial_nodes(initial, targets)
        values = cast(FloatArray, np.full(targets.shape[0], np.nan, dtype=np.float64))
        known: np.ndarray[tuple[int], np.dtype[np.bool_]] = np.zeros(
            targets.shape[0], dtype=np.bool_
        )
        known_count = 0
        for source_id, value in initial_values.items():
            if source_id >= 0:
                values[source_id] = value
                known[source_id] = True
                known_count += 1
        index = DynamicKDTreeNeighbourhood.from_arrays(
            initial_coordinates,
            initial_ids,
            rebuild_batch_size=rebuild_batch_size,
        )
        return cls(targets, target_path, values, known, index, initial_values, known_count)

    @property
    def known_count(self) -> int:
        """Return the maintained count without scanning the known mask."""
        return self._known_count

    def append(self, node: int, value: float) -> None:
        """Write one target and append its id to the dynamic neighbourhood index."""
        if isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer)):
            raise GeomodelContractError(
                "simulation node must be an exact integer",
                object_name=type(self).__name__,
                field="node",
                expected="unknown target node",
                actual=node,
            )
        resolved_node = int(node)
        if not 0 <= resolved_node < self.values.size:
            raise GeomodelContractError(
                "simulation node is outside the target domain",
                object_name=type(self).__name__,
                field="node",
                expected=f"0..{self.values.size - 1}",
                actual=resolved_node,
            )
        if self.known[resolved_node]:
            raise GeomodelContractError(
                "simulation node is already known",
                object_name=type(self).__name__,
                field="node",
                expected="unknown node",
                actual=resolved_node,
            )
        resolved_value = float(value)
        if not np.isfinite(resolved_value):
            raise GeomodelContractError(
                "simulation value must be finite",
                object_name=type(self).__name__,
                field="value",
                expected="finite float",
                actual=value,
            )
        self.values[resolved_node] = resolved_value
        self.known[resolved_node] = True
        self.index.append(resolved_node, self.targets_m[resolved_node])
        self._known_count += 1
        self.accounting.append_writes += 1
        self.accounting.index_rebuilds = self.index.index_rebuilds

    def query(self, target_m: object, spec: NeighbourhoodSpec) -> NeighbourhoodSelection:
        """Query the selected exact index and record its observable work."""
        selection = self.index.query(target_m, spec)
        self.accounting.index_queries += 1
        self.accounting.distance_checks += selection.distance_checks
        self.accounting.candidate_comparisons += selection.distance_checks
        self.accounting.index_rebuilds = self.index.index_rebuilds
        return selection

    def values_for(self, ids: object) -> FloatArray:
        """Return values for dynamic target ids and negative conditioning ids."""
        resolved_ids = np.asarray(ids, dtype=np.int64)
        output = np.empty(resolved_ids.size, dtype=np.float64)
        for position, source_id in enumerate(resolved_ids):
            index = int(source_id)
            if index >= 0:
                output[position] = self.values[index]
            else:
                output[position] = self._initial_values[index]
        return cast(FloatArray, output)


__all__ = ["SequentialSimulationState", "SimulationStateAccounting"]
