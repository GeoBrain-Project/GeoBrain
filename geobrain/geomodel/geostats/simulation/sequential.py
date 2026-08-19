"""Indexed, dimension-aware kernels for sequential simulation families.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from ...conditioning import ConditioningSet
from ...frames import ColumnRole, GeoFrame, Geometry, PropertyMetadata
from ...frames._arrays import FloatArray, as_float_array
from ...errors import (
    GeomodelContractError,
    GeomodelNumericsError,
    GeomodelResourceError,
)
from ...neighbourhood import (
    ExhaustiveNeighbourhood,
    NeighbourhoodSelection,
    NeighbourhoodSpec,
)
from .._domain import resolve_domain
from ..estimation.covariance_matrix import covariance_matrix, covariance_vector
from ..estimation.kriging_kernel import (
    KrigingSolvePolicy,
    solve_kriging_system,
    validated_variance,
)
from ..models.covariance import CovarianceModel
from ._grid_utils import snap_to_hard_data
from ._parallel import RealizationRun
from .execution import SimulationExecutionConfig
from .results import SimulationEnsemble, SimulationRealization
from .simulation_state import SequentialSimulationState

__all__ = [
    "SequentialSolvePolicy",
    "assemble_ensemble",
    "hard_conditioning",
    "make_simulation_frame",
    "preflight_sequential_domain",
    "resolve_sequential_domain",
    "simulate_gaussian_kernel",
    "simulate_realization",
]


@dataclass(frozen=True, slots=True)
class SequentialSolvePolicy:
    """Explicit response to a singular local sequential system."""

    on_singular: Literal["error", "unconditional"] = "error"

    def __post_init__(self) -> None:
        if self.on_singular not in ("error", "unconditional"):
            raise GeomodelContractError(
                "invalid sequential singular-system policy",
                object_name=type(self).__name__,
                field="on_singular",
                expected="'error' or 'unconditional'",
                actual=self.on_singular,
            )

    def to_dict(self) -> dict[str, object]:
        return {"on_singular": self.on_singular}


@dataclass(frozen=True, slots=True)
class SequentialKernelResult:
    """One owned realization array and strict JSON diagnostics."""

    values: FloatArray
    diagnostics: Mapping[str, object]


def _domain_count(domain: object) -> int:
    geometry = domain.geometry if isinstance(domain, GeoFrame) else domain
    if isinstance(geometry, Geometry):
        return int(geometry.npoints)
    values = np.asarray(domain)
    if values.ndim != 2 or values.shape[1] not in (2, 3):
        raise GeomodelContractError(
            "simulation domain must be 2-D or 3-D coordinates",
            object_name="sequential simulation",
            field="domain",
            expected="Geometry, GeoFrame, or (n, 2|3) array",
            actual=tuple(values.shape),
        )
    return int(values.shape[0])


def preflight_sequential_domain(
    domain: object,
    execution: SimulationExecutionConfig,
) -> None:
    """Reject governed quadratic work and state bytes before target allocation."""
    count = _domain_count(domain)
    distance_checks = count * (count + 1) // 2
    cumulative_copy_bytes = distance_checks * 32
    if execution.neighbourhood_backend == "exhaustive" and count >= 500_000:
        raise GeomodelResourceError(
            "exhaustive sequential simulation exceeds the governed work limit",
            object_name="sequential simulation",
            field="neighbourhood_backend",
            expected="indexed backend for 500,000 or more target nodes",
            actual={
                "target_nodes": count,
                "distance_checks": distance_checks,
                "cumulative_copy_bytes": cumulative_copy_bytes,
            },
        )
    # Coordinates, path, values, mask and dynamic-index storage per active worker.
    active_workers = min(execution.workers, execution.n_realizations)
    required_bytes = count * (3 * 8 + 8 + 8 + 1) * active_workers
    execution.require_budget(required_bytes, component="sequential state")


def resolve_sequential_domain(
    domain: object,
    *,
    execution: SimulationExecutionConfig,
    object_name: str,
) -> tuple[Geometry, FloatArray]:
    preflight_sequential_domain(domain, execution)
    geometry, targets = resolve_domain(
        domain,
        object_name=object_name,
        preserve_dimension=True,
    )
    return geometry, as_float_array(targets)


def hard_conditioning(conditioning: ConditioningSet) -> tuple[FloatArray, FloatArray]:
    """Return only present hard rows from one normalized conditioning set."""
    if conditioning.hard_values is None:
        return (
            as_float_array(np.empty((0, conditioning.coordinates_m.shape[1]))),
            as_float_array(np.empty(0)),
        )
    mask = np.ma.getmaskarray(conditioning.hard_values)
    present = ~mask
    coordinates = conditioning.coordinates_m[present]
    values = np.asarray(np.ma.getdata(conditioning.hard_values)[present], dtype=np.float64)
    return as_float_array(coordinates), as_float_array(values)


def assign_coincident_conditioning(
    cond_coords: FloatArray,
    cond_values: FloatArray,
    targets: FloatArray,
    *,
    eligible: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[int, float]]:
    """GSLIB-style relocation of hard data onto coincident target nodes.

    A hard datum lying exactly on a target node (within the same 1e-12
    squared-distance tolerance as :func:`snap_to_hard_data`) becomes that
    node's pre-assigned value and leaves the point pool. Without this step a
    later neighbourhood contains the same location twice, the datum and the
    already-simulated node, and the local kriging system is exactly
    singular. Duplicate data on one node keep the last value, mirroring
    :func:`~._grid_utils.fill_hard_data`.

    Returns ``(keep, assigned)``: a boolean keep-mask over the data rows and
    the ``{node: value}`` pre-assignments. ``eligible`` restricts which rows
    may relocate (e.g. hard-only rows for indicator data).
    """
    keep = np.ones(cond_coords.shape[0], dtype=bool)
    assigned: dict[int, float] = {}
    if cond_coords.shape[0] == 0 or targets.shape[0] == 0:
        return keep, assigned
    rows = (
        range(cond_coords.shape[0])
        if eligible is None
        else np.flatnonzero(np.asarray(eligible, dtype=bool)).tolist()
    )
    for k in rows:
        squared = np.sum((targets - cond_coords[k]) ** 2, axis=1)
        node = int(np.argmin(squared))
        if float(squared[node]) < 1.0e-12:
            assigned[node] = float(cond_values[k])
            keep[k] = False
    return keep, assigned


class SequentialTraversal:
    """Preallocated state plus an exhaustive correctness-oracle view."""

    def __init__(
        self,
        targets: FloatArray,
        path: np.ndarray,
        initial_coords: FloatArray,
        initial_values: FloatArray,
        *,
        execution: SimulationExecutionConfig,
    ) -> None:
        self.backend = execution.neighbourhood_backend
        self.initial_coords = as_float_array(initial_coords)
        self.initial_ids = -np.arange(1, initial_coords.shape[0] + 1, dtype=np.int64)
        self.state = SequentialSimulationState.create(
            targets,
            path,
            (initial_coords, initial_values),
            budget_bytes=execution.budget_bytes,
        )

    def query(self, node: int, spec: NeighbourhoodSpec) -> NeighbourhoodSelection:
        target = self.state.targets_m[node]
        if self.backend == "indexed":
            return self.state.query(target, spec)
        known_ids = np.flatnonzero(self.state.known).astype(np.int64)
        if self.initial_coords.size and known_ids.size:
            coordinates = np.concatenate(
                (self.initial_coords, self.state.targets_m[known_ids]), axis=0
            )
            source_ids = np.concatenate((self.initial_ids, known_ids))
        elif self.initial_coords.size:
            coordinates = self.initial_coords
            source_ids = self.initial_ids
        else:
            coordinates = self.state.targets_m[known_ids]
            source_ids = known_ids
        oracle = ExhaustiveNeighbourhood.from_arrays(coordinates, source_ids)
        selection = oracle.query(target, spec)
        accounting = self.state.accounting
        accounting.index_queries += 1
        accounting.distance_checks += selection.distance_checks
        accounting.candidate_comparisons += selection.distance_checks
        return selection

    def append(self, node: int, value: float) -> None:
        self.state.append(node, value)


def _solve_local(
    matrix: FloatArray,
    rhs: FloatArray,
    *,
    data_count: int,
    policy: SequentialSolvePolicy,
    object_name: str,
) -> tuple[FloatArray | None, float]:
    try:
        solution, _, residual = solve_kriging_system(
            matrix,
            rhs,
            data_count=data_count,
            policy=KrigingSolvePolicy(on_singular="error"),
            object_name=object_name,
        )
    except GeomodelNumericsError:
        if policy.on_singular == "unconditional":
            return None, 0.0
        raise
    return solution, residual


def simulate_gaussian_kernel(
    cond_coords: FloatArray,
    cond_values: FloatArray,
    targets: FloatArray,
    model: CovarianceModel,
    *,
    neighbourhood: NeighbourhoodSpec,
    execution: SimulationExecutionConfig,
    solve_policy: SequentialSolvePolicy,
    rng: np.random.Generator,
    ktype: Literal[0, 1] = 0,
    mean: float = 0.0,
    object_name: str = "SGSIM",
) -> SequentialKernelResult:
    """Simulate one Gaussian realization through indexed append-only state."""
    model.require_stationary_covariance(object_name=object_name)
    if neighbourhood.ndim != targets.shape[1]:
        raise GeomodelContractError(
            "simulation neighbourhood dimension does not match the domain",
            object_name=object_name,
            field="neighbourhood",
            expected=f"{targets.shape[1]}-D NeighbourhoodSpec",
            actual=f"{neighbourhood.ndim}-D",
        )
    if ktype not in (0, 1):
        raise GeomodelContractError(
            "Gaussian sequential kriging type is invalid",
            object_name=object_name,
            field="ktype",
            expected="0 or 1",
            actual=ktype,
        )
    path = rng.permutation(targets.shape[0])
    keep_rows, assigned_nodes = assign_coincident_conditioning(
        cond_coords, cond_values, targets
    )
    if not bool(keep_rows.all()):
        cond_coords = as_float_array(np.array(cond_coords[keep_rows], dtype=np.float64))
        cond_values = as_float_array(np.array(cond_values[keep_rows], dtype=np.float64))
    traversal = SequentialTraversal(
        targets,
        path,
        cond_coords,
        cond_values,
        execution=execution,
    )
    for assigned_node, assigned_value in assigned_nodes.items():
        traversal.append(assigned_node, assigned_value)
    sill = float(model.sill)
    sigma = math.sqrt(max(sill, 0.0))
    unconditional_nodes: list[int] = []
    singular_nodes: list[int] = []
    residuals: list[float] = []

    for raw_node in path:
        node = int(raw_node)
        if bool(traversal.state.known[node]):
            continue  # pre-assigned hard-data node, never re-simulated
        selection = traversal.query(node, neighbourhood)
        ids = np.asarray(selection.ids, dtype=np.int64)
        if ids.size == 0 or selection.status == "insufficient":
            value = float(rng.normal(mean, sigma))
            unconditional_nodes.append(node)
            traversal.append(node, value)
            continue
        nearby = np.empty((ids.size, targets.shape[1]), dtype=np.float64)
        for position, source_id in enumerate(ids.tolist()):
            nearby[position] = (
                targets[source_id]
                if source_id >= 0
                else cond_coords[-source_id - 1]
            )
        values = traversal.state.values_for(ids)
        covariance_dd = covariance_matrix(model, nearby, nearby)
        covariance_dt = covariance_vector(model, nearby, targets[node])
        nd = int(ids.size)
        if ktype == 0:
            matrix = as_float_array(covariance_dd)
            rhs = as_float_array(covariance_dt)
        else:
            matrix = as_float_array(np.zeros((nd + 1, nd + 1), dtype=np.float64))
            rhs = as_float_array(np.zeros(nd + 1, dtype=np.float64))
            matrix[:nd, :nd] = covariance_dd
            matrix[nd, :nd] = 1.0
            matrix[:nd, nd] = 1.0
            rhs[:nd] = covariance_dt
            rhs[nd] = 1.0
        solution, residual = _solve_local(
            matrix,
            rhs,
            data_count=nd,
            policy=solve_policy,
            object_name=object_name,
        )
        if solution is None:
            value = float(rng.normal(mean, sigma))
            singular_nodes.append(node)
            unconditional_nodes.append(node)
            traversal.append(node, value)
            continue
        weights = solution[:nd]
        conditional_mean = (
            float(mean + np.dot(weights, values - mean))
            if ktype == 0
            else float(np.dot(weights, values))
        )
        conditional_variance = sill - float(np.dot(weights, covariance_dt))
        if ktype == 1:
            conditional_variance -= float(solution[nd])
        conditional_variance = validated_variance(
            conditional_variance,
            sill=sill,
            object_name=object_name,
        )
        value = float(rng.normal(conditional_mean, math.sqrt(conditional_variance)))
        traversal.append(node, value)
        residuals.append(residual)

    output = np.array(traversal.state.values, dtype=np.float64, copy=True)
    snap_to_hard_data(output, targets, cond_coords, cond_values)
    accounting = traversal.state.accounting.to_dict()
    diagnostics: dict[str, object] = {
        "accounting": accounting,
        "neighbourhood_backend": execution.neighbourhood_backend,
        "unconditional_node_count": len(unconditional_nodes),
        "unconditional_nodes": unconditional_nodes,
        "singular_fallback_count": len(singular_nodes),
        "singular_fallback_nodes": singular_nodes,
        "solve_residual_max": max(residuals, default=0.0),
    }
    return SequentialKernelResult(as_float_array(output), diagnostics)


def simulate_realization(
    cond_coords: FloatArray,
    cond_values: FloatArray,
    target_coords: FloatArray,
    model: CovarianceModel,
    *,
    neighbourhood: NeighbourhoodSpec,
    execution: SimulationExecutionConfig = SimulationExecutionConfig(),
    solve_policy: SequentialSolvePolicy = SequentialSolvePolicy(),
    ktype: Literal[0, 1] = 0,
    mean: float = 0.0,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Compatibility-sized public wrapper over the indexed Gaussian kernel.

    Args:
        cond_coords / cond_values: conditioning data.
        target_coords: simulation path coordinates.
        model: covariance model.
        neighbourhood: search-neighbourhood spec (``None`` = exhaustive).
        execution: realisations / seed / workers / budget policy.
    """
    generator = np.random.default_rng() if rng is None else rng
    return simulate_gaussian_kernel(
        as_float_array(cond_coords),
        as_float_array(cond_values),
        as_float_array(target_coords),
        model,
        neighbourhood=neighbourhood,
        execution=execution,
        solve_policy=solve_policy,
        rng=generator,
        ktype=ktype,
        mean=mean,
    ).values


def _output_metadata(property: PropertyMetadata) -> PropertyMetadata:
    return PropertyMetadata(
        "simulation",
        property.kind,
        property.unit,
        property.categories,
    )


def make_simulation_frame(
    geometry: Geometry,
    values: object,
    property: PropertyMetadata,
    *,
    extras: Mapping[str, tuple[object, PropertyMetadata]] | None = None,
) -> GeoFrame:
    properties: dict[str, object] = {"simulation": values}
    metadata: dict[str, PropertyMetadata] = {"simulation": _output_metadata(property)}
    if extras:
        for name, (extra_values, extra_metadata) in extras.items():
            properties[name] = extra_values
            metadata[name] = extra_metadata
    frame = GeoFrame(geometry, properties, metadata=metadata)
    frame.set_role("simulation", ColumnRole.SIMULATION)
    return frame


def assemble_ensemble(
    property: PropertyMetadata,
    execution: SimulationExecutionConfig,
    run: RealizationRun,
    frames: tuple[GeoFrame, ...],
    *,
    diagnostics: Mapping[str, object],
) -> SimulationEnsemble:
    if len(frames) != len(run.results):
        raise GeomodelContractError(
            "simulation frames do not match worker results",
            object_name="assemble_ensemble",
            field="frames",
            expected=len(run.results),
            actual=len(frames),
        )
    realizations = tuple(
        SimulationRealization(item.index, item.seed, frame, item.diagnostics)
        for item, frame in zip(run.results, frames, strict=True)
    )
    accounting_keys = (
        "distance_checks",
        "index_queries",
        "index_rebuilds",
        "append_writes",
        "pool_rebuilds",
        "candidate_comparisons",
    )
    totals = {key: 0 for key in accounting_keys}
    for item in run.results:
        accounting = item.diagnostics.get("accounting", {})
        if isinstance(accounting, Mapping):
            for key in accounting_keys:
                totals[key] += int(cast(Any, accounting.get(key, 0)))
    payload = dict(diagnostics)
    payload.update(dict(run.diagnostics))
    payload["accounting"] = totals
    payload["neighbourhood_backend"] = execution.neighbourhood_backend
    return SimulationEnsemble(property, realizations, execution, payload)
