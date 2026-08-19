"""Indexed collocated sequential Gaussian simulation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, cast

import numpy as np

from ...conditioning import ConditioningSet, normalize_conditioning
from ...frames import GeoFrame, PropertyMetadata
from ...frames._arrays import FloatArray, as_float_array
from ...errors import GeomodelContractError
from ...neighbourhood import NeighbourhoodSpec
from .._domain import derive_realization_seeds
from ..estimation.covariance_matrix import covariance_matrix, covariance_vector
from ..estimation.kriging_kernel import validated_variance
from ..models.covariance import CovarianceModel
from ._grid_utils import snap_to_hard_data
from ._parallel import RealizationRun, run_realizations
from .execution import SimulationExecutionConfig
from .agent_contract import SimulationAgentContract
from .results import SimulationEnsemble
from .sequential import (
    SequentialSolvePolicy,
    SequentialTraversal,
    _solve_local,
    assign_coincident_conditioning,
    assemble_ensemble,
    hard_conditioning,
    make_simulation_frame,
    resolve_sequential_domain,
    simulate_gaussian_kernel,
)

__all__ = ["CoSGSIM"]


def _cosgsim_worker(
    index: int,
    seed: int,
    *,
    cond_coords: FloatArray,
    cond_values: FloatArray,
    targets: FloatArray,
    secondary: FloatArray,
    model: CovarianceModel,
    correlation: float,
    secondary_sill: float,
    primary_mean: float,
    secondary_mean: float,
    neighbourhood: NeighbourhoodSpec,
    execution: SimulationExecutionConfig,
    solve_policy: SequentialSolvePolicy,
) -> tuple[int, int, FloatArray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    if correlation == 0.0:
        result = simulate_gaussian_kernel(
            cond_coords,
            cond_values,
            targets,
            model,
            neighbourhood=neighbourhood,
            execution=execution,
            solve_policy=solve_policy,
            rng=rng,
            mean=primary_mean,
            object_name="CoSGSIM",
        )
        diagnostics = dict(result.diagnostics)
        diagnostics["collocated_adjustment_count"] = 0
        return index, seed, result.values, diagnostics

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
    primary_sill = float(model.sill)
    unconditional_nodes: list[int] = []
    singular_nodes: list[int] = []
    residuals: list[float] = []
    for raw_node in path:
        node = int(raw_node)
        if bool(traversal.state.known[node]):
            continue  # pre-assigned hard-data node, never re-simulated
        selection = traversal.query(node, neighbourhood)
        ids = np.asarray(selection.ids, dtype=np.int64)
        base_mean = primary_mean
        base_variance = primary_sill
        if ids.size and selection.status != "insufficient":
            nearby = np.empty((ids.size, targets.shape[1]), dtype=np.float64)
            for position, source_id in enumerate(ids.tolist()):
                nearby[position] = (
                    targets[source_id]
                    if source_id >= 0
                    else cond_coords[-source_id - 1]
                )
            local_values = traversal.state.values_for(ids)
            matrix = covariance_matrix(model, nearby, nearby)
            rhs = covariance_vector(model, nearby, targets[node])
            solution, residual = _solve_local(
                matrix,
                rhs,
                data_count=int(ids.size),
                policy=solve_policy,
                object_name="CoSGSIM",
            )
            if solution is None:
                singular_nodes.append(node)
                unconditional_nodes.append(node)
            else:
                weights = solution[: ids.size]
                base_mean += float(np.dot(weights, local_values - primary_mean))
                base_variance = validated_variance(
                    primary_sill - float(np.dot(weights, rhs)),
                    sill=primary_sill,
                    object_name="CoSGSIM",
                )
                residuals.append(residual)
        else:
            unconditional_nodes.append(node)
        scale = math.sqrt(max(base_variance, 0.0) / secondary_sill)
        collocated_mean = base_mean + correlation * scale * (
            float(secondary[node]) - secondary_mean
        )
        collocated_variance = validated_variance(
            base_variance * (1.0 - correlation * correlation),
            sill=primary_sill,
            object_name="CoSGSIM",
        )
        value = float(rng.normal(collocated_mean, math.sqrt(collocated_variance)))
        traversal.append(node, value)

    values = as_float_array(np.array(traversal.state.values, copy=True))
    snap_to_hard_data(values, targets, cond_coords, cond_values)
    diagnostics: dict[str, object] = {
        "accounting": traversal.state.accounting.to_dict(),
        "neighbourhood_backend": execution.neighbourhood_backend,
        "unconditional_node_count": len(unconditional_nodes),
        "unconditional_nodes": unconditional_nodes,
        "singular_fallback_count": len(singular_nodes),
        "singular_fallback_nodes": singular_nodes,
        "solve_residual_max": max(residuals, default=0.0),
        "collocated_adjustment_count": int(targets.shape[0]),
    }
    return index, seed, values, diagnostics


class CoSGSIM(SimulationAgentContract):
    """Collocated Markov-model sequential Gaussian simulator.

    Args:
        model: primary covariance model (normal-score space).
        correlation: primary-secondary correlation.
        property / secondary_property: primary (conditioning column) and
            collocated secondary properties.
        neighbourhood: search-neighbourhood spec (``None`` = exhaustive).
        execution: realisations / seed / workers / budget policy.
        solve_policy: singular-system handling policy.
        secondary_sill / primary_mean / secondary_mean: moments.
    """

    def __init__(
        self,
        model: CovarianceModel,
        correlation: float,
        *,
        property: PropertyMetadata,
        secondary_property: PropertyMetadata,
        neighbourhood: NeighbourhoodSpec,
        execution: SimulationExecutionConfig,
        solve_policy: SequentialSolvePolicy = SequentialSolvePolicy(),
        secondary_sill: float | None = None,
        primary_mean: float = 0.0,
        secondary_mean: float = 0.0,
    ) -> None:
        if not isinstance(model, CovarianceModel):
            raise GeomodelContractError(
                "CoSGSIM requires a covariance model",
                object_name=type(self).__name__, field="model",
                expected="CovarianceModel", actual=type(model).__name__,
            )
        if not isinstance(property, PropertyMetadata) or property.kind != "continuous":
            raise GeomodelContractError(
                "primary property must be continuous",
                object_name=type(self).__name__, field="property",
                expected="continuous PropertyMetadata", actual=type(property).__name__,
            )
        if not isinstance(secondary_property, PropertyMetadata) or secondary_property.kind != "continuous":
            raise GeomodelContractError(
                "secondary property must be continuous",
                object_name=type(self).__name__, field="secondary_property",
                expected="continuous PropertyMetadata", actual=type(secondary_property).__name__,
            )
        if not isinstance(neighbourhood, NeighbourhoodSpec):
            raise GeomodelContractError(
                "CoSGSIM requires NeighbourhoodSpec",
                object_name=type(self).__name__, field="neighbourhood",
                expected="NeighbourhoodSpec", actual=type(neighbourhood).__name__,
            )
        if not isinstance(execution, SimulationExecutionConfig):
            raise GeomodelContractError(
                "CoSGSIM requires SimulationExecutionConfig",
                object_name=type(self).__name__, field="execution",
                expected="SimulationExecutionConfig", actual=type(execution).__name__,
            )
        if not isinstance(solve_policy, SequentialSolvePolicy):
            raise GeomodelContractError(
                "CoSGSIM requires SequentialSolvePolicy",
                object_name=type(self).__name__, field="solve_policy",
                expected="SequentialSolvePolicy", actual=type(solve_policy).__name__,
            )
        rho = float(correlation)
        if not math.isfinite(rho) or not -1.0 <= rho <= 1.0:
            raise GeomodelContractError(
                "correlation must lie in [-1, 1]",
                object_name=type(self).__name__, field="correlation",
                expected="finite value in [-1, 1]", actual=correlation,
            )
        resolved_secondary_sill = float(model.sill if secondary_sill is None else secondary_sill)
        if not math.isfinite(resolved_secondary_sill) or resolved_secondary_sill <= 0.0:
            raise GeomodelContractError(
                "secondary sill must be positive",
                object_name=type(self).__name__, field="secondary_sill",
                expected="> 0", actual=secondary_sill,
            )
        model.require_stationary_covariance(object_name=type(self).__name__)
        self.model = model
        self.correlation = rho
        self.property = property
        self.secondary_property = secondary_property
        self.neighbourhood = neighbourhood
        self.execution = execution
        self.solve_policy = solve_policy
        self.secondary_sill = resolved_secondary_sill
        self.primary_mean = float(primary_mean)
        self.secondary_mean = float(secondary_mean)

    def __call__(
        self,
        data: GeoFrame | ConditioningSet | None,
        domain: Any,
    ) -> SimulationEnsemble:
        if not isinstance(domain, GeoFrame):
            raise GeomodelContractError(
                "CoSGSIM requires a GeoFrame domain containing the dense secondary property",
                object_name=type(self).__name__, field="domain",
                expected="GeoFrame", actual=type(domain).__name__,
            )
        if self.secondary_property.name not in domain.columns:
            raise GeomodelContractError(
                "domain is missing the collocated secondary property",
                object_name=type(self).__name__, field="domain.secondary_property",
                expected=self.secondary_property.name, actual=list(domain.columns),
            )
        geometry, targets = resolve_sequential_domain(
            domain,
            execution=self.execution,
            object_name=type(self).__name__,
        )
        secondary = as_float_array(domain[self.secondary_property.name])
        self.secondary_property.validate_values(secondary, object_name=type(self).__name__)
        conditioning = normalize_conditioning(data, geometry, self.property)
        cond_coords, cond_values = hard_conditioning(conditioning)
        seeds = derive_realization_seeds(self.execution.seed, self.execution.n_realizations)
        worker = partial(
            _cosgsim_worker,
            cond_coords=cond_coords,
            cond_values=cond_values,
            targets=targets,
            secondary=secondary,
            model=self.model,
            correlation=self.correlation,
            secondary_sill=self.secondary_sill,
            primary_mean=self.primary_mean,
            secondary_mean=self.secondary_mean,
            neighbourhood=self.neighbourhood,
            execution=self.execution,
            solve_policy=self.solve_policy,
        )
        run = cast(RealizationRun, run_realizations(worker, seeds, self.execution))
        frames = tuple(
            make_simulation_frame(geometry, item.result, self.property)
            for item in run.results
        )
        return assemble_ensemble(
            self.property,
            self.execution,
            run,
            frames,
            diagnostics={
                "algorithm": "CoSGSIM",
                "property": self.property.to_dict(),
                "secondary_property": self.secondary_property.to_dict(),
                "correlation": self.correlation,
                "neighbourhood": self.neighbourhood.to_dict(),
                "solve_policy": self.solve_policy.to_dict(),
                "conditioning": conditioning.diagnostics,
            },
        )

    def __repr__(self) -> str:
        return f"CoSGSIM(model={self.model!r}, correlation={self.correlation})"
