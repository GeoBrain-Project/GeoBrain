"""Indexed Direct Sequential Simulation following Soares (2001).

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
)

__all__ = ["DSSIM"]


def _empirical_quantile(sorted_data: FloatArray, uniform: float) -> float:
    count = int(sorted_data.size)
    if count == 1:
        return float(sorted_data[0])
    position = uniform * count - 0.5
    if position <= 0.0:
        return float(sorted_data[0])
    if position >= count - 1:
        return float(sorted_data[-1])
    lower = int(math.floor(position))
    fraction = position - lower
    return float(
        sorted_data[lower]
        + fraction * (sorted_data[lower + 1] - sorted_data[lower])
    )


def _empirical_interval_moments(sorted_data: FloatArray) -> tuple[float, float]:
    """Return exact moments of the midpoint-linear empirical quantile law."""
    count = int(sorted_data.size)
    if count == 1:
        return float(sorted_data[0]), 0.0
    width = 1.0 / count
    mean = 0.5 * width * float(sorted_data[0] + sorted_data[-1])
    second = 0.5 * width * float(
        sorted_data[0] ** 2 + sorted_data[-1] ** 2
    )
    for left, right in zip(sorted_data[:-1], sorted_data[1:]):
        a = float(left)
        b = float(right)
        mean += width * (a + b) / 2.0
        second += width * (a * a + a * b + b * b) / 3.0
    variance = max(0.0, second - mean * mean)
    return mean, math.sqrt(variance)


def _dssim_worker(
    index: int,
    seed: int,
    *,
    simulator: "DSSIM",
    cond_coords: FloatArray,
    cond_values: FloatArray,
    targets: FloatArray,
    sorted_data: FloatArray,
    distribution_mean: float,
    distribution_sigma: float,
) -> tuple[int, int, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
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
        execution=simulator.execution,
    )
    for assigned_node, assigned_value in assigned_nodes.items():
        traversal.append(assigned_node, assigned_value)
    sill = float(simulator.model.sill)
    singular_nodes: list[int] = []
    residuals: list[float] = []

    def draw(local_mean: float, local_sigma: float) -> float:
        quantile = _empirical_quantile(sorted_data, float(rng.uniform()))
        return float(
            local_mean
            + (local_sigma / distribution_sigma)
            * (quantile - distribution_mean)
        )

    for raw_node in path:
        node = int(raw_node)
        if bool(traversal.state.known[node]):
            continue  # pre-assigned hard-data node, never re-simulated
        selection = traversal.query(node, simulator.neighbourhood)
        ids = np.asarray(selection.ids, dtype=np.int64)
        if ids.size == 0 or selection.status == "insufficient":
            traversal.append(node, draw(distribution_mean, math.sqrt(sill)))
            continue
        nearby = np.empty((ids.size, targets.shape[1]), dtype=np.float64)
        for position, source_id in enumerate(ids.tolist()):
            nearby[position] = (
                targets[source_id]
                if source_id >= 0
                else cond_coords[-source_id - 1]
            )
        nearby_values = traversal.state.values_for(ids)
        covariance_dd = covariance_matrix(simulator.model, nearby, nearby)
        covariance_dt = covariance_vector(
            simulator.model, nearby, targets[node]
        )
        solution, residual = _solve_local(
            as_float_array(covariance_dd),
            as_float_array(covariance_dt),
            data_count=int(ids.size),
            policy=simulator.solve_policy,
            object_name="DSSIM",
        )
        if solution is None:
            singular_nodes.append(node)
            traversal.append(node, draw(distribution_mean, math.sqrt(sill)))
            continue
        conditional_mean = float(
            distribution_mean
            + np.dot(solution, nearby_values - distribution_mean)
        )
        conditional_variance = validated_variance(
            sill - float(np.dot(solution, covariance_dt)),
            sill=sill,
            object_name="DSSIM",
        )
        traversal.append(node, draw(conditional_mean, math.sqrt(conditional_variance)))
        residuals.append(residual)
    values = np.array(traversal.state.values, copy=True)
    snap_to_hard_data(values, targets, cond_coords, cond_values)
    return index, seed, values, {
        "accounting": traversal.state.accounting.to_dict(),
        "singular_fallback_count": len(singular_nodes),
        "singular_fallback_nodes": singular_nodes,
        "solve_residual_max": max(residuals, default=0.0),
        "empirical_interval_mean": distribution_mean,
        "empirical_interval_sigma": distribution_sigma,
    }


class DSSIM(SimulationAgentContract):
    """Direct sequential simulator with exact empirical-interval rescaling.

    Args:
        model: covariance model (data space).
        property: output :class:`PropertyMetadata` (name = data column).
        neighbourhood: search-neighbourhood spec (``None`` = exhaustive).
        execution: realisations / seed / workers / budget policy.
        solve_policy: singular-system handling policy.
    """

    def __init__(
        self,
        model: CovarianceModel,
        *,
        property: PropertyMetadata,
        neighbourhood: NeighbourhoodSpec,
        execution: SimulationExecutionConfig,
        solve_policy: SequentialSolvePolicy = SequentialSolvePolicy(),
    ) -> None:
        if not isinstance(model, CovarianceModel):
            raise GeomodelContractError(
                "DSSIM requires CovarianceModel",
                object_name=type(self).__name__,
                field="model",
                expected="CovarianceModel",
                actual=type(model).__name__,
            )
        if not isinstance(property, PropertyMetadata) or property.kind != "continuous":
            raise GeomodelContractError(
                "DSSIM property must be continuous",
                object_name=type(self).__name__,
                field="property",
                expected="continuous PropertyMetadata",
                actual=getattr(property, "kind", type(property).__name__),
            )
        if not isinstance(neighbourhood, NeighbourhoodSpec):
            raise GeomodelContractError(
                "DSSIM neighbourhood is invalid",
                object_name=type(self).__name__,
                field="neighbourhood",
                expected="NeighbourhoodSpec",
                actual=type(neighbourhood).__name__,
            )
        if not isinstance(execution, SimulationExecutionConfig):
            raise GeomodelContractError(
                "DSSIM execution is invalid",
                object_name=type(self).__name__,
                field="execution",
                expected="SimulationExecutionConfig",
                actual=type(execution).__name__,
            )
        if not isinstance(solve_policy, SequentialSolvePolicy):
            raise GeomodelContractError(
                "DSSIM solve policy is invalid",
                object_name=type(self).__name__,
                field="solve_policy",
                expected="SequentialSolvePolicy",
                actual=type(solve_policy).__name__,
            )
        model.require_stationary_covariance(object_name=type(self).__name__)
        self.model = model
        self.property = property
        self.neighbourhood = neighbourhood
        self.execution = execution
        self.solve_policy = solve_policy

    def __call__(
        self,
        data: GeoFrame | ConditioningSet,
        domain: Any,
    ) -> SimulationEnsemble:
        geometry, targets = resolve_sequential_domain(
            domain,
            execution=self.execution,
            object_name=type(self).__name__,
        )
        conditioning = normalize_conditioning(data, geometry, self.property)
        cond_coords, cond_values = hard_conditioning(conditioning)
        if cond_values.size < 2:
            raise GeomodelContractError(
                "DSSIM requires at least two hard values",
                object_name=type(self).__name__,
                field="data",
                expected="at least two non-constant hard values",
                actual=int(cond_values.size),
            )
        sorted_data = as_float_array(np.sort(cond_values))
        distribution_mean, distribution_sigma = _empirical_interval_moments(
            sorted_data
        )
        if distribution_sigma <= 0.0:
            raise GeomodelContractError(
                "DSSIM empirical distribution is constant",
                object_name=type(self).__name__,
                field="data",
                expected="positive empirical interval variance",
                actual=distribution_sigma,
            )
        seeds = derive_realization_seeds(
            self.execution.seed,
            self.execution.n_realizations,
        )
        worker = partial(
            _dssim_worker,
            simulator=self,
            cond_coords=cond_coords,
            cond_values=cond_values,
            targets=targets,
            sorted_data=sorted_data,
            distribution_mean=distribution_mean,
            distribution_sigma=distribution_sigma,
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
                "algorithm": "DSSIM",
                "reference": "Soares (2001), DOI 10.1023/A:1012246006212",
                "property": self.property.to_dict(),
                "neighbourhood": self.neighbourhood.to_dict(),
                "solve_policy": self.solve_policy.to_dict(),
                "conditioning": conditioning.diagnostics,
            },
        )
