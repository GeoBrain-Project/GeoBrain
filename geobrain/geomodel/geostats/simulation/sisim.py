"""Indexed Sequential Indicator Simulation (SISIM).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

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
from ..estimation.indicator_kriging import _order_relation_correction
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

__all__ = ["SISIM"]


def _invert_cdf(
    cdf: FloatArray,
    cutoffs: FloatArray,
    uniform: float,
    *,
    categorical: bool,
) -> float:
    index = int(np.searchsorted(cdf, uniform, side="right"))
    if categorical or index == 0:
        return float(cutoffs[min(index, cutoffs.size - 1)])
    if index >= cutoffs.size:
        return float(cutoffs[-1])
    lower_probability = float(cdf[index - 1])
    upper_probability = float(cdf[index])
    if upper_probability <= lower_probability:
        return float(cutoffs[index])
    fraction = (uniform - lower_probability) / (
        upper_probability - lower_probability
    )
    return float(
        cutoffs[index - 1]
        + fraction * (cutoffs[index] - cutoffs[index - 1])
    )


def _conditioning_cdf(
    conditioning: ConditioningSet,
    cutoffs: FloatArray,
) -> FloatArray:
    rows = conditioning.coordinates_m.shape[0]
    output = np.zeros((rows, cutoffs.size), dtype=np.float64)
    hard_mask = (
        np.ones(rows, dtype=np.bool_)
        if conditioning.hard_values is None
        else np.ma.getmaskarray(conditioning.hard_values)
    )
    if conditioning.hard_values is not None:
        hard_data = np.asarray(np.ma.getdata(conditioning.hard_values), dtype=np.float64)
        present = ~hard_mask
        output[present] = hard_data[present, None] <= cutoffs[None, :]
    if conditioning.soft_probabilities is not None:
        soft_mask = np.all(
            np.ma.getmaskarray(conditioning.soft_probabilities), axis=1
        )
        present = ~soft_mask
        probabilities = np.asarray(
            np.ma.getdata(conditioning.soft_probabilities), dtype=np.float64
        )
        output[present] = np.cumsum(probabilities[present], axis=1)
    return as_float_array(output)


def _sisim_worker(
    index: int,
    seed: int,
    *,
    simulator: "SISIM",
    conditioning: ConditioningSet,
    targets: FloatArray,
    marginal: FloatArray,
) -> tuple[int, int, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    path = rng.permutation(targets.shape[0])
    initial_cdf = _conditioning_cdf(conditioning, simulator.cutoffs)
    cond_coords = as_float_array(np.asarray(conditioning.coordinates_m, dtype=np.float64))
    # Only hard rows may relocate onto a coincident node; soft rows stay points.
    if conditioning.hard_values is None:
        hard_eligible = np.zeros(cond_coords.shape[0], dtype=bool)
        hard_data = np.zeros(cond_coords.shape[0], dtype=np.float64)
    else:
        hard_eligible = ~np.ma.getmaskarray(conditioning.hard_values)
        hard_data = np.asarray(np.ma.getdata(conditioning.hard_values), dtype=np.float64)
    keep_rows, assigned_nodes = assign_coincident_conditioning(
        cond_coords,
        as_float_array(hard_data),
        targets,
        eligible=hard_eligible,
    )
    if not bool(keep_rows.all()):
        cond_coords = as_float_array(np.array(cond_coords[keep_rows], dtype=np.float64))
        initial_cdf = as_float_array(np.array(initial_cdf[keep_rows], dtype=np.float64))
    traversal = SequentialTraversal(
        targets,
        path,
        cond_coords,
        as_float_array(np.zeros(cond_coords.shape[0])),
        execution=simulator.execution,
    )
    for assigned_node, assigned_value in assigned_nodes.items():
        traversal.append(assigned_node, assigned_value)
    singular_nodes: list[int] = []
    residuals: list[float] = []
    for raw_node in path:
        node = int(raw_node)
        if bool(traversal.state.known[node]):
            continue  # pre-assigned hard-data node, never re-simulated
        selection = traversal.query(node, simulator.neighbourhood)
        ids = np.asarray(selection.ids, dtype=np.int64)
        probabilities = np.array(marginal, copy=True)
        if ids.size and selection.status == "selected":
            nearby = np.empty((ids.size, targets.shape[1]), dtype=np.float64)
            indicator_values = np.empty((simulator.cutoffs.size, ids.size))
            for position, source_id in enumerate(ids.tolist()):
                if source_id >= 0:
                    nearby[position] = targets[source_id]
                    indicator_values[:, position] = (
                        traversal.state.values[source_id] <= simulator.cutoffs
                    )
                else:
                    row = -source_id - 1
                    nearby[position] = cond_coords[row]
                    indicator_values[:, position] = initial_cdf[row]
            solved = True
            for cutoff_index, model in enumerate(simulator.models):
                covariance_dd = covariance_matrix(model, nearby, nearby)
                covariance_dt = covariance_vector(model, nearby, targets[node])
                count = int(ids.size)
                matrix = as_float_array(
                    np.zeros((count + 1, count + 1), dtype=np.float64)
                )
                rhs = as_float_array(np.zeros(count + 1, dtype=np.float64))
                matrix[:count, :count] = covariance_dd
                matrix[count, :count] = 1.0
                matrix[:count, count] = 1.0
                rhs[:count] = covariance_dt
                rhs[count] = 1.0
                solution, residual = _solve_local(
                    matrix,
                    rhs,
                    data_count=count,
                    policy=simulator.solve_policy,
                    object_name="SISIM",
                )
                if solution is None:
                    singular_nodes.append(node)
                    solved = False
                    break
                probability = float(
                    np.dot(solution[:count], indicator_values[cutoff_index])
                )
                probabilities[cutoff_index] = min(1.0, max(0.0, probability))
                residuals.append(residual)
            if solved:
                probabilities = _order_relation_correction(
                    as_float_array(probabilities)
                )
            else:
                probabilities = marginal
        value = _invert_cdf(
            as_float_array(probabilities),
            simulator.cutoffs,
            float(rng.uniform()),
            categorical=simulator.categorical,
        )
        traversal.append(node, value)
    values = np.array(traversal.state.values, copy=True)
    hard_coords, hard_values = hard_conditioning(conditioning)
    snap_to_hard_data(values, targets, hard_coords, hard_values)
    return index, seed, values, {
        "accounting": traversal.state.accounting.to_dict(),
        "singular_fallback_count": len(singular_nodes),
        "singular_fallback_nodes": singular_nodes,
        "solve_residual_max": max(residuals, default=0.0),
    }


class SISIM(SimulationAgentContract):
    """Indicator sequential simulator with ordered conditional CDFs.

    Args:
        models: one covariance model per cutoff.
        cutoffs: indicator thresholds.
        property: output :class:`PropertyMetadata` (name = data column).
        neighbourhood: search-neighbourhood spec (``None`` = exhaustive).
        execution: realisations / seed / workers / budget policy.
        solve_policy: singular-system handling policy.
    """

    def __init__(
        self,
        models: tuple[CovarianceModel, ...] | list[CovarianceModel],
        cutoffs: tuple[float, ...] | list[float],
        *,
        property: PropertyMetadata,
        neighbourhood: NeighbourhoodSpec,
        execution: SimulationExecutionConfig,
        solve_policy: SequentialSolvePolicy = SequentialSolvePolicy(),
        categorical: bool = True,
        marginal_probs: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        owned_models = tuple(models)
        owned_cutoffs = tuple(float(value) for value in cutoffs)
        if not owned_models or len(owned_models) != len(owned_cutoffs):
            raise GeomodelContractError(
                "SISIM requires one covariance model per cutoff",
                object_name=type(self).__name__,
                field="models/cutoffs",
                expected="equal non-empty lengths",
                actual={"models": len(owned_models), "cutoffs": len(owned_cutoffs)},
            )
        if not all(isinstance(model, CovarianceModel) for model in owned_models):
            raise GeomodelContractError(
                "SISIM models are invalid",
                object_name=type(self).__name__,
                field="models",
                expected="CovarianceModel records",
                actual="invalid model",
            )
        if any(right <= left for left, right in zip(owned_cutoffs, owned_cutoffs[1:])):
            raise GeomodelContractError(
                "SISIM cutoffs must be strictly increasing",
                object_name=type(self).__name__,
                field="cutoffs",
                expected="strictly increasing finite values",
                actual=owned_cutoffs,
            )
        if not isinstance(property, PropertyMetadata):
            raise GeomodelContractError(
                "SISIM property is invalid",
                object_name=type(self).__name__,
                field="property",
                expected="PropertyMetadata",
                actual=type(property).__name__,
            )
        if categorical and (
            property.kind != "categorical"
            or tuple(float(code) for code in property.category_codes) != owned_cutoffs
        ):
            raise GeomodelContractError(
                "categorical SISIM cutoffs must match the property vocabulary",
                object_name=type(self).__name__,
                field="cutoffs",
                expected=list(property.category_codes),
                actual=owned_cutoffs,
            )
        if not categorical and property.kind != "continuous":
            raise GeomodelContractError(
                "continuous SISIM requires a continuous property",
                object_name=type(self).__name__,
                field="property",
                expected="continuous PropertyMetadata",
                actual=property.kind,
            )
        if not isinstance(neighbourhood, NeighbourhoodSpec):
            raise GeomodelContractError(
                "SISIM neighbourhood is invalid",
                object_name=type(self).__name__,
                field="neighbourhood",
                expected="NeighbourhoodSpec",
                actual=type(neighbourhood).__name__,
            )
        if not isinstance(execution, SimulationExecutionConfig):
            raise GeomodelContractError(
                "SISIM execution is invalid",
                object_name=type(self).__name__,
                field="execution",
                expected="SimulationExecutionConfig",
                actual=type(execution).__name__,
            )
        if not isinstance(solve_policy, SequentialSolvePolicy):
            raise GeomodelContractError(
                "SISIM solve policy is invalid",
                object_name=type(self).__name__,
                field="solve_policy",
                expected="SequentialSolvePolicy",
                actual=type(solve_policy).__name__,
            )
        marginal = None if marginal_probs is None else tuple(float(v) for v in marginal_probs)
        if marginal is not None and len(marginal) != len(owned_cutoffs):
            raise GeomodelContractError(
                "SISIM marginal CDF length is invalid",
                object_name=type(self).__name__,
                field="marginal_probs",
                expected=len(owned_cutoffs),
                actual=len(marginal),
            )
        for model in owned_models:
            model.require_stationary_covariance(object_name=type(self).__name__)
        self.models = owned_models
        self.cutoffs = as_float_array(owned_cutoffs)
        self.property = property
        self.neighbourhood = neighbourhood
        self.execution = execution
        self.solve_policy = solve_policy
        self.categorical = bool(categorical)
        self.marginal_probs = marginal

    def __call__(
        self,
        data: GeoFrame | ConditioningSet | None,
        domain: Any,
    ) -> SimulationEnsemble:
        geometry, targets = resolve_sequential_domain(
            domain,
            execution=self.execution,
            object_name=type(self).__name__,
        )
        conditioning = normalize_conditioning(data, geometry, self.property)
        initial_cdf = _conditioning_cdf(conditioning, self.cutoffs)
        if self.marginal_probs is not None:
            marginal = as_float_array(self.marginal_probs)
        elif initial_cdf.shape[0]:
            marginal = as_float_array(np.mean(initial_cdf, axis=0))
        else:
            marginal = as_float_array(
                np.arange(1, self.cutoffs.size + 1) / self.cutoffs.size
            )
        marginal = _order_relation_correction(marginal)
        seeds = derive_realization_seeds(
            self.execution.seed,
            self.execution.n_realizations,
        )
        worker = partial(
            _sisim_worker,
            simulator=self,
            conditioning=conditioning,
            targets=targets,
            marginal=marginal,
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
                "algorithm": "SISIM",
                "property": self.property.to_dict(),
                "neighbourhood": self.neighbourhood.to_dict(),
                "solve_policy": self.solve_policy.to_dict(),
                "conditioning": conditioning.diagnostics,
                "marginal_cdf": marginal.tolist(),
            },
        )
