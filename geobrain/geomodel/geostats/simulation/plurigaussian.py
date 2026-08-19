"""Indexed plurigaussian categorical simulation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from statistics import NormalDist
from typing import Any, cast

import numpy as np

from ...conditioning import ConditioningSet, normalize_conditioning
from ...frames import GeoFrame, PropertyMetadata
from ...frames._arrays import FloatArray, as_float_array
from ...errors import GeomodelContractError, GeomodelNumericsError
from ...neighbourhood import NeighbourhoodSpec
from .._domain import derive_realization_seeds
from ..models.covariance import CovarianceModel
from ._parallel import RealizationRun, run_realizations
from .execution import SimulationExecutionConfig
from .agent_contract import SimulationAgentContract
from .results import SimulationEnsemble
from .sequential import (
    SequentialSolvePolicy,
    assemble_ensemble,
    make_simulation_frame,
    resolve_sequential_domain,
    simulate_gaussian_kernel,
)

__all__ = ["PlurigaussianSim"]


def _thresholds(count: int) -> FloatArray:
    normal = NormalDist()
    return as_float_array(
        np.asarray([normal.inv_cdf(i / count) for i in range(1, count)], dtype=np.float64)
    )


def _digitize(values: FloatArray, count: int) -> np.ndarray:
    return np.searchsorted(_thresholds(count), values, side="right").astype(np.int64)


def _apply_rule(
    rule: Callable[..., object] | np.ndarray,
    fields: tuple[FloatArray, ...],
) -> np.ndarray:
    if callable(rule):
        values = np.asarray(rule(*fields))
    else:
        values = rule[_digitize(fields[0], rule.shape[0]), _digitize(fields[1], rule.shape[1])]
    return np.asarray(values, dtype=np.float64)


def _plurigaussian_worker(
    index: int,
    seed: int,
    *,
    targets: FloatArray,
    variograms: tuple[CovarianceModel, ...],
    rule: Callable[..., object] | np.ndarray,
    property: PropertyMetadata,
    neighbourhood: NeighbourhoodSpec,
    execution: SimulationExecutionConfig,
    solve_policy: SequentialSolvePolicy,
    hard_coordinates: FloatArray,
    hard_categories: FloatArray,
    max_iter: int,
    step: float,
) -> tuple[int, int, FloatArray, dict[str, object]]:
    master = np.random.default_rng(seed)
    field_seeds = master.integers(0, np.iinfo(np.int64).max, size=len(variograms), dtype=np.int64)
    fields: list[FloatArray] = []
    field_diagnostics: list[dict[str, object]] = []
    for model, field_seed in zip(variograms, field_seeds, strict=True):
        result = simulate_gaussian_kernel(
            as_float_array(np.empty((0, targets.shape[1]))),
            as_float_array(np.empty(0)),
            targets,
            model,
            neighbourhood=neighbourhood,
            execution=execution,
            solve_policy=solve_policy,
            rng=np.random.default_rng(int(field_seed)),
            object_name="PlurigaussianSim",
        )
        fields.append(result.values)
        field_diagnostics.append(dict(result.diagnostics))
    category_values = _apply_rule(rule, tuple(fields))
    unresolved: list[int] = []
    if hard_categories.size:
        distances = np.sum((targets[:, None, :] - hard_coordinates[None, :, :]) ** 2, axis=2)
        nodes = np.argmin(distances, axis=0)
        if not callable(rule):
            centres = tuple(
                as_float_array(
                    np.asarray(
                        [NormalDist().inv_cdf((i + 0.5) / count) for i in range(count)],
                        dtype=np.float64,
                    )
                )
                for count in rule.shape
            )
            for node, requested in zip(nodes.tolist(), hard_categories.tolist(), strict=True):
                matches = np.argwhere(rule == int(requested))
                if matches.size == 0:
                    unresolved.append(int(node))
                    continue
                selected = matches[0]
                for field_index in range(len(fields)):
                    fields[field_index][node] = centres[field_index][selected[field_index]]
            category_values = _apply_rule(rule, tuple(fields))
        else:
            for _ in range(max_iter):
                mismatches = [
                    position
                    for position, node in enumerate(nodes.tolist())
                    if int(category_values[node]) != int(hard_categories[position])
                ]
                if not mismatches:
                    break
                for position in mismatches:
                    node = int(nodes[position])
                    for field in fields:
                        field[node] += float(master.normal(0.0, step))
                category_values = _apply_rule(rule, tuple(fields))
            unresolved.extend(
                int(node)
                for position, node in enumerate(nodes.tolist())
                if int(category_values[node]) != int(hard_categories[position])
            )
    if unresolved:
        raise GeomodelNumericsError(
            "plurigaussian truncation rule could not honour hard conditioning",
            object_name="PlurigaussianSim",
            field="conditioning",
            expected="all requested categories reachable by the truncation rule",
            actual={"unresolved_nodes": sorted(set(unresolved))},
        )
    property.validate_values(category_values, object_name="PlurigaussianSim")
    packed = as_float_array(np.vstack((category_values, *fields)))
    accounting_keys = (
        "distance_checks", "index_queries", "index_rebuilds", "append_writes",
        "pool_rebuilds", "candidate_comparisons",
    )
    accounting = {key: 0 for key in accounting_keys}
    for diagnostics in field_diagnostics:
        raw = diagnostics.get("accounting", {})
        if isinstance(raw, dict):
            for key in accounting:
                if key == "append_writes":
                    # Every latent field writes each node exactly once, so the
                    # realization-level count is per-field (== n_nodes), not the
                    # sum over fields: summing would conflate field count with
                    # node count in the ``append_writes == len(frame)`` contract.
                    accounting[key] = max(accounting[key], int(raw.get(key, 0)))
                else:
                    accounting[key] += int(raw.get(key, 0))
    return index, seed, packed, {
        "accounting": accounting,
        "field_seeds": [int(item) for item in field_seeds],
        "latent_field_count": len(fields),
        "hard_conditioning_count": int(hard_categories.size),
    }


class PlurigaussianSim(SimulationAgentContract):
    """Simulate latent Gaussian fields and apply an immutable truncation rule.

    Args:
        variograms: one covariance model per Gaussian field.
        truncation_rule: rule mapping Gaussian fields to categories.
        property: output :class:`PropertyMetadata` (name = data column).
        neighbourhood: search-neighbourhood spec (``None`` = exhaustive).
        execution: realisations / seed / workers / budget policy.
        solve_policy: singular-system handling policy.
    """

    def __init__(
        self,
        variograms: Sequence[CovarianceModel],
        truncation_rule: Callable[..., object] | object,
        *,
        property: PropertyMetadata,
        neighbourhood: NeighbourhoodSpec,
        execution: SimulationExecutionConfig,
        solve_policy: SequentialSolvePolicy = SequentialSolvePolicy(),
        n_fields: int = 2,
        max_iter: int = 50,
        step: float = 0.3,
    ) -> None:
        models = tuple(variograms)
        if n_fields < 1 or len(models) != n_fields or any(
            not isinstance(item, CovarianceModel) for item in models
        ):
            raise GeomodelContractError(
                "variograms must contain one covariance model per latent field",
                object_name=type(self).__name__, field="variograms",
                expected=f"{n_fields} CovarianceModel records", actual=len(models),
            )
        if not isinstance(property, PropertyMetadata) or property.kind != "categorical":
            raise GeomodelContractError(
                "plurigaussian output property must be categorical",
                object_name=type(self).__name__, field="property",
                expected="categorical PropertyMetadata", actual=type(property).__name__,
            )
        if not isinstance(neighbourhood, NeighbourhoodSpec):
            raise GeomodelContractError(
                "PlurigaussianSim requires NeighbourhoodSpec",
                object_name=type(self).__name__, field="neighbourhood",
                expected="NeighbourhoodSpec", actual=type(neighbourhood).__name__,
            )
        if not isinstance(execution, SimulationExecutionConfig):
            raise GeomodelContractError(
                "PlurigaussianSim requires SimulationExecutionConfig",
                object_name=type(self).__name__, field="execution",
                expected="SimulationExecutionConfig", actual=type(execution).__name__,
            )
        if not isinstance(solve_policy, SequentialSolvePolicy):
            raise GeomodelContractError(
                "PlurigaussianSim requires SequentialSolvePolicy",
                object_name=type(self).__name__, field="solve_policy",
                expected="SequentialSolvePolicy", actual=type(solve_policy).__name__,
            )
        if callable(truncation_rule):
            owned_rule: Callable[..., object] | np.ndarray = truncation_rule
        else:
            lut = np.array(truncation_rule, dtype=np.int64, copy=True, order="C")
            if lut.ndim != 2 or n_fields != 2:
                raise GeomodelContractError(
                    "lookup-table truncation requires a 2-D table and two latent fields",
                    object_name=type(self).__name__, field="truncation_rule",
                    expected="2-D integer table with n_fields=2", actual=tuple(lut.shape),
                )
            owned_rule = np.frombuffer(lut.tobytes(), dtype=np.int64).reshape(lut.shape)
        if max_iter < 1 or not np.isfinite(step) or step <= 0.0:
            raise GeomodelContractError(
                "conditioning iteration controls are invalid",
                object_name=type(self).__name__, field="max_iter/step",
                expected="max_iter >= 1 and finite step > 0", actual=(max_iter, step),
            )
        for model in models:
            model.require_stationary_covariance(object_name=type(self).__name__)
        self.variograms = models
        self.truncation_rule = owned_rule
        self.property = property
        self.neighbourhood = neighbourhood
        self.execution = execution
        self.solve_policy = solve_policy
        self.n_fields = int(n_fields)
        self.max_iter = int(max_iter)
        self.step = float(step)

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
        if conditioning.hard_values is None:
            hard_coordinates = as_float_array(np.empty((0, targets.shape[1])))
            hard_categories = as_float_array(np.empty(0))
        else:
            present = ~np.ma.getmaskarray(conditioning.hard_values)
            hard_coordinates = as_float_array(conditioning.coordinates_m[present])
            hard_categories = as_float_array(
                np.asarray(np.ma.getdata(conditioning.hard_values)[present], dtype=np.float64)
            )
        seeds = derive_realization_seeds(self.execution.seed, self.execution.n_realizations)
        worker = partial(
            _plurigaussian_worker,
            targets=targets,
            variograms=self.variograms,
            rule=self.truncation_rule,
            property=self.property,
            neighbourhood=self.neighbourhood,
            execution=self.execution,
            solve_policy=self.solve_policy,
            hard_coordinates=hard_coordinates,
            hard_categories=hard_categories,
            max_iter=self.max_iter,
            step=self.step,
        )
        run = cast(RealizationRun, run_realizations(worker, seeds, self.execution))
        latent_metadata = PropertyMetadata("latent", "continuous", "1")
        frames = tuple(
            make_simulation_frame(
                geometry,
                np.asarray(item.result)[0],
                self.property,
                extras={
                    f"field_{field_index}": (
                        np.asarray(item.result)[field_index + 1],
                        PropertyMetadata(f"field_{field_index}", latent_metadata.kind, latent_metadata.unit),
                    )
                    for field_index in range(self.n_fields)
                },
            )
            for item in run.results
        )
        return assemble_ensemble(
            self.property,
            self.execution,
            run,
            frames,
            diagnostics={
                "algorithm": "PlurigaussianSim",
                "property": self.property.to_dict(),
                "latent_field_count": self.n_fields,
                "truncation_rule": "callable" if callable(self.truncation_rule) else "lookup_table",
                "neighbourhood": self.neighbourhood.to_dict(),
                "solve_policy": self.solve_policy.to_dict(),
                "conditioning": conditioning.diagnostics,
            },
        )

    def __repr__(self) -> str:
        return f"PlurigaussianSim(n_fields={self.n_fields})"
