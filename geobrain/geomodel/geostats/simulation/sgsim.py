"""Indexed Sequential Gaussian Simulation (SGSIM).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Literal, cast

import numpy as np

from ...conditioning import ConditioningSet, normalize_conditioning
from ...frames import GeoFrame, PropertyMetadata
from ...errors import GeomodelContractError
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
    hard_conditioning,
    make_simulation_frame,
    resolve_sequential_domain,
    simulate_gaussian_kernel,
)

__all__ = ["SGSIM"]


def _sgsim_worker(
    index: int,
    seed: int,
    *,
    cond_coords: np.ndarray,
    cond_values: np.ndarray,
    targets: np.ndarray,
    model: CovarianceModel,
    neighbourhood: NeighbourhoodSpec,
    execution: SimulationExecutionConfig,
    solve_policy: SequentialSolvePolicy,
    ktype: Literal[0, 1],
    mean: float,
) -> tuple[int, int, np.ndarray, dict[str, object]]:
    result = simulate_gaussian_kernel(
        cond_coords,
        cond_values,
        targets,
        model,
        neighbourhood=neighbourhood,
        execution=execution,
        solve_policy=solve_policy,
        rng=np.random.default_rng(seed),
        ktype=ktype,
        mean=mean,
        object_name="SGSIM",
    )
    return index, seed, result.values, dict(result.diagnostics)


class SGSIM(SimulationAgentContract):
    """Normal-score sequential Gaussian simulator with explicit contracts.

    Args:
        model: covariance model (normal-score space).
        property: output :class:`PropertyMetadata` (name = data column).
        neighbourhood: search-neighbourhood spec (``None`` = exhaustive).
        execution: realisations / seed / workers / budget policy.
        solve_policy: singular-system handling policy.
        ktype: 0 = simple kriging, 1 = ordinary kriging.
        mean: stationary mean for ``ktype=0``.
    """

    def __init__(
        self,
        model: CovarianceModel,
        *,
        property: PropertyMetadata,
        neighbourhood: NeighbourhoodSpec,
        execution: SimulationExecutionConfig,
        solve_policy: SequentialSolvePolicy = SequentialSolvePolicy(),
        ktype: Literal[0, 1] = 0,
        mean: float = 0.0,
    ) -> None:
        if not isinstance(model, CovarianceModel):
            raise GeomodelContractError(
                "SGSIM requires a covariance model",
                object_name=type(self).__name__,
                field="model",
                expected="CovarianceModel",
                actual=type(model).__name__,
            )
        if not isinstance(property, PropertyMetadata) or property.kind != "continuous":
            raise GeomodelContractError(
                "SGSIM property must be continuous",
                object_name=type(self).__name__,
                field="property",
                expected="continuous PropertyMetadata",
                actual=getattr(property, "kind", type(property).__name__),
            )
        if not isinstance(neighbourhood, NeighbourhoodSpec):
            raise GeomodelContractError(
                "SGSIM requires NeighbourhoodSpec",
                object_name=type(self).__name__,
                field="neighbourhood",
                expected="NeighbourhoodSpec",
                actual=type(neighbourhood).__name__,
            )
        if not isinstance(execution, SimulationExecutionConfig):
            raise GeomodelContractError(
                "SGSIM requires SimulationExecutionConfig",
                object_name=type(self).__name__,
                field="execution",
                expected="SimulationExecutionConfig",
                actual=type(execution).__name__,
            )
        if not isinstance(solve_policy, SequentialSolvePolicy):
            raise GeomodelContractError(
                "SGSIM requires SequentialSolvePolicy",
                object_name=type(self).__name__,
                field="solve_policy",
                expected="SequentialSolvePolicy",
                actual=type(solve_policy).__name__,
            )
        if ktype not in (0, 1):
            raise GeomodelContractError(
                "SGSIM ktype must be simple or ordinary kriging",
                object_name=type(self).__name__,
                field="ktype",
                expected="0 or 1",
                actual=ktype,
            )
        resolved_mean = float(mean)
        if not math.isfinite(resolved_mean):
            raise GeomodelContractError(
                "SGSIM mean must be finite",
                object_name=type(self).__name__,
                field="mean",
                expected="finite float",
                actual=mean,
            )
        model.require_stationary_covariance(object_name=type(self).__name__)
        self.model = model
        self.property = property
        self.neighbourhood = neighbourhood
        self.execution = execution
        self.solve_policy = solve_policy
        self.ktype = cast(Literal[0, 1], int(ktype))
        self.mean = resolved_mean

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
        cond_coords, cond_values = hard_conditioning(conditioning)
        seeds = derive_realization_seeds(
            self.execution.seed,
            self.execution.n_realizations,
        )
        worker = partial(
            _sgsim_worker,
            cond_coords=cond_coords,
            cond_values=cond_values,
            targets=targets,
            model=self.model,
            neighbourhood=self.neighbourhood,
            execution=self.execution,
            solve_policy=self.solve_policy,
            ktype=self.ktype,
            mean=self.mean,
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
                "algorithm": "SGSIM",
                "property": self.property.to_dict(),
                "neighbourhood": self.neighbourhood.to_dict(),
                "solve_policy": self.solve_policy.to_dict(),
                "conditioning": conditioning.diagnostics,
            },
        )

    def __repr__(self) -> str:
        return f"SGSIM(model={self.model!r}, ktype={self.ktype}, mean={self.mean})"
