"""Dense-factor Gaussian simulation with governed covariance admissibility.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import numpy as np

from ...conditioning import ConditioningSet, normalize_conditioning
from ...frames import GeoFrame, PropertyMetadata
from ...frames._arrays import FloatArray, as_float_array
from ...errors import GeomodelContractError, GeomodelNumericsError
from .._domain import derive_realization_seeds
from ..estimation.covariance_matrix import covariance_matrix
from ..models.covariance import CovarianceModel
from ._grid_utils import snap_to_hard_data
from ._parallel import RealizationRun, run_realizations
from .execution import SimulationExecutionConfig
from .agent_contract import SimulationAgentContract
from .results import SimulationEnsemble
from .sequential import assemble_ensemble, hard_conditioning, make_simulation_frame, resolve_sequential_domain

__all__ = ["DenseFactorPolicy", "LUSIM"]


@dataclass(frozen=True, slots=True)
class DenseFactorPolicy:
    """Explicitly permitted dense-factor regularisation.

    Attributes:
        jitter_relative: relative jitter ladder for the factorization.
        minimum_eigenvalue_relative: eigenvalue floor accepted.
    """

    jitter_relative: tuple[float, ...] = ()
    minimum_eigenvalue_relative: float = -1.0e-12

    def __post_init__(self) -> None:
        jitter = tuple(float(item) for item in self.jitter_relative)
        if any(not math.isfinite(item) or item <= 0.0 for item in jitter):
            raise GeomodelContractError(
                "dense-factor jitter schedule must contain finite positive values",
                object_name=type(self).__name__, field="jitter_relative",
                expected="tuple of finite values > 0", actual=jitter,
            )
        minimum = float(self.minimum_eigenvalue_relative)
        if not math.isfinite(minimum) or minimum > 0.0:
            raise GeomodelContractError(
                "minimum eigenvalue tolerance must be finite and non-positive",
                object_name=type(self).__name__, field="minimum_eigenvalue_relative",
                expected="finite value <= 0", actual=minimum,
            )
        object.__setattr__(self, "jitter_relative", jitter)
        object.__setattr__(self, "minimum_eigenvalue_relative", minimum)

    def to_dict(self) -> dict[str, object]:
        return {
            "jitter_relative": list(self.jitter_relative),
            "minimum_eigenvalue_relative": self.minimum_eigenvalue_relative,
        }


def _factor_covariance(
    covariance: FloatArray,
    *,
    sill: float,
    policy: DenseFactorPolicy,
    object_name: str,
) -> tuple[FloatArray, float]:
    matrix = as_float_array(0.5 * (covariance + covariance.T))
    attempts = (0.0, *policy.jitter_relative)
    for jitter_relative in attempts:
        candidate = matrix if jitter_relative == 0.0 else as_float_array(
            matrix + np.eye(matrix.shape[0], dtype=np.float64) * sill * jitter_relative
        )
        try:
            return as_float_array(np.linalg.cholesky(candidate)), float(jitter_relative)
        except np.linalg.LinAlgError:
            continue
    minimum = float(np.linalg.eigvalsh(matrix)[0])
    threshold = sill * policy.minimum_eigenvalue_relative
    raise GeomodelNumericsError(
        "dense covariance is not admissible under the configured factor policy",
        object_name=object_name,
        field="covariance",
        expected={"minimum_eigenvalue": f">= {threshold}", "jitter_relative": list(policy.jitter_relative)},
        actual={"minimum_eigenvalue": minimum, "attempted_jitter_relative": list(attempts)},
    )


def _lusim_worker(
    index: int,
    seed: int,
    *,
    factor: FloatArray,
    mean: FloatArray,
    targets: FloatArray,
    cond_coords: FloatArray,
    cond_values: FloatArray,
    jitter_relative: float,
) -> tuple[int, int, FloatArray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    values = as_float_array(mean + factor @ rng.standard_normal(mean.size))
    snap_to_hard_data(values, targets, cond_coords, cond_values)
    return index, seed, values, {
        "accepted_jitter_relative": jitter_relative,
        "matrix_order": int(mean.size),
    }


class LUSIM(SimulationAgentContract):
    """Gaussian simulation by an explicitly governed dense factor.

    Args:
        model: covariance model.
        property: output :class:`PropertyMetadata` (name = data column).
        execution: realisations / seed / workers / budget policy.
        factor_policy: dense-factorization robustness policy.
        mean: stationary mean added to the draws.
    """

    def __init__(
        self,
        model: CovarianceModel,
        *,
        property: PropertyMetadata,
        execution: SimulationExecutionConfig,
        factor_policy: DenseFactorPolicy = DenseFactorPolicy(),
        mean: float = 0.0,
    ) -> None:
        if not isinstance(model, CovarianceModel):
            raise GeomodelContractError(
                "LUSIM requires a covariance model",
                object_name=type(self).__name__, field="model",
                expected="CovarianceModel", actual=type(model).__name__,
            )
        if not isinstance(property, PropertyMetadata) or property.kind != "continuous":
            raise GeomodelContractError(
                "LUSIM property must be continuous",
                object_name=type(self).__name__, field="property",
                expected="continuous PropertyMetadata", actual=type(property).__name__,
            )
        if not isinstance(execution, SimulationExecutionConfig):
            raise GeomodelContractError(
                "LUSIM requires SimulationExecutionConfig",
                object_name=type(self).__name__, field="execution",
                expected="SimulationExecutionConfig", actual=type(execution).__name__,
            )
        if not isinstance(factor_policy, DenseFactorPolicy):
            raise GeomodelContractError(
                "LUSIM requires DenseFactorPolicy",
                object_name=type(self).__name__, field="factor_policy",
                expected="DenseFactorPolicy", actual=type(factor_policy).__name__,
            )
        resolved_mean = float(mean)
        if not math.isfinite(resolved_mean):
            raise GeomodelContractError(
                "LUSIM mean must be finite",
                object_name=type(self).__name__, field="mean",
                expected="finite float", actual=mean,
            )
        model.require_stationary_covariance(object_name=type(self).__name__)
        self.model = model
        self.property = property
        self.execution = execution
        self.factor_policy = factor_policy
        self.mean = resolved_mean

    def _preflight(self, nodes: int, conditioning: int) -> None:
        # covariance, factor, conditional work/output and active worker draws
        matrix_entries = 3 * nodes * nodes + conditioning * conditioning + 3 * nodes * conditioning
        required = 8 * (matrix_entries + nodes * min(self.execution.workers, self.execution.n_realizations))
        self.execution.require_budget(required, component="LUSIM dense covariance and factor")

    def __call__(
        self,
        data: GeoFrame | ConditioningSet | None,
        domain: Any,
    ) -> SimulationEnsemble:
        geometry, targets = resolve_sequential_domain(
            domain, execution=self.execution, object_name=type(self).__name__
        )
        conditioning = normalize_conditioning(data, geometry, self.property)
        cond_coords, cond_values = hard_conditioning(conditioning)
        self._preflight(targets.shape[0], cond_coords.shape[0])
        covariance_grid = covariance_matrix(self.model, targets, targets)
        mean_vector = as_float_array(np.full(targets.shape[0], self.mean, dtype=np.float64))
        solve_residual = 0.0
        if cond_coords.shape[0]:
            covariance_data = covariance_matrix(self.model, cond_coords, cond_coords)
            covariance_cross = covariance_matrix(self.model, targets, cond_coords)
            try:
                weights = as_float_array(np.linalg.solve(covariance_data, covariance_cross.T).T)
            except np.linalg.LinAlgError as exc:
                raise GeomodelNumericsError(
                    "LUSIM conditioning covariance is singular",
                    object_name=type(self).__name__, field="conditioning",
                    expected="nonsingular covariance matrix", actual="singular",
                ) from exc
            residual_matrix = covariance_data @ weights.T - covariance_cross.T
            solve_residual = float(np.linalg.norm(residual_matrix, ord=np.inf))
            mean_vector = as_float_array(self.mean + weights @ (cond_values - self.mean))
            covariance_grid = as_float_array(covariance_grid - weights @ covariance_cross.T)
        factor, jitter = _factor_covariance(
            covariance_grid,
            sill=float(self.model.sill),
            policy=self.factor_policy,
            object_name=type(self).__name__,
        )
        seeds = derive_realization_seeds(self.execution.seed, self.execution.n_realizations)
        worker = partial(
            _lusim_worker,
            factor=factor,
            mean=mean_vector,
            targets=targets,
            cond_coords=cond_coords,
            cond_values=cond_values,
            jitter_relative=jitter,
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
                "algorithm": "LUSIM",
                "property": self.property.to_dict(),
                "factor_policy": self.factor_policy.to_dict(),
                "accepted_jitter_relative": jitter,
                "conditioning_solve_residual": solve_residual,
                "conditioning": conditioning.diagnostics,
            },
        )

    def __repr__(self) -> str:
        return f"LUSIM(model={self.model!r}, mean={self.mean})"
