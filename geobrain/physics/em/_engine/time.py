"""Time-domain key helpers for exact repeated-step factor reuse.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import inspect
import json
import math

import torch

from geobrain.core.linalg import ScipySpluSolver, SparseFactorSolver
from geobrain.physics.em.errors import EMContractError

from .contracts import (
    AssemblyCacheKey,
    EMExecutionDiagnostics,
    FactorCacheKey,
    RecordingDiagnostics,
    exact_float_token,
)
from .recording import RecordingPolicy


@dataclass(frozen=True, slots=True)
class SparseSolverSettings:
    """Exact cache-relevant identity of one resolved sparse solver."""

    backend: str
    tolerance: float
    max_iterations: int

    def to_dict(self) -> dict[str, str | int]:
        """Return exact JSON-safe controls for execution metadata."""
        return {
            "backend": self.backend,
            "solver_tolerance": exact_float_token(self.tolerance),
            "solver_max_iterations": self.max_iterations,
        }


def solver_execution_metadata(
    diagnostics: EMExecutionDiagnostics,
    settings: SparseSolverSettings,
) -> dict[str, dict[str, int | str]]:
    """Attach auditable solver identity and controls to execution counters."""
    execution: dict[str, int | str] = {key: value for key, value in diagnostics.to_dict().items()}
    execution.update(settings.to_dict())
    return {"em_execution": execution}


def recording_execution_metadata(
    diagnostics: RecordingDiagnostics,
    policy: RecordingPolicy,
) -> dict[str, object]:
    """Return deterministic recording counters and their resolved policy."""
    if type(diagnostics) is not RecordingDiagnostics:
        raise EMContractError(
            "recording diagnostics must be an exact RecordingDiagnostics",
            details={"received_type": type(diagnostics).__qualname__},
            object_name="recording_execution_metadata",
            field="diagnostics",
            expected="RecordingDiagnostics",
            actual=type(diagnostics).__qualname__,
        )
    if type(policy) is not RecordingPolicy:
        raise EMContractError(
            "recording policy must be an exact RecordingPolicy",
            details={"received_type": type(policy).__qualname__},
            object_name="recording_execution_metadata",
            field="policy",
            expected="RecordingPolicy",
            actual=type(policy).__qualname__,
        )
    return {
        "em_recording": diagnostics.to_dict(),
        "recording_policy": policy.to_dict(),
    }


def _config_value(value: object) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise EMContractError(
                "sparse solver configuration must be finite",
                details={"value": str(value)},
                object_name="solver_settings",
                field="solver",
                expected="finite constructor configuration",
                actual=str(value),
            )
        return {"kind": "float", "hex": value.hex()}
    if isinstance(value, (torch.device, torch.dtype)):
        return {
            "kind": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": str(value),
        }
    if isinstance(value, bytes):
        return {"kind": "bytes", "hex": value.hex()}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_config_value(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_config_value(item) for item in value]}
    if isinstance(value, Mapping):
        encoded_items = [(_config_value(key), _config_value(item)) for key, item in value.items()]
        encoded_items.sort(
            key=lambda pair: json.dumps(
                pair[0],
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return {
            "kind": "mapping",
            "mapping_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "items": [[key, item] for key, item in encoded_items],
        }
    raise EMContractError(
        "sparse solver configuration contains an unsupported value",
        details={"received_type": f"{type(value).__module__}.{type(value).__qualname__}"},
        object_name="solver_settings",
        field="solver",
        expected="deterministically serializable constructor configuration",
        actual=type(value).__qualname__,
    )


def _solver_constructor_config(solver: SparseFactorSolver) -> dict[str, object]:
    """Encode only stable state named by the solver's constructor contract."""
    constructor = type(solver).__init__
    if constructor is object.__init__:
        return {}
    try:
        parameters = tuple(inspect.signature(constructor).parameters.values())
    except (TypeError, ValueError) as exc:
        raise EMContractError(
            "sparse solver constructor signature is not inspectable",
            details={"solver_type": f"{type(solver).__module__}.{type(solver).__qualname__}"},
            object_name="solver_settings",
            field="solver",
            expected="an inspectable constructor with named configuration parameters",
            actual="uninspectable constructor",
        ) from exc

    constructor_fields: dict[str, object] = {}
    for parameter in parameters:
        if parameter.name == "self":
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise EMContractError(
                "sparse solver variadic constructor configuration cannot form a stable identity",
                details={"parameter": parameter.name, "kind": parameter.kind.name},
                object_name="solver_settings",
                field="solver",
                expected="named constructor configuration parameters",
                actual=parameter.kind.name,
            )

        stored_values: dict[str, object] = {}
        if hasattr(solver, parameter.name):
            stored_values["public"] = _config_value(getattr(solver, parameter.name))
        private_name = f"_{parameter.name}"
        if hasattr(solver, private_name):
            stored_values["private"] = _config_value(getattr(solver, private_name))
        if not stored_values:
            raise EMContractError(
                "sparse solver constructor configuration is not persisted for stable identity",
                details={"parameter": parameter.name},
                object_name="solver_settings",
                field="solver",
                expected=f"solver.{parameter.name} or solver.{private_name}",
                actual="missing",
            )
        constructor_fields[parameter.name] = {
            "kind": "constructor_parameter",
            "stored_values": stored_values,
        }
    return constructor_fields


def resolve_engine_solver(
    solver: SparseFactorSolver | None = None,
) -> SparseFactorSolver:
    """Validate an injected solver or create one explicit SciPy LU instance."""
    resolved = ScipySpluSolver() if solver is None else solver
    if not isinstance(resolved, SparseFactorSolver):
        raise EMContractError(
            "inductive solver must implement SparseFactorSolver",
            details={"received_type": type(resolved).__qualname__},
            object_name="resolve_engine_solver",
            field="solver",
            expected="SparseFactorSolver or None",
            actual=type(resolved).__qualname__,
        )
    return resolved


def solver_settings(solver: SparseFactorSolver) -> SparseSolverSettings:
    """Return deterministic backend, tolerance, and iteration cache controls."""
    if not isinstance(solver, SparseFactorSolver):
        raise EMContractError(
            "inductive solver must implement SparseFactorSolver",
            details={"received_type": type(solver).__qualname__},
            object_name="solver_settings",
            field="solver",
            expected="SparseFactorSolver",
            actual=type(solver).__qualname__,
        )
    constructor_fields = _solver_constructor_config(solver)

    if type(solver) is ScipySpluSolver:
        backend = "scipy-splu"
    else:
        solver_type = f"{type(solver).__module__}.{type(solver).__qualname__}"
        encoded = json.dumps(
            constructor_fields,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        backend = f"{solver_type}:{encoded}"

    raw_tolerance = getattr(solver, "rtol", getattr(solver, "tol", 0.0))
    try:
        tolerance = float(raw_tolerance)
    except (TypeError, ValueError) as exc:
        raise EMContractError(
            "sparse solver tolerance must be numeric",
            details={"received_type": type(raw_tolerance).__qualname__},
            object_name="solver_settings",
            field="tolerance",
            expected="finite non-negative float",
            actual=type(raw_tolerance).__qualname__,
        ) from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise EMContractError(
            "sparse solver tolerance must be finite and non-negative",
            details={"value": str(tolerance)},
            object_name="solver_settings",
            field="tolerance",
            expected="finite float >= 0",
            actual=tolerance,
        )
    raw_max_iterations = getattr(solver, "maxiter", 0)
    max_iterations = 0 if raw_max_iterations is None else raw_max_iterations
    if type(max_iterations) is not int or max_iterations < 0:
        raise EMContractError(
            "sparse solver max iterations must be a non-negative int or None",
            details={"value": str(raw_max_iterations)},
            object_name="solver_settings",
            field="max_iterations",
            expected="non-negative int or None",
            actual=raw_max_iterations,
        )
    return SparseSolverSettings(backend, tolerance, max_iterations)


def solver_factor_key(
    assembly_key: AssemblyCacheKey,
    solver: SparseFactorSolver,
) -> FactorCacheKey:
    """Build a factor key from the same resolved solver used to factor."""
    settings = solver_settings(solver)
    if assembly_key.backend != settings.backend:
        raise EMContractError(
            "assembly backend must match the resolved factor solver",
            details={
                "assembly_backend": assembly_key.backend,
                "solver_backend": settings.backend,
            },
            object_name="solver_factor_key",
            field="assembly.backend",
            expected=settings.backend,
            actual=assembly_key.backend,
        )
    return FactorCacheKey(
        assembly=assembly_key,
        solver_tolerance=exact_float_token(settings.tolerance),
        solver_max_iterations=settings.max_iterations,
    )


__all__ = [
    "SparseSolverSettings",
    "resolve_engine_solver",
    "recording_execution_metadata",
    "solver_factor_key",
    "solver_execution_metadata",
    "solver_settings",
]
