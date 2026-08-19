"""Immutable contracts for inductive-EM assembly and factor reuse.

The keys in this module are deliberately value based.  In particular, sample
values use :meth:`float.hex` instead of decimal rounding, so adjacent binary64
frequencies or time steps can never alias.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from geobrain.physics.em.errors import EMContractError


def _nonempty_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise EMContractError(
            f"{field} must be a non-empty string",
            details={"field": field, "received_type": type(value).__qualname__},
            object_name="AssemblyCacheKey",
            field=field,
            expected="non-empty str",
            actual=value,
        )
    return value


def exact_float_token(value: float) -> str:
    """Return the exact, round-trippable binary64 token for ``value``."""
    if type(value) is not float:
        raise EMContractError(
            "exact cache sample must be a built-in float",
            details={"received_type": type(value).__qualname__},
            object_name="exact_float_token",
            field="value",
            expected="finite built-in float",
            actual=type(value).__qualname__,
        )
    if not math.isfinite(value):
        raise EMContractError(
            "exact cache sample must be finite",
            details={"value": str(value)},
            object_name="exact_float_token",
            field="value",
            expected="finite float",
            actual=value,
        )
    return value.hex()


def _exact_float_text(value: object, *, field: str) -> str:
    token = _nonempty_text(value, field=field)
    try:
        parsed = float.fromhex(token)
    except ValueError as exc:
        raise EMContractError(
            f"{field} must be a canonical hexadecimal float token",
            details={"field": field, "value": token},
            object_name="InductiveCacheKey",
            field=field,
            expected="finite float.hex() token",
            actual=token,
        ) from exc
    if not math.isfinite(parsed) or parsed.hex() != token:
        raise EMContractError(
            f"{field} must be a canonical finite hexadecimal float token",
            details={"field": field, "value": token},
            object_name="InductiveCacheKey",
            field=field,
            expected="finite float.hex() token",
            actual=token,
        )
    return token


@dataclass(frozen=True, slots=True)
class AssemblyCacheKey:
    """Complete identity of one inductive numerical matrix."""

    formulation_version: str
    mesh_fingerprint: str
    material_version: str
    boundary: str
    sample_value: str
    dtype: str
    device: str
    backend: str
    requires_gradient: bool

    def __post_init__(self) -> None:
        for field in (
            "formulation_version",
            "mesh_fingerprint",
            "material_version",
            "boundary",
            "dtype",
            "device",
            "backend",
        ):
            _nonempty_text(getattr(self, field), field=field)
        _exact_float_text(self.sample_value, field="sample_value")
        if type(self.requires_gradient) is not bool:
            raise EMContractError(
                "requires_gradient must be bool",
                details={"received_type": type(self.requires_gradient).__qualname__},
                object_name=type(self).__name__,
                field="requires_gradient",
                expected=bool,
                actual=type(self.requires_gradient),
            )


@dataclass(frozen=True, slots=True)
class FactorCacheKey:
    """Assembly identity plus solver controls that alter the factor."""

    assembly: AssemblyCacheKey
    solver_tolerance: str
    solver_max_iterations: int

    def __post_init__(self) -> None:
        if not isinstance(self.assembly, AssemblyCacheKey):
            raise EMContractError(
                "assembly must be an AssemblyCacheKey",
                details={"received_type": type(self.assembly).__qualname__},
                object_name=type(self).__name__,
                field="assembly",
                expected="AssemblyCacheKey",
                actual=type(self.assembly).__qualname__,
            )
        _exact_float_text(self.solver_tolerance, field="solver_tolerance")
        if type(self.solver_max_iterations) is not int or self.solver_max_iterations < 0:
            raise EMContractError(
                "solver_max_iterations must be a non-negative int",
                details={"value": str(self.solver_max_iterations)},
                object_name=type(self).__name__,
                field="solver_max_iterations",
                expected="non-negative int",
                actual=self.solver_max_iterations,
            )


@dataclass(slots=True)
class EMExecutionDiagnostics:
    """Deterministic counters for one explicit inductive execution scope."""

    assembly_count: int = 0
    factorization_count: int = 0
    solve_count: int = 0
    assembly_cache_hits: int = 0
    factor_cache_hits: int = 0
    projection_count: int = 0
    projected_element_count: int = 0
    rhs_count: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return a detached metadata snapshot in declaration order."""
        return asdict(self)


@dataclass(slots=True)
class RecordingDiagnostics:
    """Deterministic bounded-recording counters for one TEM recurrence."""

    n_steps: int = 0
    n_gates: int = 0
    live_state_count_max: int = 0
    retained_state_count: int = 0
    checkpoint_count: int = 0
    recomputed_step_count: int = 0
    recording_bytes: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return detached integer counters in declaration order."""
        return asdict(self)


__all__ = [
    "AssemblyCacheKey",
    "EMExecutionDiagnostics",
    "FactorCacheKey",
    "RecordingDiagnostics",
    "exact_float_token",
]
