"""Immutable execution configuration for the EM family (family template).

Cross-operator EXECUTION policy, how a forward solves, not what physics it
models. Scientific per-operator parameters (electrode pinning, chargeability
ceilings, quadrature sizes, MT modes) stay on the operator constructors.

Solver selection is dual-track, matching the platform convention
(``flow``'s ``FlowExecutionConfig.linear_solver`` id and ``wave``'s
``WaveBackendConfig.name``):

- ``linear_solver``: a registry id string (``'splu'`` / ``'krylov'`` /
  ``'cudss'``) for the common case;
- ``solver``: direct object injection of any
  :class:`~geobrain.core.linalg.SparseFactorSolver`, overriding the id
  (the escape hatch for custom or pre-configured backends).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass

from geobrain.core.linalg import SparseFactorSolver

from .errors import EMContractError

_LINEAR_SOLVER_IDS = ("splu", "krylov", "cudss")


@dataclass(frozen=True, slots=True)
class EMExecutionConfig:
    """Execution policy shared by the EM forward operators.

    Attributes:
        linear_solver: sparse-solver registry id, ``'splu'`` (SciPy LU,
            the default), ``'krylov'`` (complex Krylov, GPU-capable), or
            ``'cudss'`` (NVIDIA cuDSS; needs the optional wheel).
        solver: optional :class:`~geobrain.core.linalg.SparseFactorSolver`
            instance injected directly; overrides ``linear_solver``.
        use_closed_form_sigma_jacobian: use the closed-form conductivity
            Jacobian instead of autograd where an operator implements it
            (DC3D / MT3D / FDEM3D / TEM3D).
    """

    linear_solver: str = "splu"
    solver: SparseFactorSolver | None = None
    use_closed_form_sigma_jacobian: bool = False

    def __post_init__(self) -> None:
        if self.linear_solver not in _LINEAR_SOLVER_IDS:
            raise EMContractError(
                "EMExecutionConfig linear_solver must be a known registry id",
                details={"received": self.linear_solver},
                object_name="EMExecutionConfig",
                field="linear_solver",
                expected=" | ".join(_LINEAR_SOLVER_IDS),
                actual=self.linear_solver,
            )
        if self.solver is not None and not isinstance(self.solver, SparseFactorSolver):
            raise EMContractError(
                "EMExecutionConfig solver must implement SparseFactorSolver",
                details={"received_type": type(self.solver).__qualname__},
                object_name="EMExecutionConfig",
                field="solver",
                expected="SparseFactorSolver or None",
                actual=type(self.solver).__qualname__,
            )
        if self.use_closed_form_sigma_jacobian not in (True, False):
            raise EMContractError(
                "EMExecutionConfig use_closed_form_sigma_jacobian must be a bool",
                details={"received_type": type(self.use_closed_form_sigma_jacobian).__qualname__},
                object_name="EMExecutionConfig",
                field="use_closed_form_sigma_jacobian",
                expected="bool",
                actual=self.use_closed_form_sigma_jacobian,
            )

    def resolve_solver(self) -> SparseFactorSolver | None:
        """Return the injected solver, an id-built one, or ``None``.

        ``None`` means "use the operator's default" (SciPy LU), returned
        for the default ``'splu'`` id so operator behaviour is
        byte-identical to the pre-config era.
        """
        if self.solver is not None:
            return self.solver
        if self.linear_solver == "splu":
            return None
        if self.linear_solver == "krylov":
            from geobrain.core.linalg import KrylovSolver

            return KrylovSolver()
        from geobrain.core.linalg import CuDSSSolver  # lazy optional backend

        return CuDSSSolver()


__all__ = ["EMExecutionConfig"]
