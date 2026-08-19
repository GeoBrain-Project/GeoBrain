"""Convergence validation and structured failures for Helmholtz solves.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from geobrain.physics.wave.errors import WaveNumericsError

from .adjoint import solve_with_implicit_adjoint


@dataclass(frozen=True, slots=True)
class HelmholtzSolveResult:
    """One raw solver result before convergence acceptance."""

    solution: torch.Tensor
    status: str


def _solve_once(matrix: torch.Tensor, rhs: torch.Tensor) -> HelmholtzSolveResult:
    """Run the supported direct solver through the implicit-adjoint seam."""
    return HelmholtzSolveResult(
        solution=solve_with_implicit_adjoint(matrix, rhs), status="converged"
    )


def _normalized_residuals(
    matrix: torch.Tensor, solution: torch.Tensor, rhs: torch.Tensor
) -> tuple[float, tuple[float | None, ...]]:
    """Return the maximum and per-shot normalized residuals."""
    residual = (torch.sparse.mm(matrix, solution) - rhs).detach()
    detached_rhs = rhs.detach()
    residual_columns = residual.unsqueeze(1) if residual.ndim == 1 else residual
    rhs_columns = detached_rhs.unsqueeze(1) if detached_rhs.ndim == 1 else detached_rhs
    numerator = torch.linalg.vector_norm(residual_columns, dim=0)
    denominator = torch.linalg.vector_norm(rhs_columns, dim=0).clamp_min(
        torch.finfo(rhs.real.dtype).tiny
    )
    ratios = numerator / denominator
    finite_ratios = torch.isfinite(ratios)
    residual_by_shot = tuple(
        float(value) if bool(is_finite) else None
        for value, is_finite in zip(ratios, finite_ratios)
    )
    if not bool(finite_ratios.all()):
        return math.inf, residual_by_shot
    return float(ratios.max()), residual_by_shot


def solve_helmholtz_system(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    *,
    frequency_hz: float,
    relative_tolerance: float = 1.0e-9,
) -> torch.Tensor:
    """Solve and accept only a finite, converged, low-residual field."""
    try:
        result = _solve_once(matrix, rhs)
    except Exception as exc:
        raise WaveNumericsError(
            "Helmholtz direct solve raised an exception",
            object_name="Helmholtz2D",
            field="solve",
            actual={
                "solver": "direct-splu",
                "frequency_hz": float(frequency_hz),
                "status": "exception",
                "residual": None,
                "residual_by_shot": None,
            },
            hint="check velocity positivity, frequency sampling, and boundary settings",
        ) from exc
    solution = result.solution
    finite_solution = bool(torch.isfinite(solution.detach()).all())
    if finite_solution:
        residual, residual_by_shot = _normalized_residuals(matrix, solution, rhs)
    else:
        n_shot = 1 if rhs.ndim == 1 else rhs.shape[1]
        residual = math.inf
        residual_by_shot = (None,) * n_shot
    if result.status != "converged" or not finite_solution or not math.isfinite(residual) or residual > relative_tolerance:
        if result.status != "converged":
            status = result.status
        elif not finite_solution:
            status = "nonfinite-solution"
        elif not math.isfinite(residual):
            status = "nonfinite-residual"
        else:
            status = "residual-too-large"
        raise WaveNumericsError(
            "Helmholtz solve did not satisfy the convergence contract",
            object_name="Helmholtz2D",
            field="solve",
            expected={"status": "converged", "relative_residual_max": relative_tolerance},
            actual={
                "solver": "direct-splu",
                "frequency_hz": float(frequency_hz),
                "status": status,
                "residual": residual if math.isfinite(residual) else None,
                "residual_by_shot": residual_by_shot,
            },
            hint="refine the mesh, change the boundary, or avoid a resonant frequency",
        )
    return solution


__all__ = ["HelmholtzSolveResult", "solve_helmholtz_system"]
