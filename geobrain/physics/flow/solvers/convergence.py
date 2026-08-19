"""
CNV / MB convergence criteria (industry-standard).

A bare ``|R|_∞ < tol`` test in surface-volume rate (STB/day) is grid- and
rate-dependent: the same tolerance over-solves a fine grid and under-solves a
coarse one. Reservoir simulators instead use two **dimensionless** measures of
the conservation error:

* **CNV** (local): the largest per-cell normalized residual::

      CNV_α = B̄_α · dt · max_c |R_α,c| / pv_c

* **MB** (global, material balance): the summed residual over the field::

      MB_α  = B̄_α · dt · |Σ_c R_α,c| / Σ_c pv_c

where ``R_α,c`` is the surface-volume residual [STB/day] of phase α in cell c,
``pv_c = V_c·φ_c`` the **reservoir** pore volume [m³], and
``B̄_α`` the mean FVF. (A mass-unit formulation carries a surface density ``ρ_S``
and ``r``; GeoBrain's residual is already a surface volume, so ``r/ρ_S → R`` and the
``ρ_S`` cancels, leaving ``B·dt·R/pv``, dimensionless = reservoir volume per pore
volume.) Convergence is ``max_α CNV_α < tol_cnv`` **and** ``max_α MB_α < tol_mb``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence
import math

import torch

from ..errors import FlowContractError

# Industry-standard defaults.
TOL_CNV = 1.0e-3
TOL_MB = 1.0e-7


def cnv_mb_errors(
    residual_phases: Sequence[torch.Tensor],
    pore_volume: torch.Tensor,
    fvf_phases: Sequence[torch.Tensor],
    dt: float,
) -> tuple[list[float], list[float]]:
    """Per-phase ``(CNV, MB)`` convergence errors.

    Args:
        residual_phases: per-phase surface-volume residual ``R_α`` [m³/s],
            each shape ``(n_cells,)``.
        pore_volume: **reservoir** pore volume per cell ``V·φ/β`` [RB].
        fvf_phases: per-phase FVF ``B_α`` per cell (same order as the residuals).
        dt: time step [s].

    Returns: ``(cnv, mb)``, two lists of floats, one entry per phase.
    """
    pv_t = pore_volume.sum()
    cnv: list[float] = []
    mb: list[float] = []
    for r_a, b_a in zip(residual_phases, fvf_phases):
        scale = b_a.mean() * float(dt)  # B̄_α · dt  (ρ_S cancels)
        cnv.append(float(scale * (r_a.abs() / pore_volume).max()))
        mb.append(float(scale * r_a.sum().abs() / pv_t))
    return cnv, mb


class _ConvergenceModel(Protocol):
    def convergence_inputs(
        self,
        residual: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[Sequence[torch.Tensor], torch.Tensor, Sequence[torch.Tensor]]: ...


class _GridWithVolumes(Protocol):
    def _cell_volumes_view(self) -> torch.Tensor: ...


class _RockWithPorosity(Protocol):
    def porosity_at_pressure(self, pressure: torch.Tensor) -> torch.Tensor: ...


class CNVMBCriterion:
    """CNV/MB convergence checker bound to a flow model.

    The model must provide ``convergence_inputs(residual, state) →
    (residual_phases, pore_volume, fvf_phases)``. Use :meth:`closure` to build a
    ``(residual, state) → bool`` test for :meth:`NewtonSolver.solve`.
    """

    def __init__(
        self,
        model: _ConvergenceModel,
        dt: float,
        tol_cnv: float = TOL_CNV,
        tol_mb: float = TOL_MB,
    ) -> None:
        self.model = model
        values = {"dt": dt, "tol_cnv": tol_cnv, "tol_mb": tol_mb}
        for field, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise FlowContractError(
                    f"{field} must be positive and finite",
                    object_name=type(self).__name__,
                    field=field,
                    expected="finite value > 0",
                    actual=value,
                )
        self.dt = float(dt)
        self.tol_cnv = float(tol_cnv)
        self.tol_mb = float(tol_mb)
        self.last_errors: dict[str, list[float]] | None = None

    def errors(self, residual: torch.Tensor, state: torch.Tensor) -> dict[str, list[float]]:
        r_ph, pv, b_ph = self.model.convergence_inputs(residual, state)
        cnv, mb = cnv_mb_errors(r_ph, pv, b_ph, self.dt)
        result = {"cnv": cnv, "mb": mb}
        self.last_errors = result
        return result

    def converged(self, residual: torch.Tensor, state: torch.Tensor) -> bool:
        e = self.errors(residual, state)
        return max(e["cnv"]) < self.tol_cnv and max(e["mb"]) < self.tol_mb

    def closure(self) -> Callable[[torch.Tensor, torch.Tensor], bool]:
        def check(residual: torch.Tensor, state: torch.Tensor) -> bool:
            return self.converged(residual, state)

        check.flow_criterion = self  # type: ignore[attr-defined]
        return check


def reservoir_pore_volume(
    grid: _GridWithVolumes,
    rock: _RockWithPorosity,
    p: torch.Tensor,
) -> torch.Tensor:
    """Reservoir pore volume per cell ``V·φ(p)/β`` [RB]."""
    return grid._cell_volumes_view() * rock.porosity_at_pressure(p)


__all__ = ["TOL_CNV", "TOL_MB", "cnv_mb_errors", "CNVMBCriterion", "reservoir_pore_volume"]
