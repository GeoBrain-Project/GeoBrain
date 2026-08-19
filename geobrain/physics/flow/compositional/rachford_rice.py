"""
Rachford-Rice: vapor mole fraction ``V`` for given K-values and feed composition.

Given equilibrium ratios ``K_i = y_i/x_i`` and feed mole fractions ``z_i``, the
vapor fraction ``V`` solves the Rachford-Rice equation::

    RR(V) = Σ_i  z_i·(K_i − 1) / (1 + V·(K_i − 1))  =  0

``RR`` is monotonically decreasing in ``V`` with poles at ``V = 1/(1 − K_i)``; the
physical root lies in ``(1/(1 − K_max), 1/(1 − K_min))``. We solve with a
bracket-safeguarded Newton (bisecting whenever a Newton step leaves the bracket),
batched over the leading axes (e.g. one row per grid cell), then take one
implicit-function-theorem cleanup step so ``∂V/∂K`` and ``∂V/∂z`` flow through
autograd exactly. ``V`` outside ``[0, 1]`` is a *negative flash* result (the
mixture is single-phase); the stability test, not this solver, makes that call.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
import torch

from ....core import GeoBrainError
from ..errors import FlowContractError


def rachford_rice_residual(V: torch.Tensor, K: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """``RR(V) = Σ_i z_i(K_i−1)/(1+V(K_i−1))`` (sum over the component axis)."""
    dK = K - 1.0
    return (z * dK / (1.0 + V.unsqueeze(-1) * dK)).sum(dim=-1)


def _rr_dV(V: torch.Tensor, K: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    dK = K - 1.0
    return -(z * dK * dK / (1.0 + V.unsqueeze(-1) * dK) ** 2).sum(dim=-1)


def solve_rachford_rice(
    K: torch.Tensor,
    z: torch.Tensor,
    *,
    tol: float = 1e-12,
    maxiter: int = 200,
) -> torch.Tensor:
    """Vapor mole fraction ``V`` for K-values ``K`` and feed ``z``.

    ``K`` and ``z`` are ``(..., n_components)``; returns ``V`` of shape ``(...)``,
    differentiable w.r.t. ``K`` and ``z`` (implicit-function-theorem cleanup).
    """
    if (
        not isinstance(K, torch.Tensor)
        or not K.is_floating_point()
        or not isinstance(z, torch.Tensor)
        or not z.is_floating_point()
    ):
        raise FlowContractError(
            "K and z must be floating tensors",
            object_name="solve_rachford_rice",
            field="K/z",
            expected="floating torch.Tensor values",
            actual=(type(K).__name__, type(z).__name__),
        )
    if K.shape != z.shape:
        raise GeoBrainError(f"K and z must share shape; got {tuple(K.shape)} vs {tuple(z.shape)}")
    if K.ndim < 1 or K.shape[-1] < 2:
        raise FlowContractError(
            "K and z need a component axis",
            object_name="solve_rachford_rice",
            field="K/z.shape",
            expected="[..., component] with at least two components",
            actual=tuple(K.shape),
        )
    if K.dtype != z.dtype:
        raise FlowContractError(
            "K and z dtype must match",
            object_name="solve_rachford_rice",
            field="dtype",
            expected=str(z.dtype),
            actual=str(K.dtype),
        )
    if K.device != z.device:
        raise FlowContractError(
            "K and z device must match",
            object_name="solve_rachford_rice",
            field="device",
            expected=str(z.device),
            actual=str(K.device),
        )
    if (
        not bool(torch.isfinite(K).all())
        or bool((K <= 0).any())
        or not bool(torch.isfinite(z).all())
        or bool((z < 0).any())
    ):
        raise FlowContractError(
            "K must be positive and z must be non-negative",
            object_name="solve_rachford_rice",
            field="K/z",
            expected="finite K > 0 and finite z >= 0",
            actual="contains a non-finite or out-of-domain value",
        )
    total = z.sum(dim=-1)
    # Preserve the unconstrained one-component probes used by gradcheck and by
    # reduced-composition Jacobians without silently renormalizing the feed.
    simplex_tolerance = max(1.0e-5, 64.0 * torch.finfo(z.dtype).eps)
    if not bool(
        torch.isclose(
            total,
            torch.ones_like(total),
            rtol=simplex_tolerance,
            atol=simplex_tolerance,
        ).all()
    ):
        raise FlowContractError(
            "z must sum to one",
            object_name="solve_rachford_rice",
            field="z",
            expected="sum(component) == 1",
            actual="one or more rows are off the simplex",
        )
    if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1:
        raise FlowContractError(
            "maxiter must be a positive integer",
            object_name="solve_rachford_rice",
            field="maxiter",
            expected="integer >= 1",
            actual=maxiter,
        )
    if (
        isinstance(tol, bool)
        or not isinstance(tol, (int, float))
        or not math.isfinite(float(tol))
        or float(tol) <= 0
    ):
        raise FlowContractError(
            "tol must be positive and finite",
            object_name="solve_rachford_rice",
            field="tol",
            expected="> 0",
            actual=tol,
        )
    k_max = K.amax(dim=-1)
    k_min = K.amin(dim=-1)
    # An all-equal-K feed makes the Rachford-Rice equation singular (no
    # composition-dependent split): the bracket is zero-width and a K of exactly
    # 1 sends a pole to ±inf, both of which turn the bisection into NaN. Flag
    # those rows and give them a finite dummy bracket so the no-grad loop never
    # sees inf/NaN; they are overwritten with a single-phase sentinel below.
    degenerate = k_max == k_min
    # Poles bounding the physical root; replace non-finite poles (K == 1) with a
    # finite dummy, then nudge just inside to avoid evaluating on one.
    v_a = 1.0 / (1.0 - k_max)
    v_b = 1.0 / (1.0 - k_min)
    v_a = torch.where(torch.isfinite(v_a), v_a, torch.full_like(v_a, -1.0))
    v_b = torch.where(torch.isfinite(v_b), v_b, torch.full_like(v_b, 1.0))
    lo = torch.minimum(v_a, v_b)
    hi = torch.maximum(v_a, v_b)
    hi = torch.maximum(hi, lo + 1e-30)            # strictly positive width even when degenerate
    span = (hi - lo).clamp(min=1e-300)
    lo = lo + 1e-10 * span
    hi = hi - 1e-10 * span

    with torch.no_grad():
        V = 0.5 * (lo + hi)
        for _ in range(maxiter):
            r = rachford_rice_residual(V, K, z)
            converged = r.abs() < tol
            if bool(converged.all()):
                break
            # RR decreasing: r > 0 ⇒ root is at higher V, tighten the lower bracket.
            lo = torch.where(r > 0, V, lo)
            hi = torch.where(r < 0, V, hi)
            df = _rr_dV(V, K, z)
            v_newton = V - r / df
            inside = (v_newton > lo) & (v_newton < hi)
            v_upd = torch.where(inside, v_newton, 0.5 * (lo + hi))
            # Freeze cells that already met tolerance so a converged root is never
            # overwritten by a bracket/bisection step still running for other cells.
            V = torch.where(converged, V, v_upd)

    # One implicit-FT step from the converged root: V ≈ V_conv (RR≈0) but now a
    # differentiable function of (K, z): ∂V/∂K_i = −(∂RR/∂K_i)/(∂RR/∂V), etc.
    # Evaluate the IFT cleanup at a safe off-pole V for degenerate rows: their
    # bisection sits on a real pole, so RR(V) and dRR/dV are ±inf and would poison
    # the backward (0·inf survives the masking) even though the residual is zeroed.
    # The row is replaced by the single-phase sentinel V = 0 anyway.
    V_eval = torch.where(degenerate, torch.zeros_like(V), V)
    r = rachford_rice_residual(V_eval, K, z)
    df = _rr_dV(V_eval, K, z)
    r = torch.where(degenerate, torch.zeros_like(r), r)
    df = torch.where(df.abs() < 1e-300, torch.full_like(df, -1e-300), df)
    V_out = V_eval - r / df
    return torch.where(degenerate, torch.zeros_like(V_out), V_out)


__all__ = ["solve_rachford_rice", "rachford_rice_residual"]
