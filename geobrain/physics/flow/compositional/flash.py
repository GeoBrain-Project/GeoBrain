"""
Two-phase vapor-liquid flash (successive substitution), differentiable.

Given a feed ``z`` at ``(p, T)``, find the equilibrium split into a liquid ``x``
and vapor ``y`` with vapor mole fraction ``V``. Successive substitution (SSI):

  1. seed equilibrium ratios from Wilson, ``K_i``;
  2. solve Rachford-Rice for ``V`` ⇒ ``x_i = z_i/(1+V(K_i−1))``, ``y_i = K_i·x_i``;
  3. update ``K_i ← φ_i^L(x)/φ_i^V(y)`` from the cubic-EOS fugacity coefficients;
  4. iterate until the fugacity ratio ``f_i^L/f_i^V → 1`` (equal fugacities).

(At the Rachford-Rice root ``Σx_i = Σy_i = 1`` automatically.) The forward
iteration runs without autograd; an implicit-function-theorem cleanup step on
``lnK`` then makes the converged ``(V, x, y)`` differentiable w.r.t. ``(z, p, T)``.

:func:`flash_2ph` assumes the feed is two-phase. :func:`flash` is the full
stability-gated entry point: it runs the Michelsen tangent-plane stability test
first and only flashes the cells it finds unstable; the stable cells are returned
as a single phase whose cubic root is the one with the lower normalized Gibbs
energy ``g = Σ z_i·ln φ_i`` (the thermodynamically correct dense-vs-light root),
``V`` snapped to ``0`` (liquid-like) or ``1`` (vapor-like). The whole thing stays
batched (a per-cell stable/unstable mask) and differentiable in ``(z, p, T)``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch

from ..errors import FlowContractError, FlowConvergenceError
from ..solvers.diagnostics import convergence_diagnostics
from .cubic_eos import CubicEOS
from .rachford_rice import solve_rachford_rice
from .stability import stability_test


@dataclass(frozen=True, slots=True)
class FlashResult:
    """Converged two-phase flash."""

    V: torch.Tensor  # vapor mole fraction, shape (...)
    x: torch.Tensor  # liquid composition, shape (..., n)
    y: torch.Tensor  # vapor composition, shape (..., n)
    K: torch.Tensor  # equilibrium ratios y/x, shape (..., n)
    iterations: torch.Tensor  # per-cell iteration count, shape (...)
    converged: torch.Tensor  # bool, shape (...)
    max_iterations: int
    tolerance: float
    pressure_pa: torch.Tensor | None = None
    temperature_k: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class StabilityFlashResult:
    """Stability-gated flash: per-cell single- or two-phase outcome.

    ``single_phase`` (bool, shape ``(...)``) flags the cells the stability test
    found stable; for those ``x = y = z``, ``V`` is ``0`` (liquid-like) or ``1``
    (vapor-like), and ``Z`` is the Gibbs-minimizing cubic root. For two-phase
    cells ``V`` is the Rachford-Rice root and ``Z`` is the vapor root of ``y``
    (``Z_liquid`` carries the liquid root of ``x``); single-phase cells share the
    one selected root in both ``Z`` and ``Z_liquid``.
    """

    V: torch.Tensor  # vapor mole fraction, shape (...)
    x: torch.Tensor  # liquid composition, shape (..., n)
    y: torch.Tensor  # vapor composition, shape (..., n)
    K: torch.Tensor  # equilibrium ratios y/x, shape (..., n)
    Z: torch.Tensor  # selected compressibility (vapor root of y if two-phase)
    Z_liquid: torch.Tensor  # liquid root of x (two-phase); selected root (single)
    single_phase: torch.Tensor  # bool, shape (...): stable single phase
    liquid_like: torch.Tensor  # bool, shape (...): single phase is liquid-like
    converged: torch.Tensor  # bool, shape (...)
    iterations: torch.Tensor  # per-cell flash iteration count, shape (...)
    max_iterations: int
    tolerance: float
    pressure_pa: torch.Tensor | None = None
    temperature_k: torch.Tensor | None = None


def _validate_flash_inputs(
    p: object,
    T: object,
    z: object,
    *,
    K_init: object = None,
    maxiter: object,
    tol: object,
    object_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if (
        not isinstance(z, torch.Tensor)
        or not z.is_floating_point()
        or z.ndim < 1
        or z.shape[-1] < 2
    ):
        raise FlowContractError(
            "flash composition must be a floating component tensor",
            object_name=object_name,
            field="z",
            expected="floating [..., component] tensor with at least two components",
            actual=(type(z).__name__, tuple(getattr(z, "shape", ()))),
        )
    if not bool(torch.isfinite(z).all()) or bool((z < 0).any()):
        raise FlowContractError(
            "flash composition must be finite and non-negative",
            object_name=object_name,
            field="z",
            expected="finite mole fractions >= 0",
            actual="contains a negative or non-finite value",
        )
    total = z.sum(dim=-1)
    # Permit the small off-simplex probes used by finite differences and
    # unconstrained reduced-composition Jacobians without silently
    # renormalizing them. Materially invalid feeds still fail before EOS use.
    tolerance = max(1.0e-5, 64.0 * torch.finfo(z.dtype).eps)
    if not bool(torch.isclose(total, torch.ones_like(total), rtol=tolerance, atol=tolerance).all()):
        raise FlowContractError(
            "flash composition must sum to one",
            object_name=object_name,
            field="z",
            expected="sum(component) == 1",
            actual=total.detach().cpu().tolist(),
        )

    tensors: list[tuple[str, object]] = [("p", p), ("T", T)]
    for field, value in tensors:
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise FlowContractError(
                f"flash {field} must be a floating tensor",
                object_name=object_name,
                field=field,
                expected="floating torch.Tensor",
                actual=type(value).__name__,
            )
        if value.dtype != z.dtype or value.device != z.device:
            raise FlowContractError(
                "flash tensors must share one dtype and device",
                object_name=object_name,
                field=field,
                expected=(str(z.dtype), str(z.device)),
                actual=(str(value.dtype), str(value.device)),
            )
        if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise FlowContractError(
                f"flash {field} must be positive and finite",
                object_name=object_name,
                field=field,
                expected="> 0 in canonical SI units",
                actual="contains a non-positive or non-finite value",
            )
        try:
            broadcast_shape = torch.broadcast_shapes(value.shape, z.shape[:-1])
        except RuntimeError as error:
            raise FlowContractError(
                f"flash {field} does not broadcast over the composition batch",
                object_name=object_name,
                field=field,
                expected=tuple(z.shape[:-1]),
                actual=tuple(value.shape),
            ) from error
        if broadcast_shape != z.shape[:-1]:
            raise FlowContractError(
                f"flash {field} would expand the composition batch",
                object_name=object_name,
                field=field,
                expected=tuple(z.shape[:-1]),
                actual=tuple(broadcast_shape),
            )

    K_tensor: torch.Tensor | None = None
    if K_init is not None:
        if (
            not isinstance(K_init, torch.Tensor)
            or not K_init.is_floating_point()
            or K_init.shape != z.shape
            or K_init.dtype != z.dtype
            or K_init.device != z.device
            or not bool(torch.isfinite(K_init).all())
            or bool((K_init <= 0).any())
        ):
            raise FlowContractError(
                "K_init must be positive and align with z",
                object_name=object_name,
                field="K_init",
                expected=(str(z.dtype), str(z.device), tuple(z.shape), "> 0"),
                actual=(
                    type(K_init).__name__,
                    str(getattr(K_init, "dtype", None)),
                    str(getattr(K_init, "device", None)),
                    tuple(getattr(K_init, "shape", ())),
                ),
            )
        K_tensor = K_init
    if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1:
        raise FlowContractError(
            "flash maxiter must be a positive integer",
            object_name=object_name,
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
            "flash tol must be positive and finite",
            object_name=object_name,
            field="tol",
            expected="> 0",
            actual=tol,
        )
    assert isinstance(p, torch.Tensor)
    assert isinstance(T, torch.Tensor)
    return p, T, z, K_tensor


def _finite_range(value: torch.Tensor | None) -> tuple[float, float] | None:
    if value is None or value.numel() == 0:
        return None
    detached = value.detach()
    finite = detached[torch.isfinite(detached)]
    if finite.numel() == 0:
        return None
    return float(finite.min()), float(finite.max())


def require_flash_converged(
    result: FlashResult | StabilityFlashResult,
    *,
    object_name: str,
) -> None:
    """Raise before phase properties are read when any flash cell failed."""

    converged = result.converged
    expected_shape = result.V.shape
    malformed = (
        not isinstance(converged, torch.Tensor)
        or converged.dtype != torch.bool
        or converged.shape != expected_shape
        or not isinstance(result.iterations, torch.Tensor)
        or result.iterations.dtype != torch.int64
        or result.iterations.shape != expected_shape
    )
    if malformed:
        diagnostics = convergence_diagnostics(
            stage="flash",
            converged=False,
            reason="invalid_state",
            iterations=0,
            max_iterations=result.max_iterations,
            initial_residual_norm=float("inf"),
            residual_norm=float("inf"),
            residual_history=(),
            failed_cells=(),
        )
        raise FlowConvergenceError(
            "flash convergence metadata is malformed",
            object_name=object_name,
            field="converged/iterations",
            expected=f"bool/int64 tensors with shape {tuple(expected_shape)}",
            actual={
                "converged_dtype": str(getattr(converged, "dtype", None)),
                "converged_shape": tuple(getattr(converged, "shape", ())),
                "iterations_dtype": str(getattr(result.iterations, "dtype", None)),
                "iterations_shape": tuple(getattr(result.iterations, "shape", ())),
            },
            diagnostics=diagnostics,
        )
    if bool(converged.all()):
        return

    failed = torch.nonzero(~converged.reshape(-1), as_tuple=False).reshape(-1)
    failed_cells = tuple(int(index) for index in failed.detach().cpu().tolist())
    failed_iterations = tuple(
        int(value) for value in result.iterations.reshape(-1)[failed].detach().cpu().tolist()
    )
    failed_vapor_fraction = result.V.reshape(-1)[failed]
    if not bool(torch.isfinite(failed_vapor_fraction).all()):
        failure_reason: Literal["nonfinite", "invalid_state", "max_iterations"] = "nonfinite"
    elif bool(
        (
            (failed_vapor_fraction < -1.0e-9)
            | (failed_vapor_fraction > 1.0 + 1.0e-9)
        ).any()
    ) or any(
        iteration < 0 or iteration != result.max_iterations
        for iteration in failed_iterations
    ):
        failure_reason = "invalid_state"
    else:
        failure_reason = "max_iterations"
    diagnostics = convergence_diagnostics(
        stage="flash",
        converged=False,
        reason=failure_reason,
        iterations=max(failed_iterations, default=0),
        max_iterations=result.max_iterations,
        initial_residual_norm=float("inf"),
        residual_norm=float("inf"),
        residual_history=(),
        failed_cells=failed_cells,
    )
    actual = {
        "stage": diagnostics.stage,
        "failed_cells": diagnostics.failed_cells,
        "iterations": failed_iterations,
        "max_iterations": diagnostics.max_iterations,
        "tolerance": result.tolerance,
        "pressure_range_pa": _finite_range(result.pressure_pa),
        "temperature_range_k": _finite_range(result.temperature_k),
    }
    raise FlowConvergenceError(
        "phase-equilibrium flash did not converge",
        object_name=object_name,
        field="flash",
        expected="every cell converged before phase properties are read",
        actual=actual,
        hint="inspect failed cells and revise the state, flash tolerance, or iteration budget",
        diagnostics=diagnostics,
    )


def _split(
    K: torch.Tensor,
    z: torch.Tensor,
    V: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = z / (1.0 + V.unsqueeze(-1) * (K - 1.0))
    y = K * x
    return x, y


def _safe_comp(c: torch.Tensor) -> torch.Tensor:
    """Project a (possibly invalid) composition onto the probability simplex so
    the cubic EOS never receives NaN / negative mole fractions.

    For a genuine two-phase split at the Rachford-Rice root the columns are
    already non-negative and sum to one, so this is the identity; it only bites
    on non-two-phase feeds (``V`` outside ``[0, 1]`` ⇒ negative / non-finite
    fractions) where it prevents the ``NaN`` → :func:`torch.linalg.eigvals` →
    LAPACK-DGEBAL segmentation fault. It is the identity at the converged
    solution, so the implicit-function-theorem gradient is unaffected.
    """
    c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    return c / c.sum(dim=-1, keepdim=True).clamp_min(1e-300)


def _equilibrium_lnK(
    eos: CubicEOS,
    K: torch.Tensor,
    z: torch.Tensor,
    p: torch.Tensor,
    T: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One SSI map: K,z → updated lnK = lnφ^L(x) − lnφ^V(y), plus (V, x, y)."""
    V = solve_rachford_rice(K, z)
    x, y = _split(K, z, V)
    _, ln_phi_l = eos.phase(_safe_comp(x), p, T, root="liquid")
    _, ln_phi_v = eos.phase(_safe_comp(y), p, T, root="vapor")
    return ln_phi_l - ln_phi_v, V, x, y


def flash_2ph(
    eos: CubicEOS,
    p: torch.Tensor,
    T: torch.Tensor,
    z: torch.Tensor,
    *,
    K_init: torch.Tensor | None = None,
    maxiter: int = 200,
    tol: float = 1e-12,
    differentiable: bool = True,
) -> FlashResult:
    """Two-phase flash of feed ``z`` at ``(p, T)``.

    ``z`` is ``(..., n_components)``; ``p``/``T`` broadcast over the leading axes.
    Returns a :class:`FlashResult`; ``V`` is differentiable w.r.t. ``(z, p, T)``
    when ``differentiable=True``.
    """
    p, T, z, K_init = _validate_flash_inputs(
        p,
        T,
        z,
        K_init=K_init,
        maxiter=maxiter,
        tol=tol,
        object_name="flash_2ph",
    )

    # ---- forward SSI (no autograd) ----
    with torch.no_grad():
        K = eos.mixture.wilson_k(p, T) if K_init is None else K_init.clone()
        eps = z.new_full(z.shape[:-1], float("inf"))
        done = torch.zeros(z.shape[:-1], dtype=torch.bool, device=z.device)
        iterations = torch.zeros(z.shape[:-1], dtype=torch.int64, device=z.device)
        it = 0
        for it in range(1, maxiter + 1):
            ln_phi_l, V, x, y = _equilibrium_lnK(eos, K, z, p, T)
            K_new = torch.exp(ln_phi_l)
            # fugacity-ratio convergence: f_i^L/f_i^V = φ_i^L x_i /(φ_i^V y_i) → 1
            ratio = K_new / K
            eps_new = (ratio - 1.0).abs().amax(dim=-1)
            newly_done = (~done) & (eps_new < tol)
            iterations = torch.where(
                newly_done,
                torch.full_like(iterations, it),
                iterations,
            )
            K = torch.where(done.unsqueeze(-1), K, K_new)
            eps = torch.where(done, eps, eps_new)
            done = done | newly_done
            if bool(done.all()):
                break
        iterations = torch.where(
            iterations > 0,
            iterations,
            torch.full_like(iterations, maxiter),
        )
        V = solve_rachford_rice(K, z)
        x, y = _split(K, z, V)
        # A run only "converged" if the fugacity ratio settled AND it landed on a
        # physical two-phase root (V ∈ [0, 1], all finite). Single-phase / diverged
        # feeds (V ~ -1e13, K → 1) must report False, not a silent garbage split.
        converged = (eps < tol) & torch.isfinite(V) & (V >= -1e-9) & (V <= 1.0 + 1e-9)

    result_kwargs = {
        "iterations": iterations,
        "converged": converged,
        "max_iterations": maxiter,
        "tolerance": tol,
        "pressure_pa": p,
        "temperature_k": T,
    }
    if not differentiable:
        return FlashResult(V=V, x=x, y=y, K=K, **result_kwargs)

    # A consumer must reject this result before reading phase properties.  Do
    # not enter the IFT solve for a failed batch: its Jacobian may be singular,
    # which would replace the actionable convergence diagnostics with a linear
    # algebra exception.
    if not bool(converged.all()):
        return FlashResult(V=V, x=x, y=y, K=K, **result_kwargs)

    # ---- implicit-function-theorem cleanup on lnK ----
    # At convergence R(lnK) = lnφ^L(x) − lnφ^V(y) − lnK = 0. One Newton step in
    # lnK from the converged value yields lnK ≈ lnK* (forward) but differentiable
    # in (z, p, T): ∂lnK/∂θ = −(∂R/∂lnK)⁻¹ ∂R/∂θ.
    lnK0 = torch.log(K).detach()

    def residual(lnK_flat: torch.Tensor) -> torch.Tensor:
        lnK = lnK_flat.reshape(K.shape)
        ln_phi_l, _, _, _ = _equilibrium_lnK(eos, torch.exp(lnK), z, p, T)
        return (ln_phi_l - lnK).reshape(-1)

    flat0 = lnK0.reshape(-1)
    R = residual(flat0)
    J = torch.autograd.functional.jacobian(residual, flat0, create_graph=True, vectorize=True)
    # cells are independent → J is block-diagonal; solving the full system is
    # equivalent and keeps the autograd graph intact for the IFT gradient.
    dlnK = torch.linalg.solve(J, R)
    lnK = (flat0 - dlnK).reshape(K.shape)

    K = torch.exp(lnK)
    V = solve_rachford_rice(K, z)
    x, y = _split(K, z, V)
    return FlashResult(V=V, x=x, y=y, K=K, **result_kwargs)


def _gibbs_single_root(
    eos: CubicEOS,
    z: torch.Tensor,
    p: torch.Tensor,
    T: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the single-phase cubic root by minimum normalized Gibbs energy.

    A stable single phase has, in general, two admissible cubic roots (a dense
    "liquid" root and a light "vapor" root). The physically realized phase is the
    one with the lower molar Gibbs energy; up to the ideal-gas reference (common
    to both roots at fixed ``z, p, T``) that is ``g = Σ_i z_i·ln φ_i``. Returns
    ``(Z, ln_phi, liquid_like)`` for the winning root, differentiable in
    ``(z, p, T)`` through the EOS.
    """
    Z_l, ln_phi_l = eos.phase(z, p, T, root="liquid")
    Z_v, ln_phi_v = eos.phase(z, p, T, root="vapor")
    g_l = (z * ln_phi_l).sum(dim=-1)
    g_v = (z * ln_phi_v).sum(dim=-1)
    liquid_like = g_l <= g_v
    sel = liquid_like.unsqueeze(-1)
    Z = torch.where(liquid_like, Z_l, Z_v)
    ln_phi = torch.where(sel, ln_phi_l, ln_phi_v)
    return Z, ln_phi, liquid_like


def flash(
    eos: CubicEOS,
    p: torch.Tensor,
    T: torch.Tensor,
    z: torch.Tensor,
    *,
    maxiter: int = 200,
    tol: float = 1e-12,
    stability_tol: float = 1e-10,
    stability_maxiter: int = 1000,
    differentiable: bool = True,
) -> StabilityFlashResult:
    """Stability-gated vapor-liquid flash of feed ``z`` at ``(p, T)``.

    Runs the Michelsen tangent-plane stability test, then for each cell:

    * **unstable** → two-phase flash warm-started by the stability ``K``
      (Rachford-Rice ``V`` ∈ (0, 1), liquid ``x`` / vapor ``y``);
    * **stable** → single phase, ``x = y = z``, ``V`` snapped to ``0`` (the
      Gibbs-minimizing root is the dense "liquid" one) or ``1`` (the light
      "vapor" one), and ``Z`` the Gibbs-minimizing cubic root.

    ``z`` is ``(..., n_components)``; ``p``/``T`` broadcast over the leading axes.
    The single-/two-phase split is decided by GeoBrain itself (no external input),
    and ``V`` is differentiable w.r.t. ``(z, p, T)`` when ``differentiable=True``.
    """
    p, T, z, _ = _validate_flash_inputs(
        p,
        T,
        z,
        maxiter=maxiter,
        tol=tol,
        object_name="flash",
    )
    if (
        isinstance(stability_maxiter, bool)
        or not isinstance(stability_maxiter, int)
        or stability_maxiter < 1
    ):
        raise FlowContractError(
            "stability_maxiter must be a positive integer",
            object_name="flash",
            field="stability_maxiter",
            expected="integer >= 1",
            actual=stability_maxiter,
        )
    if (
        isinstance(stability_tol, bool)
        or not isinstance(stability_tol, (int, float))
        or not math.isfinite(float(stability_tol))
        or float(stability_tol) <= 0
    ):
        raise FlowContractError(
            "stability_tol must be positive and finite",
            object_name="flash",
            field="stability_tol",
            expected="> 0",
            actual=stability_tol,
        )

    # ---- 1) stability decides the split (discrete control flow, no autograd) ----
    stab = stability_test(eos, p, T, z, tol=stability_tol, maxiter=stability_maxiter)
    if not bool(stab.converged.all()):
        require_flash_converged(
            FlashResult(
                V=torch.zeros_like(stab.S_vapor),
                x=z,
                y=z,
                K=stab.K,
                iterations=stab.iterations,
                converged=stab.converged,
                max_iterations=stability_maxiter,
                tolerance=stability_tol,
                pressure_pa=p,
                temperature_k=T,
            ),
            object_name="flash-stability",
        )
    single_phase = stab.stable

    # ---- 2) two-phase branch (warm-started by the stability K) ----
    # Flash only unstable cells. Running a forced two-phase IFT on stable cells
    # can produce an unphysical root or a singular Jacobian before the stable
    # branch is masked, hiding the useful stability result.
    batch_shape = z.shape[:-1]
    n_components = z.shape[-1]
    z_flat = z.reshape(-1, n_components)
    p_flat = torch.broadcast_to(p, batch_shape).reshape(-1)
    T_flat = torch.broadcast_to(T, batch_shape).reshape(-1)
    unstable_flat = (~single_phase).reshape(-1)
    unstable_index = torch.nonzero(unstable_flat, as_tuple=False).reshape(-1)

    V_two_flat = torch.zeros_like(p_flat)
    x_two_flat = z_flat
    y_two_flat = z_flat
    K_two_flat = stab.K.reshape(-1, n_components)
    converged_two_flat = torch.ones_like(unstable_flat)
    iterations_two_flat = torch.zeros_like(unstable_flat, dtype=torch.int64)
    if unstable_index.numel() > 0:
        two = flash_2ph(
            eos,
            p_flat[unstable_index],
            T_flat[unstable_index],
            z_flat[unstable_index],
            K_init=K_two_flat[unstable_index],
            maxiter=maxiter,
            tol=tol,
            differentiable=differentiable,
        )
        require_flash_converged(two, object_name="flash-two-phase")
        V_two_flat = V_two_flat.index_copy(0, unstable_index, two.V.reshape(-1))
        x_two_flat = x_two_flat.index_copy(0, unstable_index, two.x.reshape(-1, n_components))
        y_two_flat = y_two_flat.index_copy(0, unstable_index, two.y.reshape(-1, n_components))
        K_two_flat = K_two_flat.index_copy(0, unstable_index, two.K.reshape(-1, n_components))
        converged_two_flat = converged_two_flat.index_copy(
            0, unstable_index, two.converged.reshape(-1)
        )
        iterations_two_flat = iterations_two_flat.index_copy(
            0, unstable_index, two.iterations.reshape(-1)
        )

    V_two = V_two_flat.reshape(batch_shape)
    x_two = x_two_flat.reshape(*batch_shape, n_components)
    y_two = y_two_flat.reshape(*batch_shape, n_components)
    K_two = K_two_flat.reshape(*batch_shape, n_components)
    converged_two = converged_two_flat.reshape(batch_shape)
    iterations_two = iterations_two_flat.reshape(batch_shape)

    # ---- 3) single-phase branch (Gibbs-min cubic root) ----
    Z_single, _, liquid_like = _gibbs_single_root(eos, z, p, T)
    # Z of the liquid/vapor roots of the two-phase split, for reporting.
    Z_l_two, _ = eos.phase(x_two, p, T, root="liquid")
    Z_v_two, _ = eos.phase(y_two, p, T, root="vapor")

    # ---- 4) merge per-cell on the stable mask (keeps autograd graph) ----
    sp = single_phase
    sp_col = sp.unsqueeze(-1)
    # single-phase vapor fraction: 0 (liquid-like) / 1 (vapor-like). Differentiable
    # zero-perturbations from z keep dV/d(z,p,T)=0 for stable cells (correct: the
    # label is locally constant), while two-phase cells carry the live RR gradient.
    z_pert = (z - z.detach()).sum(dim=-1)  # ≡ 0, but graph-connected to z
    V_single = torch.where(liquid_like, z_pert, 1.0 + z_pert)
    V = torch.where(sp, V_single, V_two)
    x = torch.where(sp_col, z, x_two)
    y = torch.where(sp_col, z, y_two)
    K = torch.where(sp_col, stab.K, K_two)
    Z = torch.where(sp, Z_single, Z_v_two)
    Z_liquid = torch.where(sp, Z_single, Z_l_two)
    # two-phase cells must have actually converged; single-phase cells are "done".
    converged = stab.converged & converged_two

    return StabilityFlashResult(
        V=V,
        x=x,
        y=y,
        K=K,
        Z=Z,
        Z_liquid=Z_liquid,
        single_phase=single_phase,
        liquid_like=liquid_like,
        converged=converged,
        iterations=iterations_two,
        max_iterations=maxiter,
        tolerance=tol,
        pressure_pa=p,
        temperature_k=T,
    )


__all__ = [
    "flash",
    "flash_2ph",
    "FlashResult",
    "StabilityFlashResult",
    "require_flash_converged",
]
