"""
Michelsen two-phase stability test (tangent-plane distance).

Before flashing we must decide whether a feed ``z`` at ``(p, T)`` is a stable
single phase or splits into two. Michelsen's test (1982) launches two trial
phases, a vapor-like one (``W_i = z_i·K_i``) and a liquid-like one
(``W_i = z_i/K_i``), and drives each to a stationary point of the tangent-plane
distance by successive substitution on the equilibrium ratios::

    W_i = z_i·K_i  (vapor trial)  /  z_i/K_i  (liquid trial)
    w   = W / ΣW                  (normalized trial composition)
    R_i = f_i(z) / (ΣW · f_i(w))  (vapor)   or   ΣW·f_i(w) / f_i(z)  (liquid)
    K_i ← K_i · R_i

A trial that collapses to the feed (``K → 1``, *trivial*) or whose mole-number
sum ``ΣW ≤ 1`` leaves the tangent plane non-negative, stable. If either trial
reaches ``ΣW > 1`` non-trivially the single phase is **unstable** (two-phase),
and the converged trials give the flash a warm K-value start (``K = y/x``).

The verdict is a discrete control-flow decision, so the test runs without
autograd (the *flash* that follows is the differentiable part).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .cubic_eos import CubicEOS

_EPS = 1e-18


@dataclass(frozen=True, slots=True)
class StabilityResult:
    """Outcome of the two-phase stability test (per cell)."""

    stable: torch.Tensor  # bool (...): single-phase stable
    K: torch.Tensor  # (..., n) warm-start equilibrium ratios for the flash
    trivial_vapor: torch.Tensor
    trivial_liquid: torch.Tensor
    S_vapor: torch.Tensor  # (...) vapor-trial mole-number sum (TPD indicator)
    S_liquid: torch.Tensor
    converged: torch.Tensor
    iterations: torch.Tensor
    max_iterations: int
    tolerance: float
    pressure_pa: torch.Tensor
    temperature_k: torch.Tensor


def _michelsen_trial(
    eos: CubicEOS,
    p: torch.Tensor,
    T: torch.Tensor,
    z: torch.Tensor,
    ln_fz: torch.Tensor,
    trial: Literal["vapor", "liquid"],
    tol: float,
    maxiter: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """One Michelsen trial. Returns ``(stable, trivial, S, w)`` (no autograd)."""
    is_vapor = trial == "vapor"
    K = eos.mixture.wilson_k(p, T).clone()
    S = torch.ones(z.shape[:-1], dtype=z.dtype, device=z.device)
    trivial = torch.zeros(z.shape[:-1], dtype=torch.bool, device=z.device)
    done = torch.zeros_like(trivial)
    iterations = torch.zeros(z.shape[:-1], dtype=torch.int64, device=z.device)
    w = z.clone()
    for iteration in range(maxiter):
        W = z * K if is_vapor else z / K  # trial mole numbers
        S_new = W.sum(dim=-1)
        w_new = W / S_new.unsqueeze(-1)
        _, ln_phi_w = eos.phase(w_new, p, T, root=trial)
        ln_fxy = ln_phi_w + torch.log(w_new.clamp_min(_EPS))  # ln(f_w / p)
        ln_S = torch.log(S_new).unsqueeze(-1)
        ln_R = (ln_fz - ln_S - ln_fxy) if is_vapor else (ln_S + ln_fxy - ln_fz)
        R = torch.exp(ln_R)
        K_new = K * R
        r_norm = ((R - 1.0) ** 2).sum(dim=-1)
        k_norm = (torch.log(K_new) ** 2).sum(dim=-1)
        triv = k_norm < tol
        conv = r_norm < tol
        step_done = triv | conv
        # freeze cells already finished (keep their converged K, S, trivial)
        upd = ~done
        newly_done = upd & step_done
        iterations = torch.where(
            newly_done,
            torch.full_like(iterations, iteration + 1),
            iterations,
        )
        K = torch.where(upd.unsqueeze(-1), K_new, K)
        S = torch.where(upd, S_new, S)
        w = torch.where(upd.unsqueeze(-1), w_new, w)
        trivial = torch.where(upd & step_done, triv, trivial)
        done = done | step_done
        if bool(done.all()):
            break
    iterations = torch.where(
        iterations > 0,
        iterations,
        torch.full_like(iterations, maxiter),
    )
    stable = trivial | (S <= 1.0 + tol)
    return stable, trivial, S, w, K, done, iterations


def stability_test(
    eos: CubicEOS,
    p: torch.Tensor,
    T: torch.Tensor,
    z: torch.Tensor,
    *,
    tol: float = 1e-10,
    maxiter: int = 1000,
) -> StabilityResult:
    """Michelsen stability of feed ``z`` at ``(p, T)``.

    ``z`` is ``(..., n_components)``; ``p``/``T`` broadcast over the leading axes.
    Returns a :class:`StabilityResult` whose ``stable`` flag is per cell and
    whose ``K`` is the two-phase warm start (``y/x`` from the trials) where
    unstable, or the Wilson estimate where stable.
    """
    z = eos.mixture.validate_composition_tensor(
        z,
        object_name="stability_test",
    )
    p, T = eos.mixture.validate_state_tensors(
        p,
        T,
        object_name="stability_test",
    )
    with torch.no_grad():
        # feed log-fugacity (z taken as vapor root), p cancels in the ratio R
        _, ln_phi_z = eos.phase(z, p, T, root="vapor")
        ln_fz = ln_phi_z + torch.log(z.clamp_min(_EPS))

        stable_v, trivial_v, S_v, w_v, _, done_v, iterations_v = _michelsen_trial(
            eos, p, T, z, ln_fz, "vapor", tol, maxiter
        )
        stable_l, trivial_l, S_l, w_l, _, done_l, iterations_l = _michelsen_trial(
            eos, p, T, z, ln_fz, "liquid", tol, maxiter
        )
        stable = stable_v & stable_l

        # Warm-start K for the flash: y (vapor trial) over x (liquid trial),
        # falling back to Wilson where the trial was trivial / stable.
        wilson = eos.mixture.wilson_k(p, T)
        K_split = w_v / w_l.clamp_min(_EPS)
        usable = (~stable).unsqueeze(-1) & torch.isfinite(K_split) & (K_split > 0)
        K = torch.where(usable, K_split, wilson)
    return StabilityResult(
        stable=stable,
        K=K,
        trivial_vapor=trivial_v,
        trivial_liquid=trivial_l,
        S_vapor=S_v,
        S_liquid=S_l,
        converged=done_v & done_l,
        iterations=torch.maximum(iterations_v, iterations_l),
        max_iterations=maxiter,
        tolerance=tol,
        pressure_pa=p,
        temperature_k=T,
    )


__all__ = ["stability_test", "StabilityResult"]
