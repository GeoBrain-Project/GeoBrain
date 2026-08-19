"""
Layered-earth EM numerical kernels (MT 1D Wait recursion).

Other 1D layered solvers (CSEM1D, TEM1D, HEM, VTEM) live with their
operators in the phase-E3 family modules. This file hosts only solutions
that are genuinely reusable across families.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math

import torch

from ....core.errors import GeoBrainError
from geobrain.physics.em.conventions import MU_0


def mt1d_wait_impedance(
    sigma: torch.Tensor,
    thickness: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    mu0: float = MU_0,
) -> torch.Tensor:
    """
    1D layered-earth surface MT impedance via Wait's recursion.

    Args:
        sigma:       ``(n_layer,)`` real conductivities (S/m), top→bottom.
        thickness:   ``(n_layer - 1,)`` layer thicknesses (m). The bottom
                     layer is implicitly infinite.
        frequencies: ``(n_freq,)`` frequencies (Hz).
        mu0:         vacuum permeability (defaults to E1 constant ``MU_0``).

    Returns:
        ``(n_freq,)`` complex128 impedance at the surface.

    Numerics: complex128 throughout for precision. Bottom-up recursion of
    the intrinsic impedance with ``tanh(k h)`` at each interface.
    """
    if sigma.ndim != 1:
        raise GeoBrainError(
            f"sigma must be 1D, got shape {tuple(sigma.shape)}",
            object_name="mt1d_wait_impedance",
            field="sigma",
            expected="1D tensor",
            actual=tuple(sigma.shape),
        )
    if thickness.ndim != 1:
        raise GeoBrainError(
            f"thickness must be 1D, got shape {tuple(thickness.shape)}",
            object_name="mt1d_wait_impedance",
            field="thickness",
            expected="1D tensor",
            actual=tuple(thickness.shape),
        )
    if thickness.numel() != max(0, int(sigma.numel()) - 1):
        raise GeoBrainError(
            "thickness must have one fewer element than sigma "
            f"(got sigma={int(sigma.numel())}, "
            f"thickness={int(thickness.numel())})",
            object_name="mt1d_wait_impedance",
            field="thickness",
            expected=f"numel == sigma.numel() - 1 ({max(0, int(sigma.numel()) - 1)})",
            actual=int(thickness.numel()),
        )
    if frequencies.ndim != 1:
        raise GeoBrainError(
            f"frequencies must be 1D, got shape {tuple(frequencies.shape)}",
            object_name="mt1d_wait_impedance",
            field="frequencies",
            expected="1D tensor",
            actual=tuple(frequencies.shape),
        )

    omega = (2.0 * math.pi) * frequencies.to(device=sigma.device, dtype=sigma.dtype)
    omega_c = omega.to(torch.complex128)
    sigma_c = sigma.to(torch.complex128)
    thickness_c = thickness.to(torch.complex128)

    k_n = torch.sqrt(1j * omega_c * mu0 * sigma_c[-1])
    z = (1j * omega_c * mu0) / k_n
    for layer in range(int(sigma.numel()) - 2, -1, -1):
        k = torch.sqrt(1j * omega_c * mu0 * sigma_c[layer])
        z_intrinsic = (1j * omega_c * mu0) / k
        tanh_arg = torch.tanh(k * thickness_c[layer])
        z = z_intrinsic * (z + z_intrinsic * tanh_arg) / (
            z_intrinsic + z * tanh_arg
        )
    return z


def _safe_tanh(arg: torch.Tensor) -> torch.Tensor:
    """``tanh(arg)`` that saturates to ``sign(Re(arg))`` for huge real parts.

    For the layered-earth recursion ``arg = u_i * h_i`` the real part can grow
    without bound (large frequency, large layer thickness). ``torch.tanh`` of a
    complex argument internally evaluates ``e^{|Re|}`` and overflows for
    ``|Re| > ~700`` in float64. Saturation is the correct asymptotic value
    (``tanh(x + iy) -> sign(x)`` as ``|x| -> infty``) and the recursion is
    well-defined there.
    """
    re = arg.real
    big = re.abs() > 350.0
    if not bool(big.any()):
        return torch.tanh(arg)
    safe_arg = torch.where(big, torch.zeros_like(arg), arg)
    tanh_val = torch.tanh(safe_arg)
    sign_real = torch.sign(re).to(arg.dtype)
    return torch.where(big, sign_real, tanh_val)


def layered_te_reflection(
    lam: torch.Tensor,
    omega: torch.Tensor,
    sigma: torch.Tensor,
    thickness: torch.Tensor,
    mu0: float = MU_0,
    *,
    safe_tanh: bool = True,
) -> torch.Tensor:
    """Surface TE reflection coefficient ``r_TE(λ, ω)`` of a layered earth.

    Wait-style bottom-up admittance recursion, shared by the 1-D EM operators:
    CSEM1D (frequency domain) and TEM1D (time domain) build the SAME recursion;
    only the surrounding Hankel / time transform and the time convention (baked
    into the field equations downstream, not here) differ.

    ``lam`` and ``omega`` must already broadcast against each other (the caller
    shapes them, e.g. ``(1, n_lambda)`` / ``(n_freq, 1)`` for a fixed grid, or a
    3-axis broadcast for a nested Hankel integrand). ``sigma`` is ``(n_layers,)``
    real; ``thickness`` is ``(n_layers - 1,)`` real (empty for a half-space).
    Returns complex128 of shape ``broadcast(lam, omega)``.

    Recursion::

        u_n   = √(λ² + i ω μ₀ σ_n)            (principal sqrt, Re(u_n) >= 0)
        Y_n   = u_n / (i ω μ₀)
        Y_hat = Y_N ;  Y_hat_i = Y_i (Y_hat + Y_i tanh(u_i h_i))
                                     / (Y_i  + Y_hat tanh(u_i h_i))
        r_TE  = (Y_0_air - Y_hat) / (Y_0_air + Y_hat),  Y_0_air = λ / (i ω μ₀)

    ``safe_tanh`` saturates ``tanh`` for huge ``|Re(u_i h_i)|`` (needed by the
    time-domain integrand; a no-op for the moderate frequency-domain range).
    """
    n_layers = int(sigma.numel())
    lam_c = lam.to(torch.complex128)
    omega_c = omega.to(torch.complex128)
    s_mu0 = (1j * mu0) * omega_c
    sigma_c = sigma.to(torch.complex128)
    lam_sq = lam_c * lam_c
    tanh_fn = _safe_tanh if safe_tanh else torch.tanh

    u_layers: list[torch.Tensor] = []
    Y_layers: list[torch.Tensor] = []
    for i in range(n_layers):
        u_i = torch.sqrt(lam_sq + s_mu0 * sigma_c[i])
        u_layers.append(u_i)
        Y_layers.append(u_i / s_mu0)

    Y_hat = Y_layers[-1]
    if n_layers > 1:
        thickness_c = thickness.to(torch.complex128)
        for i in range(n_layers - 2, -1, -1):
            Y_i = Y_layers[i]
            u_i = u_layers[i]
            tanh_uh = tanh_fn(u_i * thickness_c[i])
            Y_hat = Y_i * (Y_hat + Y_i * tanh_uh) / (Y_i + Y_hat * tanh_uh)

    Y_0 = lam_c / s_mu0
    return (Y_0 - Y_hat) / (Y_0 + Y_hat)
