"""
Split-field Convolutional PML (CPML), after Komatitsch & Martin (2007).

This is the main numerical upgrade over GeoBrain's current Gerjan exponential
damping: a true CPML with per-axis memory variables ``ψ`` and a complex
frequency-shifted (CFS) stretch ``(κ, α)``. CPML absorbs grazing-incidence and
evanescent energy far better than a scalar damping field.

Coordinate stretching replaces each spatial derivative ``∂`` by::

    ∂̃ f = (1/κ) ∂f + ψ ,      ψ^n = b ψ^{n-1} + a (∂f)^n

with, for a damping profile ``d``, stretch ``κ ≥ 1`` and shift ``α ≥ 0``::

    b = exp(-(d/κ + α) Δt)
    a = d / (κ (d + κ α)) · (b − 1)        (a = 0 where d = 0)

The damping profile inside a PML of thickness ``L = nabc·Δh`` is::

    d(ℓ) = d0 (ℓ/L)^p ,   d0 = −(p+1) v_max ln(R) / (2 L)

where ``ℓ`` is the distance from the inner PML edge, ``R`` the target reflection
coefficient and ``p`` the polynomial order. ``α`` tapers linearly from
``α_max`` at the inner edge to 0 at the outer edge; ``κ`` ramps from 1 to
``κ_max``.

Because pressure lives on integer nodes and velocities on half nodes, profiles
are built at **both** the integer offset (0.0) and the half offset (0.5) for
each axis, so every stretched derivative uses the profile co-located with the
node it updates.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor


FloatArray: TypeAlias = NDArray[np.float64]
ProfileTensors: TypeAlias = dict[str, tuple[Tensor, Tensor, Tensor]]


def damping_profile_1d(
    n: int,
    nabc: int,
    dh: float,
    vmax: float,
    *,
    reflection: float,
    npower: float,
    kappa_max: float,
    alpha_max: float,
    pml_left: bool,
    pml_right: bool,
    offset: float = 0.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build 1-D CPML profiles ``(d, kappa, alpha)`` of length ``n``.

    Args:
        n: number of grid points along the axis (including PML padding).
        nabc: PML thickness in cells.
        dh: grid spacing along the axis.
        vmax: maximum velocity (sets peak damping ``d0``).
        pml_left / pml_right: whether the low-/high-index face has a PML
            (a free surface disables the corresponding face).
        offset: node offset in cells (0.0 for integer nodes, 0.5 for the
            staggered half nodes).

    Returns:
        Three arrays ``d``, ``kappa``, ``alpha`` of shape ``(n,)``.
    """
    length = nabc * dh
    d0 = -(npower + 1.0) * vmax * np.log(reflection) / (2.0 * length)

    idx = np.arange(n, dtype=np.float64) + offset
    ell: FloatArray = np.zeros(n, dtype=np.float64)

    if pml_left:
        # Inner edge of the left PML sits at index nabc; ell grows outward.
        left = nabc - idx
        mask = left > 0.0
        ell[mask] = np.maximum(ell[mask], left[mask] * dh)
    if pml_right:
        inner = (n - 1) - nabc
        right = idx - inner
        mask = right > 0.0
        ell[mask] = np.maximum(ell[mask], right[mask] * dh)

    xnorm = np.clip(ell / length, 0.0, 1.0)
    d = d0 * xnorm**npower
    kappa = 1.0 + (kappa_max - 1.0) * xnorm**npower
    alpha = alpha_max * (1.0 - xnorm)
    # Outside the PML d=0; alpha there is irrelevant (a=0) but keep it tidy.
    alpha = np.where(xnorm > 0.0, alpha, 0.0)
    return d, kappa, alpha


def _ba_from_profile(
    d: FloatArray, kappa: FloatArray, alpha: FloatArray, dt: float
) -> tuple[FloatArray, FloatArray]:
    """Convert ``(d, kappa, alpha)`` to CPML recursion coefficients ``(b, a)``."""
    b = np.exp(-(d / kappa + alpha) * dt)
    denom = kappa * (d + kappa * alpha)
    a = np.zeros_like(d)
    nz = d > 0.0
    a[nz] = d[nz] / denom[nz] * (b[nz] - 1.0)
    return b, a


@dataclass
class CPML:
    """Precomputed CPML recursion coefficients, broadcast-ready.

    The ``*_int`` tensors are co-located with integer nodes (used by the
    pressure update, which differences velocities) and the ``*_half`` tensors
    with the staggered half nodes (used by the velocity updates, which
    difference pressure). x-tensors broadcast as ``(1, 1, 1, nx)`` and
    z-tensors as ``(1, 1, nz, 1)``.
    """

    bx_int: Tensor
    ax_int: Tensor
    kx_int: Tensor
    bz_int: Tensor
    az_int: Tensor
    kz_int: Tensor
    bx_half: Tensor
    ax_half: Tensor
    kx_half: Tensor
    bz_half: Tensor
    az_half: Tensor
    kz_half: Tensor

    @staticmethod
    def stretch(
        deriv: Tensor, psi: Tensor, b: Tensor, a: Tensor, kappa: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply the CPML stretch to ``deriv``; return ``(stretched, psi_new)``.

        ``psi_new = b·psi + a·deriv`` and ``stretched = deriv/κ + psi_new``.
        Pure / differentiable.
        """
        psi_new = b * psi + a * deriv
        return deriv / kappa + psi_new, psi_new


def build_cpml(
    nz: int,
    nx: int,
    nabc: int,
    dz: float,
    dx: float,
    dt: float,
    vmax: float,
    *,
    reflection: float = 1.0e-4,
    npower: float = 2.0,
    kappa_max: float = 1.0,
    alpha_max: float = 0.0,
    free_surface: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CPML:
    """Construct a :class:`CPML` for a padded ``(nz, nx)`` grid.

    The top z-face PML is disabled when ``free_surface`` is True.
    """
    common = dict(
        reflection=reflection, npower=npower, kappa_max=kappa_max, alpha_max=alpha_max
    )

    def make(
        n: int,
        dh: float,
        *,
        pml_left: bool,
        pml_right: bool,
        axis: str,
    ) -> ProfileTensors:
        out: ProfileTensors = {}
        for tag, off in (("int", 0.0), ("half", 0.5)):
            d, k, al = damping_profile_1d(
                n, nabc, dh, vmax, pml_left=pml_left, pml_right=pml_right,
                offset=off, **common,
            )
            b, a = _ba_from_profile(d, k, al, dt)
            shape = (1, 1, 1, n) if axis == "x" else (1, 1, n, 1)

            def t(arr: FloatArray) -> Tensor:
                return torch.as_tensor(arr, dtype=dtype, device=device).reshape(shape)

            out[tag] = (t(b), t(a), t(k))
        return out

    x = make(nx, dx, pml_left=True, pml_right=True, axis="x")
    z = make(nz, dz, pml_left=not free_surface, pml_right=True, axis="z")

    return CPML(
        bx_int=x["int"][0], ax_int=x["int"][1], kx_int=x["int"][2],
        bz_int=z["int"][0], az_int=z["int"][1], kz_int=z["int"][2],
        bx_half=x["half"][0], ax_half=x["half"][1], kx_half=x["half"][2],
        bz_half=z["half"][0], az_half=z["half"][1], kz_half=z["half"][2],
    )
