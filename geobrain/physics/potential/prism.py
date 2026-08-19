# pyright: reportPrivateImportUsage=false
"""
Analytic 3-D rectangular-prism kernels (gravity only).

Pure-torch, fully
differentiable, no custom ``autograd.Function`` bridge needed because
the Nagy 2000 closed-form integrals are smooth functions of the prism
corners and densities. Gradients through ``density`` flow trivially;
gradients through cell geometry flow through the corner sums.

Kernel naming follows the geophysical convention:

    g_z, g_xx, g_yy, g_zz, g_xy, g_xz, g_yz

The eight-corner summation rule for any kernel ``K(dx, dy, dz)`` is

    K_prism = Σ_{a, b, c ∈ {0, 1}}  (-1)^(a+b+c) · K(x_{a+1} − x_s,
                                                     y_{b+1} − y_s,
                                                     z_{c+1} − z_s)

implemented by :func:`_corner_sum`. Each kernel function provides only
the integrand evaluated at one corner; ``_corner_sum`` does the sign
bookkeeping.

Numerical guards:
    - ``_safe_r``: ``√(dx² + dy² + dz² + ε)`` so r is never exactly 0
      (station on a prism corner).
    - ``_safe_log``: ``log(|x| + ε)`` to prevent ``log(0)``.
    - ``_safe_arctan_quotient``: ``arctan(y / (sign(x) · (|x| + ε)))``,
      the *principal-value* form, documented as the
      canonical Nagy/geoana form; the more common ``atan2`` form
      diverges by ±π for ``z·r < 0`` (mass below observation), which
      produces geometry-dependent magnitude/sign errors.

Reference: Nagy 2000, *J. Geodesy* 74: "The gravitational potential
and its derivatives for the prism".

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
from torch import Tensor


# Physical constants: single source of truth in helpers.py.
from .helpers import EPS, G_SI, M_TO_EOTVOS, M_TO_MGAL

# Nagy-kernel near-corner numerical floor. Intentionally LARGER than the core
# ``EPS`` (1e-30): the 3-D prism log/arctan quantities sit at a coarser
# scale than the 2-D Talwani line integral, so this floor is kernel-specific.
_NAGY_EPS = 1e-20

_Corner = tuple[tuple[int, Tensor], tuple[int, Tensor], tuple[int, Tensor]]
_CornerKernel = Callable[[Tensor, Tensor, Tensor], Tensor]


# --- shared numerical helpers ---------------------------------------------


def _safe_r(dx: Tensor, dy: Tensor, dz: Tensor) -> Tensor:
    """``√(dx² + dy² + dz² + ε)``: guarantees ``r > 0``."""
    return torch.sqrt(dx * dx + dy * dy + dz * dz + _NAGY_EPS)


def _safe_log(x: Tensor) -> Tensor:
    """``log(|x| + ε)``: prevents ``log(0)`` from corners colliding with
    observation points (rare in practice but guards against edge cases)."""
    return torch.log(x.abs() + _NAGY_EPS)


def _safe_arctan_quotient(y: Tensor, x: Tensor) -> Tensor:
    """
    Principal-value ``arctan(y/x)`` ∈ (-π/2, π/2), safe at ``x = 0``.

    Use this (**not** ``atan2``) for the Nagy ``arctan(xy/(zr))``
    term. ``atan2`` ∈ (-π, π] introduces ±π jumps where mass sits below
    the observation (``z·r < 0``) and breaks agreement with Newton's law
    and independent reference kernels on common geometries.
    """
    sign_x = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    return torch.arctan(y / (sign_x * (x.abs() + _NAGY_EPS)))


def _corners(
    x1: Tensor,
    x2: Tensor,
    y1: Tensor,
    y2: Tensor,
    z1: Tensor,
    z2: Tensor,
) -> Iterator[_Corner]:
    """
    Yield the 8 prism corners with their parity signs ``(±1, ±1, ±1)``.

    Sign convention from Nagy 2000: ``(x1, y1, z1)`` gets ``(+, +, +)``,
    ``(x2, y1, z1)`` gets ``(−, +, +)``, etc., alternation along each
    axis. The combined parity ``sign_x · sign_y · sign_z`` is what
    multiplies the corner contribution in :func:`_corner_sum`.
    """
    for sign_x, xc in ((1, x1), (-1, x2)):
        for sign_y, yc in ((1, y1), (-1, y2)):
            for sign_z, zc in ((1, z1), (-1, z2)):
                yield (sign_x, xc), (sign_y, yc), (sign_z, zc)


def _corner_sum(
    obs_x: Tensor,
    obs_y: Tensor,
    obs_z: Tensor,
    x1: Tensor,
    x2: Tensor,
    y1: Tensor,
    y2: Tensor,
    z1: Tensor,
    z2: Tensor,
    kernel_fn: _CornerKernel,
) -> Tensor:
    """
    8-corner summation of a kernel function.

    Args:
        obs_x, obs_y, obs_z: observation coords, shape ``(n_obs, 1)``.
        x1, x2, y1, y2, z1, z2: prism bounds, shape ``(1, n_cells)`` or
            broadcastable.
        kernel_fn: callable ``(dx, dy, dz) → Tensor`` of shape
            ``(n_obs, n_cells)``.

    Returns:
        Result of shape ``(n_obs, n_cells)``.
    """
    out = torch.zeros(
        obs_x.shape[0],
        x1.shape[-1],
        device=obs_x.device,
        dtype=obs_x.dtype,
    )
    for (sign_x, xc), (sign_y, yc), (sign_z, zc) in _corners(x1, x2, y1, y2, z1, z2):
        dx = xc - obs_x
        dy = yc - obs_y
        dz = zc - obs_z
        out = out + (sign_x * sign_y * sign_z) * kernel_fn(dx, dy, dz)
    return out


# --- Gravity kernels (Nagy 2000) -----------------------------------------


def _gz_kernel(
    obs_x: Tensor,
    obs_y: Tensor,
    obs_z: Tensor,
    x1: Tensor,
    x2: Tensor,
    y1: Tensor,
    y2: Tensor,
    z1: Tensor,
    z2: Tensor,
) -> Tensor:
    """gz kernel: Nagy 2000 closed-form integral, unit density.

    The third term uses :func:`_safe_arctan_quotient` not ``atan2`` to
    avoid the ±π discontinuity when mass lies below the observation.
    Returns shape ``(n_obs, n_cells)``.
    """

    def kernel_fn(dx: Tensor, dy: Tensor, dz: Tensor) -> Tensor:
        r = _safe_r(dx, dy, dz)
        return (
            dx * _safe_log(dy + r)
            + dy * _safe_log(dx + r)
            - dz * _safe_arctan_quotient(dx * dy, dz * r)
        )

    return _corner_sum(obs_x, obs_y, obs_z, x1, x2, y1, y2, z1, z2, kernel_fn)


def _gzz_kernel(
    obs_x: Tensor,
    obs_y: Tensor,
    obs_z: Tensor,
    x1: Tensor,
    x2: Tensor,
    y1: Tensor,
    y2: Tensor,
    z1: Tensor,
    z2: Tensor,
) -> Tensor:
    """Vertical second derivative ``g_zz``, unit density."""

    def kernel_fn(dx: Tensor, dy: Tensor, dz: Tensor) -> Tensor:
        r = _safe_r(dx, dy, dz)
        return _safe_arctan_quotient(dx * dy, dz * r)

    return _corner_sum(obs_x, obs_y, obs_z, x1, x2, y1, y2, z1, z2, kernel_fn)


def _gxx_kernel(
    obs_x: Tensor,
    obs_y: Tensor,
    obs_z: Tensor,
    x1: Tensor,
    x2: Tensor,
    y1: Tensor,
    y2: Tensor,
    z1: Tensor,
    z2: Tensor,
) -> Tensor:
    """Horizontal second derivative ``g_xx`` (cycled gzz)."""

    def kernel_fn(dx: Tensor, dy: Tensor, dz: Tensor) -> Tensor:
        r = _safe_r(dx, dy, dz)
        return _safe_arctan_quotient(dy * dz, dx * r)

    return _corner_sum(obs_x, obs_y, obs_z, x1, x2, y1, y2, z1, z2, kernel_fn)


def _gyy_kernel(
    obs_x: Tensor,
    obs_y: Tensor,
    obs_z: Tensor,
    x1: Tensor,
    x2: Tensor,
    y1: Tensor,
    y2: Tensor,
    z1: Tensor,
    z2: Tensor,
) -> Tensor:
    """Horizontal second derivative ``g_yy`` (cycled gzz)."""

    def kernel_fn(dx: Tensor, dy: Tensor, dz: Tensor) -> Tensor:
        r = _safe_r(dx, dy, dz)
        return _safe_arctan_quotient(dx * dz, dy * r)

    return _corner_sum(obs_x, obs_y, obs_z, x1, x2, y1, y2, z1, z2, kernel_fn)


def _gxy_kernel(
    obs_x: Tensor,
    obs_y: Tensor,
    obs_z: Tensor,
    x1: Tensor,
    x2: Tensor,
    y1: Tensor,
    y2: Tensor,
    z1: Tensor,
    z2: Tensor,
) -> Tensor:
    """Cross derivative ``g_xy``."""

    def kernel_fn(dx: Tensor, dy: Tensor, dz: Tensor) -> Tensor:
        r = _safe_r(dx, dy, dz)
        return -_safe_log(dz + r)

    return _corner_sum(obs_x, obs_y, obs_z, x1, x2, y1, y2, z1, z2, kernel_fn)


def _gxz_kernel(
    obs_x: Tensor,
    obs_y: Tensor,
    obs_z: Tensor,
    x1: Tensor,
    x2: Tensor,
    y1: Tensor,
    y2: Tensor,
    z1: Tensor,
    z2: Tensor,
) -> Tensor:
    """Cross derivative ``g_xz``."""

    def kernel_fn(dx: Tensor, dy: Tensor, dz: Tensor) -> Tensor:
        r = _safe_r(dx, dy, dz)
        return -_safe_log(dy + r)

    return _corner_sum(obs_x, obs_y, obs_z, x1, x2, y1, y2, z1, z2, kernel_fn)


def _gyz_kernel(
    obs_x: Tensor,
    obs_y: Tensor,
    obs_z: Tensor,
    x1: Tensor,
    x2: Tensor,
    y1: Tensor,
    y2: Tensor,
    z1: Tensor,
    z2: Tensor,
) -> Tensor:
    """Cross derivative ``g_yz``."""

    def kernel_fn(dx: Tensor, dy: Tensor, dz: Tensor) -> Tensor:
        r = _safe_r(dx, dy, dz)
        return -_safe_log(dx + r)

    return _corner_sum(obs_x, obs_y, obs_z, x1, x2, y1, y2, z1, z2, kernel_fn)


# --- Public API: forward operators on flat cell arrays --------------------


def gravity_prism_gz(
    obs_x: Tensor,
    obs_y: Tensor,
    obs_z: Tensor,
    x1: Tensor,
    x2: Tensor,
    y1: Tensor,
    y2: Tensor,
    z1: Tensor,
    z2: Tensor,
    density: Tensor,
) -> Tensor:
    """
    Vertical gravity ``g_z`` from a collection of rectangular prisms.

    Args:
        obs_x, obs_y, obs_z: observation coords ``(n_obs,)``.
        x1, x2, y1, y2, z1, z2: prism bounds ``(n_cells,)``.
        density: ``(n_cells,)`` in kg/m³.

    Returns:
        ``g_z`` in mGal, shape ``(n_obs,)``.
    """
    ox = obs_x.unsqueeze(1)
    oy = obs_y.unsqueeze(1)
    oz = obs_z.unsqueeze(1)
    xa = x1.unsqueeze(0)
    xb = x2.unsqueeze(0)
    ya = y1.unsqueeze(0)
    yb = y2.unsqueeze(0)
    za = z1.unsqueeze(0)
    zb = z2.unsqueeze(0)

    K = _gz_kernel(ox, oy, oz, xa, xb, ya, yb, za, zb)  # (n_obs, n_cells)
    return (K * density.unsqueeze(0)).sum(dim=1) * G_SI * M_TO_MGAL


def gravity_prism_gradient_tensor(
    obs_x: Tensor,
    obs_y: Tensor,
    obs_z: Tensor,
    x1: Tensor,
    x2: Tensor,
    y1: Tensor,
    y2: Tensor,
    z1: Tensor,
    z2: Tensor,
    density: Tensor,
) -> dict[str, Tensor]:
    """
    Full gravity gradient tensor ``(g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)``.

    Returns a dict of six components, each ``(n_obs,)`` in Eotvos
    (1 E = 1e-9 / s²).

    Args:
        obs_x / obs_y / obs_z: observation coordinates [m].
        x1 / x2, y1 / y2, z1 / z2: prism face coordinates [m].
        density: prism density [kg/m^3].
    """
    ox = obs_x.unsqueeze(1)
    oy = obs_y.unsqueeze(1)
    oz = obs_z.unsqueeze(1)
    xa, xb = x1.unsqueeze(0), x2.unsqueeze(0)
    ya, yb = y1.unsqueeze(0), y2.unsqueeze(0)
    za, zb = z1.unsqueeze(0), z2.unsqueeze(0)

    rho = density.unsqueeze(0)
    factor = G_SI * M_TO_EOTVOS

    return {
        "gxx": (_gxx_kernel(ox, oy, oz, xa, xb, ya, yb, za, zb) * rho).sum(1) * factor,
        "gyy": (_gyy_kernel(ox, oy, oz, xa, xb, ya, yb, za, zb) * rho).sum(1) * factor,
        "gzz": (_gzz_kernel(ox, oy, oz, xa, xb, ya, yb, za, zb) * rho).sum(1) * factor,
        "gxy": (_gxy_kernel(ox, oy, oz, xa, xb, ya, yb, za, zb) * rho).sum(1) * factor,
        "gxz": (_gxz_kernel(ox, oy, oz, xa, xb, ya, yb, za, zb) * rho).sum(1) * factor,
        "gyz": (_gyz_kernel(ox, oy, oz, xa, xb, ya, yb, za, zb) * rho).sum(1) * factor,
    }


__all__ = [
    "EPS",
    "G_SI",
    "M_TO_EOTVOS",
    "M_TO_MGAL",
    "gravity_prism_gradient_tensor",
    "gravity_prism_gz",
]
