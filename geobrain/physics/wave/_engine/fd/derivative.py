"""
Staggered-grid first-derivative operators (2-D), arbitrary order.

Field layout (acoustic velocity-stress):

    p   at integer nodes      (i,   j)
    vx  staggered in x        (i,   j+1/2)
    vz  staggered in z        (i+1/2, j)

Two adjoint staggered operators per axis:

- **forward** ``D+`` : value at integer nodes → derivative at the ``+1/2`` node.
  Used for ``∂p/∂x`` (updates ``vx``) and ``∂p/∂z`` (updates ``vz``)::

      D+ f[i] = (1/h) Σ_k c_k ( f[i+k] − f[i−k+1] )

- **backward** ``D−`` : value at ``+1/2`` nodes → derivative at the integer node.
  Used for ``∂vx/∂x`` and ``∂vz/∂z`` (update ``p``)::

      D− f[i] = (1/h) Σ_k c_k ( f[i+k−1] − f[i−k] )

``D+`` and ``D−`` are discrete adjoints (up to sign), which is what makes the
leapfrog scheme non-dissipative. All operators are pure and differentiable
(slice + concat shifts with zero fill), so ``torch.autograd`` flows through them.

Tensors are shape ``(..., nz, nx)``; the x-axis is ``-1`` and the z-axis ``-2``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def _shift(f: Tensor, s: int, axis: int) -> Tensor:
    """Shift ``f`` along ``axis`` so that ``out[i] = f[i - s]``, zero-filled.

    ``s > 0`` moves content toward larger indices (zeros appear at the front);
    ``s < 0`` toward smaller indices (zeros at the back). Fully differentiable.
    """
    if s == 0:
        return f
    n = f.shape[axis]
    if abs(s) >= n:
        return torch.zeros_like(f)
    if s > 0:
        keep = f.narrow(axis, 0, n - s)
        pad = torch.zeros_like(f.narrow(axis, 0, s))
        return torch.cat([pad, keep], dim=axis)
    a = -s
    keep = f.narrow(axis, a, n - a)
    pad = torch.zeros_like(f.narrow(axis, 0, a))
    return torch.cat([keep, pad], dim=axis)


def _stagger_forward(f: Tensor, coeffs: Sequence[float], h: float, axis: int) -> Tensor:
    """``D+`` along ``axis``: ``Σ_k c_k (f[i+k] − f[i−k+1]) / h``."""
    out: Tensor | None = None
    for k, ck in enumerate(coeffs, start=1):
        term = _shift(f, -k, axis) - _shift(f, k - 1, axis)
        out = ck * term if out is None else out + ck * term
    assert out is not None  # coeffs is non-empty
    return out / h


def _stagger_backward(f: Tensor, coeffs: Sequence[float], h: float, axis: int) -> Tensor:
    """``D−`` along ``axis``: ``Σ_k c_k (f[i+k−1] − f[i−k]) / h``."""
    out: Tensor | None = None
    for k, ck in enumerate(coeffs, start=1):
        term = _shift(f, -(k - 1), axis) - _shift(f, k, axis)
        out = ck * term if out is None else out + ck * term
    assert out is not None
    return out / h


def diff_x_forward(f: Tensor, coeffs: Sequence[float], dx: float) -> Tensor:
    """``∂f/∂x`` at the ``+1/2`` node (integer-node input)."""
    return _stagger_forward(f, coeffs, dx, axis=-1)


def diff_x_backward(f: Tensor, coeffs: Sequence[float], dx: float) -> Tensor:
    """``∂f/∂x`` at the integer node (``+1/2``-node input)."""
    return _stagger_backward(f, coeffs, dx, axis=-1)


def _z_axis(f: Tensor) -> int:
    """Array axis of the FIRST spatial dim (z). Fields are laid out
    ``(batch, channel, *spatial)``, so z is ``-2`` in 2-D ``(...,nz,nx)`` and
    ``-3`` in 3-D ``(...,nz,ny,nx)``, while x stays ``-1`` and y (3-D) is ``-2``.
    (Previously hard-coded to ``-2``, which on a 3-D field differenced along the
    y-axis, a physics bug masked by cubic/x-symmetric test grids.)"""
    return -int(f.ndim - 2)


def diff_z_forward(f: Tensor, coeffs: Sequence[float], dz: float) -> Tensor:
    """``∂f/∂z`` at the ``+1/2`` node (integer-node input)."""
    return _stagger_forward(f, coeffs, dz, axis=_z_axis(f))


def diff_z_backward(f: Tensor, coeffs: Sequence[float], dz: float) -> Tensor:
    """``∂f/∂z`` at the integer node (``+1/2``-node input)."""
    return _stagger_backward(f, coeffs, dz, axis=_z_axis(f))


# --- free-surface (image-method) z-derivatives at the top edge -------------
# A pressure-release / traction-free surface is imposed by the *image method*:
# mirror the field into ``h = len(coeffs)`` ghost rows above the top edge (row 0)
# with the parity the boundary condition demands, apply the ordinary staggered
# operator on the extended field, then crop back. This replaces the implicit
# zero-fill of :func:`_shift` (which forces a *rigid* wall (the vz ghost is 0) # giving reflection ``R = +1``) with the correct free-surface image (``R = -1``).
# Only the ``h`` rows nearest the surface change; the deep (bottom) edge keeps the
# identical zero-fill of the plain operators, so the interior/bottom are untouched.
def diff_z_forward_freesurface_odd(
    f: Tensor, coeffs: Sequence[float], dz: float
) -> Tensor:
    """``∂f/∂z`` at ``+1/2`` nodes with an ODD image of ``f`` about integer row 0.

    ``f`` (e.g. acoustic pressure) is co-located with the surface node, which is a
    zero (pressure release); the ghost above the surface is ``f[-m] = -f[m]``. Used
    by the velocity update ``∂p/∂z`` so the top-most ``vz`` sees the mirrored
    pressure. Differentiable (flip / cat / narrow)."""
    az = _z_axis(f)
    h = len(coeffs)
    ghost = -torch.flip(f.narrow(az, 1, h), dims=[az])   # rows -h..-1 = -f[h]..-f[1]
    g = torch.cat([ghost, f], dim=az)
    d = _stagger_forward(g, coeffs, dz, az)
    return d.narrow(az, h, f.shape[az])


def diff_z_backward_freesurface_even(
    f: Tensor, coeffs: Sequence[float], dz: float
) -> Tensor:
    """``∂f/∂z`` at integer nodes with an EVEN image of the half-node field ``f``
    about the surface plane ``z = 0`` (row-0 integer node).

    ``f`` (e.g. ``vz``, whose samples sit at ``(i+1/2)``) is mirrored symmetrically,
    ``f[-1-m] = f[m]``, so the vertical strain ``∂vz/∂z`` at the surface row
    vanishes, consistent with a pressure node. Used by the pressure update
    ``∂vz/∂z``. Differentiable (flip / cat / narrow)."""
    az = _z_axis(f)
    h = len(coeffs)
    ghost = torch.flip(f.narrow(az, 0, h), dims=[az])    # rows -h..-1 = f[h-1]..f[0]
    g = torch.cat([ghost, f], dim=az)
    d = _stagger_backward(g, coeffs, dz, az)
    return d.narrow(az, h, f.shape[az])


# --- elastic traction-free (Levander stress-imaging) z-derivatives ---------
# The elastic free surface couples P↔SV, so no single field parity satisfies both
# traction conditions (σzz = σxz = 0). The Levander (1988) / Robertsson (1996)
# staggered stress-imaging construction mirrors the fields about the surface with
# fixed, component-specific parities and forces the on-surface normal stress to
# zero (done in the equation ``step``). With the surface on the integer z-row 0
# (where σxx, σzz, vx live; vz, σxz sit half a cell below at ``i+1/2``):
#
#   ODD (antisymmetric) about z = 0:  σzz (integer, =0 on row 0), σxz (half), vz (half)
#   EVEN (symmetric)   about z = 0:  vx (integer)
#
# The pressure-release acoustic ops above already provide the two the acoustic
# path needs (integer-odd ``forward``, half-even ``backward``). The elastic path
# adds the other two parities below. Only the ``h`` surface-adjacent rows change;
# the bottom/interior keep the plain operator's zero-fill.
def diff_z_backward_freesurface_odd(
    f: Tensor, coeffs: Sequence[float], dz: float
) -> Tensor:
    """``∂f/∂z`` at integer nodes with an ODD image of the half-node field ``f``
    about the surface plane ``z = 0`` (row-0 integer node).

    ``f`` (half-node samples at ``(i+1/2)``) is mirrored antisymmetrically,
    ``f[-1-m] = -f[m]``, so ``f`` is zero on the surface plane. Used by the elastic
    free surface for ``∂σxz/∂z`` (vx update) and ``∂vz/∂z`` (σxx/σzz update).
    Differentiable (flip / cat / narrow)."""
    az = _z_axis(f)
    h = len(coeffs)
    ghost = -torch.flip(f.narrow(az, 0, h), dims=[az])   # rows -h..-1 = -f[h-1]..-f[0]
    g = torch.cat([ghost, f], dim=az)
    d = _stagger_backward(g, coeffs, dz, az)
    return d.narrow(az, h, f.shape[az])


def diff_z_forward_freesurface_even(
    f: Tensor, coeffs: Sequence[float], dz: float
) -> Tensor:
    """``∂f/∂z`` at ``+1/2`` nodes with an EVEN image of the integer-node field
    ``f`` about the surface plane ``z = 0`` (row-0 integer node).

    ``f`` (e.g. ``vx``, integer z-node co-located with the surface) is mirrored
    symmetrically, ``f[-m] = f[m]``. Used by the elastic free surface for
    ``∂vx/∂z`` (σxz update), the SV partner of the acoustic integer-odd image.
    Differentiable (flip / cat / narrow)."""
    az = _z_axis(f)
    h = len(coeffs)
    ghost = torch.flip(f.narrow(az, 1, h), dims=[az])    # rows -h..-1 = f[h]..f[1]
    g = torch.cat([ghost, f], dim=az)
    d = _stagger_forward(g, coeffs, dz, az)
    return d.narrow(az, h, f.shape[az])


# --- bilinear averages between integer and half-half subgrids (2-D) -------
# Needed by tilted-anisotropy (TTI), whose off-diagonal stiffness couples the
# integer-node normal stresses with the (x½, z½) shear node. Both are 2nd-order
# accurate and differentiable.
def avg_hh_to_ii(f: Tensor) -> Tensor:
    """Average a ``(x½, z½)`` field to the integer node (4-point)."""
    fx = f + _shift(f, 1, -1)
    return 0.25 * (fx + _shift(fx, 1, -2))


def avg_ii_to_hh(f: Tensor) -> Tensor:
    """Average an integer-node field to the ``(x½, z½)`` node (4-point)."""
    fx = f + _shift(f, -1, -1)
    return 0.25 * (fx + _shift(fx, -1, -2))


def avg1_hh_to_ii(f: Tensor, axis: int) -> Tensor:
    """1-D half→integer average along ``axis``: ``0.5(f[i-½] + f[i+½])``."""
    return 0.5 * (f + _shift(f, 1, axis))


def avg1_ii_to_hh(f: Tensor, axis: int) -> Tensor:
    """1-D integer→half average along ``axis``: ``0.5(f[i] + f[i+1])``."""
    return 0.5 * (f + _shift(f, -1, axis))


# --- 3-D adds a y axis at -2 (tensors are (..., nz, ny, nx): z=-3, y=-2, x=-1) --
def diff_y_forward(f: Tensor, coeffs: Sequence[float], dy: float) -> Tensor:
    """``∂f/∂y`` at the ``+1/2`` node (integer-node input); 3-D only."""
    return _stagger_forward(f, coeffs, dy, axis=-2)


def diff_y_backward(f: Tensor, coeffs: Sequence[float], dy: float) -> Tensor:
    """``∂f/∂y`` at the integer node (``+1/2``-node input); 3-D only."""
    return _stagger_backward(f, coeffs, dy, axis=-2)
