"""
Split-field CPML for 3-D grids.

Same construction as the 2-D :mod:`wave.boundary.cpml` (per-axis damping ``d``,
stretch ``κ`` and CFS shift ``α`` → recursion coefficients ``b, a``), extended
with a third axis ``y``. Tensors broadcast against ``(B, 1, nz, ny, nx)``: x as
``(1,1,1,1,nx)``, y as ``(1,1,1,ny,1)``, z as ``(1,1,nz,1,1)``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .cpml import FloatArray, ProfileTensors, _ba_from_profile, damping_profile_1d


@dataclass
class CPML3D:
    """Precomputed 3-D CPML recursion coefficients, broadcast-ready."""

    bx_int: Tensor
    ax_int: Tensor
    kx_int: Tensor
    by_int: Tensor
    ay_int: Tensor
    ky_int: Tensor
    bz_int: Tensor
    az_int: Tensor
    kz_int: Tensor
    bx_half: Tensor
    ax_half: Tensor
    kx_half: Tensor
    by_half: Tensor
    ay_half: Tensor
    ky_half: Tensor
    bz_half: Tensor
    az_half: Tensor
    kz_half: Tensor

    @staticmethod
    def stretch(
        deriv: Tensor, psi: Tensor, b: Tensor, a: Tensor, kappa: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply the CPML stretch; return ``(deriv/κ + ψ_new, ψ_new)``."""
        psi_new = b * psi + a * deriv
        return deriv / kappa + psi_new, psi_new


def build_cpml_3d(
    nz: int, ny: int, nx: int,
    nabc: int, dz: float, dy: float, dx: float, dt: float, vmax: float,
    *,
    reflection: float = 1.0e-4, npower: float = 2.0,
    kappa_max: float = 1.0, alpha_max: float = 0.0,
    free_surface: bool = False,
    device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
) -> CPML3D:
    """Construct a :class:`CPML3D` for a padded ``(nz, ny, nx)`` grid."""
    common = dict(reflection=reflection, npower=npower,
                  kappa_max=kappa_max, alpha_max=alpha_max)

    def make(
        n: int,
        dh: float,
        axis: str,
        *,
        pml_left: bool,
        pml_right: bool,
    ) -> ProfileTensors:
        out: ProfileTensors = {}
        for tag, off in (("int", 0.0), ("half", 0.5)):
            d, k, al = damping_profile_1d(
                n, nabc, dh, vmax, pml_left=pml_left, pml_right=pml_right,
                offset=off, **common,
            )
            b, a = _ba_from_profile(d, k, al, dt)
            if axis == "x":
                shape = (1, 1, 1, 1, n)
            elif axis == "y":
                shape = (1, 1, 1, n, 1)
            else:
                shape = (1, 1, n, 1, 1)

            def t(arr: FloatArray) -> Tensor:
                return torch.as_tensor(arr, dtype=dtype, device=device).reshape(shape)

            out[tag] = (t(b), t(a), t(k))
        return out

    x = make(nx, dx, "x", pml_left=True, pml_right=True)
    y = make(ny, dy, "y", pml_left=True, pml_right=True)
    z = make(nz, dz, "z", pml_left=not free_surface, pml_right=True)
    return CPML3D(
        bx_int=x["int"][0], ax_int=x["int"][1], kx_int=x["int"][2],
        by_int=y["int"][0], ay_int=y["int"][1], ky_int=y["int"][2],
        bz_int=z["int"][0], az_int=z["int"][1], kz_int=z["int"][2],
        bx_half=x["half"][0], ax_half=x["half"][1], kx_half=x["half"][2],
        by_half=y["half"][0], ay_half=y["half"][1], ky_half=y["half"][2],
        bz_half=z["half"][0], az_half=z["half"][1], kz_half=z["half"][2],
    )
