"""
Source-RHS builders for the 3D TEM operator.

Discretise a step-off vertical magnetic dipole (VMD) as a 4-edge
rectangular horizontal current loop on a Yee mesh. For an explicitly
normalised vertical magnetic dipole, the equivalent loop has area ``A = dx * dy``
and carries current ``I = magnetic_moment_am2 / A``; the loop perimeter contributes
``± I * L_edge`` to four edges (two x-edges, two y-edges) and zero
elsewhere.

Edge index / family ordering matches
:func:`~geobrain.physics.em.numerics.finite_volume.curl_curl.yee_curl_e_to_f`:

- Ex:    ``i + j*nx + k*nx*(ny + 1)``           with ``i ∈ [0, nx)``, ``j ∈ [0, ny]``, ``k ∈ [0, nz]``.
- Ey:    ``n_ex + i + j*(nx + 1) + k*(nx + 1)*ny``    with ``i ∈ [0, nx]``, ``j ∈ [0, ny)``, ``k ∈ [0, nz]``.
- Ez:    ``n_ex + n_ey + i + j*(nx + 1) + k*(nx + 1)*(ny + 1)``  with ``i ∈ [0, nx]``, ``j ∈ [0, ny]``, ``k ∈ [0, nz)``.

Right-handed loop about ``+z`` (Stokes' theorem with normal ``+z``,
counter-clockwise viewed from ``+z``). Starting at the lower-left
corner ``(i_cx, i_cy)`` of the host cell at z-node ``i_nz`` the loop is:

1. +x along ex(i_cx,     i_cy,     i_nz):   current = +I, leg length = ``dx``.
2. +y along ey(i_cx + 1, i_cy,     i_nz):   current = +I, leg length = ``dy``.
3. -x along ex(i_cx,     i_cy + 1, i_nz):   current = -I, leg length = ``dx``.
4. -y along ey(i_cx,     i_cy,     i_nz):   current = -I, leg length = ``dy``.

Per-edge value follows the mimetic convention ``s_e = sign * I *
L_edge``. The edge-volume division (for a bare-mass time-stepper) is
deferred to the operator, so this helper exposes a units-clean
integrated current.

The source is constant in time, TEM3D modelling switches it off at
``t = 0`` (step-off), so this vector represents the static ``t < 0``
current that drives the DC initial-condition solve.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Sequence

import torch

from geobrain.core import GeoBrainError
from geobrain.mesh import TensorMesh
from geobrain.physics.em.numerics.finite_volume.curl_curl import (
    yee_edge_count,
)

__all__ = ["build_vmd_step_off_source"]


def build_vmd_step_off_source(
    mesh: TensorMesh,
    location: Sequence[float],
    magnetic_moment_am2: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Yee-discretised vertical (+z) magnetic dipole step-off source.

    Discretises a vertical magnetic dipole as a 4-edge horizontal
    current loop in the xy-plane. The loop sits in the cell containing
    ``location`` (nearest cell in xy, nearest z-node in z) with area
            ``dx * dy`` and current ``I = magnetic_moment_am2 / (dx * dy)``. Right-handed
    about ``+z`` so the equivalent magnetic moment points ``+z``.

    Args:
        mesh: 3D :class:`~geobrain.mesh.TensorMesh` with shape ``(nz, ny, nx)``
            and uniform spacing ``(dz, dy, dx)``.
        location: ``(xs, ys, zs)`` transmitter position in mesh coordinates.
            Coordinates measure from the origin (corner of cell ``(0, 0,
            0)``) so cell ``(i, j, k)`` spans ``x ∈ [i*dx, (i+1)*dx]``,
            ``y ∈ [j*dy, (j+1)*dy]``, ``z ∈ [k*dz, (k+1)*dz]``.
        magnetic_moment_am2: Explicit dipole moment in A·m².
        device, dtype: Standard torch placement / precision controls. ``dtype`` must
            be a real floating type.

    Returns:
        ``(n_edges,)`` source vector with exactly 4 nonzero entries
        (two x-edge values ``± I * dx``, two y-edge values
        ``± I * dy``). All z-edge entries are zero.

    Raises:
        ValueError: If ``mesh`` is not 3D, ``location`` does not have length 3, or
            the host cell has zero area.

    Notes:
    Produces the "mimetic integrated current" stage. The per-edge
    dual-volume division (for a bare-mass time-stepper) is left to the
    operator, so the output here is units-clean
    (current × length, A·m).
    """
    if mesh.n_dim != 3:
        raise GeoBrainError(
            "build_vmd_step_off_source requires a 3D TensorMesh, got "
            f"n_dim={mesh.n_dim}",
            object_name="build_vmd_step_off_source",
            field="mesh",
            expected="3D TensorMesh", actual=mesh.n_dim,
        )
    loc = tuple(float(v) for v in location)
    if len(loc) != 3:
        raise GeoBrainError(
            "location must be a 3-tuple (xs, ys, zs); got length "
            f"{len(loc)}",
            object_name="build_vmd_step_off_source",
            field="location",
            expected="length-3 sequence (xs, ys, zs)", actual=len(loc),
        )

    nz, ny, nx = mesh.shape
    dz, dy, dx = mesh.spacing
    xs, ys, zs = loc

    area = dx * dy
    if area <= 0.0:
        raise GeoBrainError(
            "Source cell has zero area; check mesh spacing",
            object_name="build_vmd_step_off_source",
            field="area",
            expected="positive cell area (dx * dy)", actual=area,
        )

    # Binding convention is deliberately asymmetric across axes (do NOT unify
    # to floor): x/y use floor() because they are CELL indices (range 0..n-1);
    # z uses round() = nearest NODE because the horizontal loop edges lie on a
    # Yee z-NODE plane (range 0..nz), so nearest-node is the geometrically
    # correct snap there. Flipping z to floor() would shift the loop down half a
    # cell: verify against the TEM loop-placement test before touching this.
    # Coordinates are physical: the mesh origin (engine frame (oz, oy, ox))
    # is subtracted before snapping: subtracting an exact 0.0 is a bitwise
    # no-op, so zero-origin meshes keep the historical indices exactly.
    oz, oy, ox = mesh.origin
    i_cx = int(min(max(int((xs - ox) // dx), 0), nx - 1))
    i_cy = int(min(max(int((ys - oy) // dy), 0), ny - 1))
    i_nz = int(min(max(round((zs - oz) / dz), 0), nz))

    n_ex, n_ey, _n_ez = yee_edge_count(mesh)
    n_edges = n_ex + n_ey + _n_ez

    # Loop current (amperes).
    current = float(magnetic_moment_am2) / area

    # The four horizontal edges of cell (i_cx, i_cy, *) at z-node i_nz:
    # Ex flat: i + j * nx + k * nx * (ny + 1)
    # Ey flat: n_ex + i + j * (nx + 1) + k * (nx + 1) * ny
    ex_bot = i_cx + i_cy * nx + i_nz * (nx * (ny + 1))
    ex_top = i_cx + (i_cy + 1) * nx + i_nz * (nx * (ny + 1))
    ey_left = n_ex + i_cx + i_cy * (nx + 1) + i_nz * ((nx + 1) * ny)
    ey_right = n_ex + (i_cx + 1) + i_cy * (nx + 1) + i_nz * ((nx + 1) * ny)

    # Right-handed loop about +z (CCW viewed from +z):
    #   +x along ex_bot   →  +I * dx
    #   +y along ey_right →  +I * dy
    #   -x along ex_top   →  -I * dx
    #   -y along ey_left  →  -I * dy
    s_e = torch.zeros(n_edges, dtype=dtype, device=device)
    s_e[ex_bot] = +current * dx
    s_e[ey_right] = +current * dy
    s_e[ex_top] = -current * dx
    s_e[ey_left] = -current * dy
    return s_e
