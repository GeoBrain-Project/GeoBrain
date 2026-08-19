# pyright: reportPrivateImportUsage=false
"""Primary/secondary potential split (singularity removal) for the DC FV seam.

The discrete cell-centred FV DC solution's dominant error is the ``1/r``
singularity at the current electrodes, the worst-resolved feature, especially
on coarse or irregular tetrahedral meshes where the nearest cell centre is far
from the true source and the harmonic-mean TPFA stencil cannot represent the
sharp near-field cusp. The classical cure (Lowry/Coggon-style *singularity
removal*, the form node-based ERT codes implement at
mesh nodes) splits the potential into a known singular *primary* part carried
analytically and a smooth *secondary* part solved on the same discrete grid::

    u = u_p + u_s,        A(σ) u_s = b − A(σ) u_p

with ``u_p`` the analytic potential of the electrode pair in a HOMOGENEOUS
half-space of a reference conductivity ``σ_ref``, evaluated at the cell centres.
Where ``σ == σ_ref`` and away from the electrodes the residual source
``b − A u_p`` is ≈ 0; the singular near-field is then carried exactly by ``u_p``
and the discrete solve only has to resolve the smooth correction ``u_s``.

This module supplies the analytic ``u_p`` (pure geometry + scalars, fully
detached). For an electrode at ``x_e`` carrying current ``I`` over a half-space
of conductivity ``σ_ref`` with a free surface at ``z = free_surface_coord`` (the
GeoBrain frame uses depth-first ``(z, y, x)`` coordinates with ``z`` the depth,
so the free surface is ``z = 0``), the half-space Green's function is the source
plus its mirror image across the surface plane::

    φ(x) = I / (4π σ_ref) · (1/|x − x_e| + 1/|x − x_e'|)

where ``x_e'`` is ``x_e`` reflected across the surface plane (``z → −z`` for a
surface at ``z = 0``). For a SURFACE electrode (``z_e = 0``) the two terms
coincide and collapse to the textbook ``I / (2π σ_ref r)``. The source A injects
``+I``, the sink B ``−I``; the returned ``u_p`` is the superposition of both.

Self-cell regularisation
------------------------
Node-based ERT codes evaluate ``u_p`` at mesh NODES, which never coincide with a point
electrode. In the cell-centred FV convention ``u_p`` is evaluated at cell
CENTRES, and the binding snaps each electrode to its nearest cell centre, so an
electrode typically sits EXACTLY at a cell centre and the analytic ``1/r`` would
blow up there. We regularise the electrode's OWN cell by replacing ``r = 0`` (and
any ``r`` smaller than the cell's effective radius) with the cell's
volume-equivalent sphere radius::

    r_eff = (3 V_cell / 4π)^(1/3)

i.e. the radius of a sphere with the same volume as the cell. This is the
standard self-potential / self-cell regularisation used in integral-equation and
FV point-source codes: it represents the average ``1/r`` over the finite cell
volume rather than the divergent value at the singular point. The homogeneous
exactness test validates the choice, with ``σ ≡ σ_ref`` the TOTAL receiver
voltage reproduces the analytic half-space answer to high accuracy and ``u_s``
is numerical-noise small.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math

import torch

from geobrain.core.errors import GeoBrainError


__all__ = ["primary_potential", "effective_cell_radius"]


def effective_cell_radius(cell_volumes: torch.Tensor) -> torch.Tensor:
    """Volume-equivalent sphere radius ``r_eff = (3 V / 4π)^(1/3)`` per cell.

    The radius of a sphere with the same volume as each cell, the standard
    self-cell regularisation length for a point source sitting at a cell centre
    (see the module docstring). Pure geometry; returned detached float64.

    Args:
        cell_volumes: ``(n_cells,)`` positive cell volumes (m³).

    Returns:
        ``(n_cells,)`` float64 detached ``r_eff``.
    """
    v = cell_volumes.detach().to(torch.float64).reshape(-1)
    coef = 3.0 / (4.0 * math.pi)
    return (coef * v.clamp_min(0.0)) ** (1.0 / 3.0)


def _monopole_field(
    cell_centers: torch.Tensor,
    electrode: torch.Tensor,
    current: float,
    sigma_ref: float,
    *,
    free_surface_axis: int,
    free_surface_coord: float,
    self_cell: int | None,
    r_eff: torch.Tensor | None,
) -> torch.Tensor:
    """Analytic half-space monopole ``φ`` at every cell centre for one electrode.

    ``φ(x) = I/(4π σ_ref)·(1/r + 1/r')`` with ``r' `` the mirror-image distance
    across the free-surface plane. The electrode's OWN cell (``self_cell``) gets
    its ``r`` floored at ``r_eff[self_cell]`` to avoid the ``1/0`` singularity; a
    surface electrode additionally has its mirror distance floored there (the
    image coincides with the source at the surface).
    """
    centers = cell_centers.to(torch.float64)
    x_e = electrode.to(torch.float64).reshape(-1)

    # Real-image (mirror) electrode across the free-surface plane.
    x_e_mirror = x_e.clone()
    x_e_mirror[free_surface_axis] = 2.0 * float(free_surface_coord) - x_e[free_surface_axis]

    r = torch.linalg.norm(centers - x_e.view(1, -1), dim=1)            # (n_cells,)
    r_img = torch.linalg.norm(centers - x_e_mirror.view(1, -1), dim=1)

    # Self-cell regularisation: floor the electrode's own-cell distance(s) at the
    # cell's effective radius. The direct distance is ~0 at the snapped cell; the
    # mirror distance is ~0 too when the electrode is on (or near) the surface, so
    # floor both at r_eff for that cell. Other cells are untouched.
    if self_cell is not None and r_eff is not None:
        re = r_eff[self_cell]
        r = r.clone()
        r_img = r_img.clone()
        r[self_cell] = torch.clamp_min(r[self_cell], re)
        r_img[self_cell] = torch.clamp_min(r_img[self_cell], re)

    coef = float(current) / (4.0 * math.pi * float(sigma_ref))
    return coef * (1.0 / r + 1.0 / r_img)


def primary_potential(
    cell_centers: torch.Tensor,
    source_pos: torch.Tensor,
    sink_pos: torch.Tensor,
    current: float,
    sigma_ref: float,
    *,
    free_surface_axis: int = 0,
    free_surface_coord: float = 0.0,
    self_cells: tuple[int | None, int | None] | None = None,
    cell_volumes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Analytic primary potential ``u_p`` of a DC electrode pair at cell centres.

    The half-space Green's-function potential of a ``+I`` source at
    ``source_pos`` and a ``−I`` sink at ``sink_pos`` over a homogeneous half-space
    of conductivity ``sigma_ref``, evaluated at every cell centre, with each
    electrode imaged across the free-surface plane (mirror-image method)::

        u_p(x) = I/(4π σ_ref)·(1/|x−A| + 1/|x−A'|)
                 − I/(4π σ_ref)·(1/|x−B| + 1/|x−B'|)

    Pure geometry + scalars; the returned tensor is fully DETACHED (no autograd).
    ``σ_ref`` is a fixed scalar here; the σ-differentiability of the full
    primary/secondary split flows through ``A(σ)`` in the residual source
    ``b − A(σ) u_p`` of the secondary solve, NOT through ``u_p`` itself.

    Args:
        cell_centers: ``(n_cells, n_dim)`` cell-centre coordinates (metres),
            depth-first ``(z, y, x)`` columns (the GeoBrain frame).
        source_pos: ``(n_dim,)`` ``+I`` electrode position (metres, ``(z, y, x)``).
        sink_pos:   ``(n_dim,)`` ``−I`` electrode position (metres, ``(z, y, x)``).
        current:    injected current magnitude ``I`` (A), positive.
        sigma_ref:  reference half-space conductivity ``σ_ref`` (S/m), positive.
        free_surface_axis: axis whose ``≈ free_surface_coord`` plane is the free
            surface (default ``0`` = depth ``z``); each electrode is mirrored
            across it.
        free_surface_coord: coordinate of the free-surface plane along
            ``free_surface_axis`` (default ``0.0``).
        self_cells: ``(source_cell, sink_cell)`` flat cell indices the source and
            sink snap to. The electrode's own cell gets its singular distance
            floored at ``r_eff`` (requires ``cell_volumes``). ``None`` entries or
            ``self_cells=None`` skip the floor (use only when no electrode sits at
            a cell centre, e.g. node-based evaluation).
        cell_volumes: ``(n_cells,)`` cell volumes (m³), required when
            ``self_cells`` floors any electrode's own cell; used for
            ``r_eff = (3 V/4π)^(1/3)``.

    Returns:
        ``(n_cells,)`` detached float64 ``u_p``.
    """
    if cell_centers.ndim != 2:
        raise GeoBrainError(
            "primary_potential: cell_centers must be (n_cells, n_dim)",
            object_name="primary_potential",
            field="cell_centers.ndim",
            expected=2,
            actual=cell_centers.ndim,
        )
    if float(sigma_ref) <= 0.0:
        raise GeoBrainError(
            "primary_potential: sigma_ref must be positive",
            object_name="primary_potential",
            field="sigma_ref",
            expected="> 0",
            actual=sigma_ref,
        )
    if float(current) <= 0.0:
        raise GeoBrainError(
            "primary_potential: current must be positive",
            object_name="primary_potential",
            field="current",
            expected="> 0",
            actual=current,
        )
    n_dim = int(cell_centers.shape[1])
    if not (0 <= int(free_surface_axis) < n_dim):
        raise GeoBrainError(
            "primary_potential: free_surface_axis out of range",
            object_name="primary_potential",
            field="free_surface_axis",
            expected=f"[0, {n_dim})",
            actual=free_surface_axis,
        )

    src_cell: int | None = None
    snk_cell: int | None = None
    r_eff: torch.Tensor | None = None
    if self_cells is not None:
        src_cell, snk_cell = self_cells
        if (src_cell is not None or snk_cell is not None):
            if cell_volumes is None:
                raise GeoBrainError(
                    "primary_potential: cell_volumes required to floor the "
                    "self-cell singular distance (self_cells given)",
                    object_name="primary_potential",
                    field="cell_volumes",
                    expected="(n_cells,) tensor",
                    actual=None,
                )
            r_eff = effective_cell_radius(cell_volumes)

    u_src = _monopole_field(
        cell_centers, source_pos, current, sigma_ref,
        free_surface_axis=int(free_surface_axis),
        free_surface_coord=float(free_surface_coord),
        self_cell=src_cell, r_eff=r_eff,
    )
    u_snk = _monopole_field(
        cell_centers, sink_pos, current, sigma_ref,
        free_surface_axis=int(free_surface_axis),
        free_surface_coord=float(free_surface_coord),
        self_cell=snk_cell, r_eff=r_eff,
    )
    return (u_src - u_snk).detach()
