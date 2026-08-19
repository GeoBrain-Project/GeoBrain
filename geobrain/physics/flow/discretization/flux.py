"""Canonical Flow face orientation, gravity, and divergence primitives.

For an internal face ``(left, right)``, positive flux is left-to-right and
therefore contributes ``+q`` to the left-cell divergence and ``-q`` to the
right-cell divergence. Boundary flux is positive from the cell to the
exterior. Depth is positive downward and phase potential is
``pressure - density * g * depth``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from ..errors import FlowContractError

STANDARD_GRAVITY_M_S2 = 9.80665


def _floating_vector(value: object, *, object_name: str, field: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or not value.is_floating_point():
        raise FlowContractError(
            f"{field} must be a floating tensor vector",
            object_name=object_name,
            field=field,
            expected="floating tensor [item]",
            actual=(
                type(value).__name__
                if not isinstance(value, torch.Tensor)
                else (str(value.dtype), tuple(value.shape))
            ),
        )
    if not bool(torch.isfinite(value).all()):
        raise FlowContractError(
            f"{field} must be finite",
            object_name=object_name,
            field=field,
            expected="finite values",
            actual="contains NaN or infinity",
        )
    return value


def _cell_count(value: object, *, object_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FlowContractError(
            "n_cells must be a positive integer",
            object_name=object_name,
            field="n_cells",
            expected="integer >= 1",
            actual=value,
        )
    return value


def _internal_face_cells(
    value: object,
    *,
    object_name: str,
    n_faces: int,
    device: torch.device,
    n_cells: int | None = None,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.int64
        or value.shape != (n_faces, 2)
    ):
        raise FlowContractError(
            "face_cells must align with face_flux",
            object_name=object_name,
            field="face_flux/face_cells",
            expected=f"int64 [{n_faces}, 2]",
            actual=(
                type(value).__name__
                if not isinstance(value, torch.Tensor)
                else (str(value.dtype), tuple(value.shape))
            ),
        )
    if value.device != device:
        raise FlowContractError(
            "face tensors must share one device",
            object_name=object_name,
            field="device",
            expected=str(device),
            actual=str(value.device),
        )
    if n_cells is not None and (bool((value < 0).any()) or bool((value >= n_cells).any())):
        raise FlowContractError(
            "face cell index is outside the grid",
            object_name=object_name,
            field="face_cells",
            expected=f"indices in [0, {n_cells})",
            actual=value.detach().cpu().tolist(),
        )
    return value


def phase_potential(
    pressure_pa: torch.Tensor,
    density_kg_m3: torch.Tensor,
    depth_m: torch.Tensor,
) -> torch.Tensor:
    """Return ``p - rho*g*z`` for SI tensors with depth positive downward."""

    object_name = "phase_potential"
    pressure = _floating_vector(pressure_pa, object_name=object_name, field="pressure_pa")
    density = _floating_vector(
        density_kg_m3,
        object_name=object_name,
        field="density_kg_m3",
    )
    depth = _floating_vector(depth_m, object_name=object_name, field="depth_m")
    if density.shape != pressure.shape or depth.shape != pressure.shape:
        raise FlowContractError(
            "phase-potential tensors must have identical shapes",
            object_name=object_name,
            field="shape",
            expected=tuple(pressure.shape),
            actual=(tuple(density.shape), tuple(depth.shape)),
        )
    if density.dtype != pressure.dtype or depth.dtype != pressure.dtype:
        raise FlowContractError(
            "phase-potential tensors must share one dtype",
            object_name=object_name,
            field="dtype",
            expected=str(pressure.dtype),
            actual=(str(density.dtype), str(depth.dtype)),
        )
    if density.device != pressure.device or depth.device != pressure.device:
        raise FlowContractError(
            "phase-potential tensors must share one device",
            object_name=object_name,
            field="device",
            expected=str(pressure.device),
            actual=(str(density.device), str(depth.device)),
        )
    if bool((density < 0).any()):
        raise FlowContractError(
            "density must be non-negative",
            object_name=object_name,
            field="density_kg_m3",
            expected=">= 0",
            actual=float(density.min()),
        )
    return pressure - density * pressure.new_tensor(STANDARD_GRAVITY_M_S2) * depth


def upwind_cell(face_flux: torch.Tensor, face_cells: torch.Tensor) -> torch.Tensor:
    """Return the left cell for non-negative flux and right cell otherwise."""

    object_name = "upwind_cell"
    flux = _floating_vector(face_flux, object_name=object_name, field="face_flux")
    cells = _internal_face_cells(
        face_cells,
        object_name=object_name,
        n_faces=flux.numel(),
        device=flux.device,
    )
    return torch.where(flux >= 0, cells[:, 0], cells[:, 1])


def scatter_internal_face_flux(
    face_flux: torch.Tensor,
    face_cells: torch.Tensor,
    n_cells: int,
) -> torch.Tensor:
    """Scatter left-to-right face flux into canonical cell divergence."""

    object_name = "scatter_internal_face_flux"
    flux = _floating_vector(face_flux, object_name=object_name, field="face_flux")
    count = _cell_count(n_cells, object_name=object_name)
    cells = _internal_face_cells(
        face_cells,
        object_name=object_name,
        n_faces=flux.numel(),
        device=flux.device,
        n_cells=count,
    )
    divergence = flux.new_zeros(count)
    divergence = divergence.scatter_add(0, cells[:, 0], flux)
    return divergence.scatter_add(0, cells[:, 1], -flux)


def scatter_boundary_outflow(
    boundary_flux: torch.Tensor,
    boundary_cells: torch.Tensor,
    n_cells: int,
) -> torch.Tensor:
    """Scatter cell-to-exterior boundary flux into cell divergence."""

    object_name = "scatter_boundary_outflow"
    flux = _floating_vector(
        boundary_flux,
        object_name=object_name,
        field="boundary_flux",
    )
    count = _cell_count(n_cells, object_name=object_name)
    if (
        not isinstance(boundary_cells, torch.Tensor)
        or boundary_cells.dtype != torch.int64
        or boundary_cells.shape != flux.shape
    ):
        raise FlowContractError(
            "boundary_cells must align with boundary_flux",
            object_name=object_name,
            field="boundary_flux/boundary_cells",
            expected=f"int64 {tuple(flux.shape)}",
            actual=(
                type(boundary_cells).__name__
                if not isinstance(boundary_cells, torch.Tensor)
                else (str(boundary_cells.dtype), tuple(boundary_cells.shape))
            ),
        )
    if boundary_cells.device != flux.device:
        raise FlowContractError(
            "boundary tensors must share one device",
            object_name=object_name,
            field="device",
            expected=str(flux.device),
            actual=str(boundary_cells.device),
        )
    if bool((boundary_cells < 0).any()) or bool((boundary_cells >= count).any()):
        raise FlowContractError(
            "boundary cell index is outside the grid",
            object_name=object_name,
            field="boundary_cells",
            expected=f"indices in [0, {count})",
            actual=boundary_cells.detach().cpu().tolist(),
        )
    return flux.new_zeros(count).scatter_add(0, boundary_cells, flux)


__all__ = [
    "STANDARD_GRAVITY_M_S2",
    "phase_potential",
    "scatter_boundary_outflow",
    "scatter_internal_face_flux",
    "upwind_cell",
]
