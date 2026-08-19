"""SI geometric transmissibility and phase flux for TPFA grids.

The two-point scheme is scientifically consistent on K-orthogonal grids. Face
orientation follows :mod:`geobrain.physics.flow.discretization.flux`: positive
phase flux travels from the left cell to the right cell.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from ..errors import FlowContractError
from .flux import _floating_vector, _internal_face_cells, upwind_cell


def _floating_matrix(value: object, *, field: str) -> torch.Tensor:
    object_name = "tpfa_face_transmissibility"
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[1] not in (2, 3)
        or not value.is_floating_point()
    ):
        raise FlowContractError(
            f"{field} must use 2-D or 3-D coordinate columns",
            object_name=object_name,
            field=field,
            expected="floating [item, 2|3]",
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
            expected="finite coordinates in metres",
            actual="contains NaN or infinity",
        )
    return value


def _scaled_norm(vectors: torch.Tensor) -> torch.Tensor:
    scale = vectors.abs().amax(dim=1)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return safe_scale * torch.linalg.vector_norm(vectors / safe_scale[:, None], dim=1)


def _series_transmissibility(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Combine non-negative half-face transmissibilities without products."""

    smaller = torch.minimum(left, right)
    larger = torch.maximum(left, right)
    safe_larger = torch.where(larger > 0, larger, torch.ones_like(larger))
    ratio = smaller / safe_larger
    return torch.where(larger > 0, smaller / (1.0 + ratio), torch.zeros_like(smaller))


def tpfa_face_transmissibility(
    cell_centres_m: torch.Tensor,
    face_centres_m: torch.Tensor,
    face_areas_m2: torch.Tensor,
    permeability_m2: torch.Tensor,
    face_cells: torch.Tensor,
) -> torch.Tensor:
    """Return SI face transmissibility ``A*k/d`` combined in series.

    ``permeability_m2`` is the positive cell permeability projected onto the
    face normal. The result has units of cubic metres and remains
    differentiable with respect to geometry and permeability.
    """

    object_name = "tpfa_face_transmissibility"
    centres = _floating_matrix(cell_centres_m, field="cell_centres_m")
    face_centres = _floating_matrix(face_centres_m, field="face_centres_m")
    areas = _floating_vector(
        face_areas_m2,
        object_name=object_name,
        field="face_areas_m2",
    )
    permeability = _floating_vector(
        permeability_m2,
        object_name=object_name,
        field="permeability_m2",
    )
    if face_centres.shape != (areas.numel(), centres.shape[1]):
        raise FlowContractError(
            "face geometry arrays do not align",
            object_name=object_name,
            field="face_centres_m/face_areas_m2",
            expected=(areas.numel(), centres.shape[1]),
            actual=tuple(face_centres.shape),
        )
    if permeability.shape != (centres.shape[0],):
        raise FlowContractError(
            "permeability must provide one projected value per cell",
            object_name=object_name,
            field="permeability_m2",
            expected=(centres.shape[0],),
            actual=tuple(permeability.shape),
        )
    tensors = (face_centres, areas, permeability)
    if any(item.dtype != centres.dtype for item in tensors):
        raise FlowContractError(
            "TPFA geometry and permeability must share one dtype",
            object_name=object_name,
            field="dtype",
            expected=str(centres.dtype),
            actual=tuple(str(item.dtype) for item in tensors),
        )
    if any(item.device != centres.device for item in tensors):
        raise FlowContractError(
            "TPFA geometry and permeability must share one device",
            object_name=object_name,
            field="device",
            expected=str(centres.device),
            actual=tuple(str(item.device) for item in tensors),
        )
    if bool((areas <= 0).any()) or bool((permeability <= 0).any()):
        raise FlowContractError(
            "TPFA face areas and projected permeabilities must be positive",
            object_name=object_name,
            field="face_areas_m2/permeability_m2",
            expected="> 0",
            actual=(float(areas.min()), float(permeability.min())),
        )
    cells = _internal_face_cells(
        face_cells,
        object_name=object_name,
        n_faces=areas.numel(),
        device=centres.device,
        n_cells=centres.shape[0],
    )
    left_distance = _scaled_norm(face_centres - centres[cells[:, 0]])
    right_distance = _scaled_norm(face_centres - centres[cells[:, 1]])
    if bool((left_distance <= 0).any()) or bool((right_distance <= 0).any()):
        raise FlowContractError(
            "cell centres must not coincide with their face centre",
            object_name=object_name,
            field="cell_centres_m/face_centres_m",
            expected="strictly positive half-face distances",
            actual=(float(left_distance.min()), float(right_distance.min())),
        )
    left = areas * permeability[cells[:, 0]] / left_distance
    right = areas * permeability[cells[:, 1]] / right_distance
    transmissibility = _series_transmissibility(left, right)
    if not bool(torch.isfinite(transmissibility).all()) or bool((transmissibility <= 0).any()):
        raise FlowContractError(
            "TPFA transmissibility is not positive and finite",
            object_name=object_name,
            field="transmissibility",
            expected="positive finite m^3",
            actual=transmissibility.detach().cpu().tolist(),
        )
    return transmissibility


def tpfa_phase_flux(
    transmissibility_m3: torch.Tensor,
    phase_potential_pa: torch.Tensor,
    mobility_pa_s_inv: torch.Tensor,
    face_cells: torch.Tensor,
) -> torch.Tensor:
    """Return left-to-right phase volumetric flux in cubic metres per second."""

    object_name = "tpfa_phase_flux"
    transmissibility = _floating_vector(
        transmissibility_m3,
        object_name=object_name,
        field="transmissibility_m3",
    )
    potential = _floating_vector(
        phase_potential_pa,
        object_name=object_name,
        field="phase_potential_pa",
    )
    mobility = _floating_vector(
        mobility_pa_s_inv,
        object_name=object_name,
        field="mobility_pa_s_inv",
    )
    if mobility.shape != potential.shape:
        raise FlowContractError(
            "phase mobility must provide one value per cell potential",
            object_name=object_name,
            field="phase_potential_pa/mobility_pa_s_inv",
            expected=tuple(potential.shape),
            actual=tuple(mobility.shape),
        )
    if mobility.dtype != potential.dtype or transmissibility.dtype != potential.dtype:
        raise FlowContractError(
            "TPFA flux tensors must share one dtype",
            object_name=object_name,
            field="dtype",
            expected=str(potential.dtype),
            actual=(str(transmissibility.dtype), str(mobility.dtype)),
        )
    if mobility.device != potential.device or transmissibility.device != potential.device:
        raise FlowContractError(
            "TPFA flux tensors must share one device",
            object_name=object_name,
            field="device",
            expected=str(potential.device),
            actual=(str(transmissibility.device), str(mobility.device)),
        )
    if bool((transmissibility <= 0).any()) or bool((mobility < 0).any()):
        raise FlowContractError(
            "transmissibility must be positive and mobility non-negative",
            object_name=object_name,
            field="transmissibility_m3/mobility_pa_s_inv",
            expected="transmissibility > 0 and mobility >= 0",
            actual=(float(transmissibility.min()), float(mobility.min())),
        )
    cells = _internal_face_cells(
        face_cells,
        object_name=object_name,
        n_faces=transmissibility.numel(),
        device=potential.device,
        n_cells=potential.numel(),
    )
    driving_flux = transmissibility * (potential[cells[:, 0]] - potential[cells[:, 1]])
    upstream = upwind_cell(driving_flux, cells)
    flux = driving_flux * mobility[upstream]
    if not bool(torch.isfinite(flux).all()):
        raise FlowContractError(
            "TPFA phase flux is not finite",
            object_name=object_name,
            field="flux",
            expected="finite m^3/s",
            actual=flux.detach().cpu().tolist(),
        )
    return flux


__all__ = ["tpfa_face_transmissibility", "tpfa_phase_flux"]
