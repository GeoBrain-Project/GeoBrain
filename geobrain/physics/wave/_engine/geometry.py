"""Geometry compilation for the internal Wave propagation engine.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.core import survey as core_survey
from geobrain.mesh.capabilities import UniformMesh
from geobrain.mesh.tensor import TensorMesh

from ..acquisition import Seismic2DSurvey, Seismic3DSurvey
from ..errors import WaveContractError
from .contracts import CompiledAcquisition


def _geometry_error(field: str, expected: object, actual: object) -> WaveContractError:
    """Build a structured acquisition compilation error."""
    return WaveContractError(
        "invalid Wave acquisition geometry",
        object_name="compile_acquisition",
        field=field,
        expected=expected,
        actual=actual,
    )


def _validate_bounds(
    positions: torch.Tensor,
    *,
    field: str,
    public_to_platform: tuple[int, ...],
    mesh: TensorMesh,
) -> None:
    """Reject every invalid world coordinate before shared mapping is invoked."""
    for platform_axis, public_axis in enumerate(public_to_platform):
        lower = mesh.origin[platform_axis]
        upper = lower + mesh.shape[platform_axis] * mesh.spacing[platform_axis]
        coordinate = positions[:, public_axis]
        if bool(((coordinate < lower) | (coordinate >= upper)).any()):
            raise _geometry_error(
                field,
                f"world coordinates within [{lower}, {upper}) on platform axis {platform_axis}",
                coordinate.tolist(),
            )


def _map_positions(
    positions: torch.Tensor,
    *,
    public_to_platform: tuple[int, ...],
    mesh: TensorMesh,
    device: torch.device,
) -> torch.Tensor:
    """Apply the shared mapper per platform axis after bounds validation."""
    mapped = [
        core_survey.coords_to_cell_indices(
            positions[:, public_axis],
            mesh.spacing[platform_axis],
            mesh.shape[platform_axis],
            origin=mesh.origin[platform_axis],
        )
        for platform_axis, public_axis in enumerate(public_to_platform)
    ]
    return torch.stack(mapped, dim=1).to(device=device, dtype=torch.int64)


def compile_acquisition(
    survey: Seismic2DSurvey | Seismic3DSurvey,
    mesh: TensorMesh,
    *,
    device: torch.device | str,
) -> CompiledAcquisition:
    """Compile canonical packed world coordinates into platform indices."""
    if UniformMesh not in getattr(mesh, "mesh_capabilities", frozenset()):
        raise _geometry_error("mesh", "a uniform TensorMesh", type(mesh).__name__)
    public_to_platform: tuple[int, ...]
    if isinstance(survey, Seismic2DSurvey):
        public_to_platform = (1, 0)
    elif isinstance(survey, Seismic3DSurvey):
        public_to_platform = (2, 0, 1)
    else:
        raise _geometry_error(
            "survey", "Seismic2DSurvey or Seismic3DSurvey", type(survey).__name__
        )
    if len(mesh.shape) != len(public_to_platform):
        raise _geometry_error(
            "mesh",
            f"{len(public_to_platform)}-D TensorMesh",
            f"{len(mesh.shape)}-D TensorMesh",
        )
    try:
        selected_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise _geometry_error("device", "valid torch device", device) from exc

    _validate_bounds(
        survey.source_positions,
        field="source_positions",
        public_to_platform=public_to_platform,
        mesh=mesh,
    )
    _validate_bounds(
        survey.receiver_positions,
        field="receiver_positions",
        public_to_platform=public_to_platform,
        mesh=mesh,
    )
    source_indices = _map_positions(
        survey.source_positions,
        public_to_platform=public_to_platform,
        mesh=mesh,
        device=selected_device,
    )
    receiver_indices = _map_positions(
        survey.receiver_positions,
        public_to_platform=public_to_platform,
        mesh=mesh,
        device=selected_device,
    )
    return CompiledAcquisition(
        source_indices=source_indices,
        receiver_indices=receiver_indices,
        source_shot_index=survey.source_shot_index.detach().clone(),
        receiver_shot_index=survey.receiver_shot_index.detach().clone(),
        n_shot=survey.n_shot,
        nt=survey.nt,
        dt=survey.dt,
        t0=survey.t0,
        survey_fingerprint=survey.fingerprint,
        mesh_shape=tuple(mesh.shape),
        spacing=tuple(mesh.spacing),
    )


__all__ = ["compile_acquisition"]
