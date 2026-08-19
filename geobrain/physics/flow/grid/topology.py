"""Canonical oriented face topology for Flow finite-volume grids.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..errors import FlowContractError


@dataclass(frozen=True, slots=True)
class OrientedFaceTopology:
    """Interior-face orientation and outward Cartesian boundary records."""

    face_cells: torch.Tensor
    boundary_cells: torch.Tensor
    boundary_normals: torch.Tensor
    incidence: torch.Tensor

    def clone(self) -> "OrientedFaceTopology":
        """Return a storage-independent public snapshot."""

        return OrientedFaceTopology(
            face_cells=self.face_cells.clone(),
            boundary_cells=self.boundary_cells.clone(),
            boundary_normals=self.boundary_normals.clone(),
            incidence=self.incidence.clone(),
        )

    def __post_init__(self) -> None:
        if self.face_cells.dtype != torch.int64 or self.face_cells.ndim != 2:
            raise FlowContractError(
                "face_cells must be an int64 [face, 2] tensor",
                object_name="OrientedFaceTopology",
                field="face_cells",
                expected="int64 [face, 2]",
                actual=(str(self.face_cells.dtype), tuple(self.face_cells.shape)),
            )
        if self.face_cells.shape[1:] != (2,):
            raise FlowContractError(
                "face_cells must have two oriented cell columns",
                object_name="OrientedFaceTopology",
                field="face_cells",
                expected="[face, 2]",
                actual=tuple(self.face_cells.shape),
            )
        if self.boundary_cells.dtype != torch.int64 or self.boundary_cells.ndim != 1:
            raise FlowContractError(
                "boundary_cells must be an int64 vector",
                object_name="OrientedFaceTopology",
                field="boundary_cells",
                expected="int64 [boundary_face]",
                actual=(str(self.boundary_cells.dtype), tuple(self.boundary_cells.shape)),
            )
        if self.boundary_normals.shape != (self.boundary_cells.numel(), 3):
            raise FlowContractError(
                "boundary_normals must use xyz columns for every boundary face",
                object_name="OrientedFaceTopology",
                field="boundary_normals",
                expected=(self.boundary_cells.numel(), 3),
                actual=tuple(self.boundary_normals.shape),
            )
        if self.incidence.layout != torch.sparse_coo:
            raise FlowContractError(
                "incidence must be a sparse COO tensor",
                object_name="OrientedFaceTopology",
                field="incidence",
                expected="torch.sparse_coo",
                actual=str(self.incidence.layout),
            )
        devices = {
            self.face_cells.device,
            self.boundary_cells.device,
            self.boundary_normals.device,
            self.incidence.device,
        }
        if len(devices) != 1:
            raise FlowContractError(
                "topology tensors must share one device",
                object_name="OrientedFaceTopology",
                field="device",
                expected="one device",
                actual=tuple(sorted(map(str, devices))),
            )


def cartesian_oriented_topology(
    *,
    nx: int,
    ny: int,
    nz: int,
    face_cells: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> OrientedFaceTopology:
    """Build canonical sparse orientation records for a Cartesian grid."""

    owned_face_cells = face_cells.clone()
    n_cells = nx * ny * nz
    n_faces = int(owned_face_cells.shape[0])
    face_ids = torch.arange(n_faces, dtype=torch.int64, device=device)
    if n_faces:
        rows = torch.cat((owned_face_cells[:, 0], owned_face_cells[:, 1]))
        cols = torch.cat((face_ids, face_ids))
        values = torch.cat(
            (
                torch.ones(n_faces, dtype=dtype, device=device),
                -torch.ones(n_faces, dtype=dtype, device=device),
            )
        )
        indices = torch.stack((rows, cols))
    else:
        indices = torch.empty((2, 0), dtype=torch.int64, device=device)
        values = torch.empty(0, dtype=dtype, device=device)
    incidence = torch.sparse_coo_tensor(
        indices,
        values,
        size=(n_cells, n_faces),
        dtype=dtype,
        device=device,
    ).coalesce()

    cell_ids = torch.arange(n_cells, dtype=torch.int64, device=device).reshape(nz, ny, nx)
    boundary_parts = (
        cell_ids[:, :, 0].reshape(-1),
        cell_ids[:, :, -1].reshape(-1),
        cell_ids[:, 0, :].reshape(-1),
        cell_ids[:, -1, :].reshape(-1),
        cell_ids[0, :, :].reshape(-1),
        cell_ids[-1, :, :].reshape(-1),
    )
    normal_values = (
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
    )
    boundary_cells = torch.cat(boundary_parts)
    boundary_normals = torch.cat(
        [
            torch.tensor(normal, dtype=dtype, device=device).expand(cells.numel(), 3)
            for cells, normal in zip(boundary_parts, normal_values, strict=True)
        ],
        dim=0,
    )
    return OrientedFaceTopology(
        face_cells=owned_face_cells,
        boundary_cells=boundary_cells,
        boundary_normals=boundary_normals,
        incidence=incidence,
    )


__all__ = ["OrientedFaceTopology", "cartesian_oriented_topology"]
