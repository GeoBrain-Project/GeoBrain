"""The "this tensor is a cell field on this mesh" contract, in one place.

Every mesh-consuming operator must validate that an input tensor actually
lays out one value per cell of the ctx mesh. The contract has exactly two
dialects, each with one canonical validator (single-sourced here so the
physics families cannot drift apart):

- dual-path operators (DC3D, DC2.5D, IP, SIP, MT2D, VTEM, ...): grid-shaped
  ``mesh.shape`` field on a StructuredMesh, flat ``(n_cells,)`` on any other
  ConnectivityMesh: :func:`canonicalize_cell_field`;
- grid-only operators (wave, Helmholtz, gravity/magnetics, ...): the field
  must match ``mesh.shape`` exactly and KEEPS its grid layout,
  :func:`require_field_matches_mesh`.

Both raise the platform's structured :class:`~geobrain.core.errors.FieldShapeError` with the field
name and the offending shape, and both are pure views (no copy, autograd
flows through untouched).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from ..core.errors import FieldShapeError
from .capabilities import StructuredMesh

__all__ = ["canonicalize_cell_field", "require_field_matches_mesh"]


def canonicalize_cell_field(
    mesh,
    field: torch.Tensor,
    *,
    name: str = "field",
    owner: str | None = None,
) -> torch.Tensor:
    """Validate a per-cell field against the mesh; return the flat view.

    The dual-path contract: a mesh declaring
    :class:`~geobrain.mesh.capabilities.StructuredMesh` takes the field
    in grid layout (``field.shape == mesh.shape``) or already flat
    (``(n_cells,)``, platform C-order flat-index convention); any other mesh
    takes it flat only. Returns the flat view (``field.reshape(-1)``, a
    view, so autograd and device placement pass through untouched).

    Args:
        mesh: The (ctx) mesh the field must live on.
        field: The candidate per-cell tensor.
        name: Field name woven into the error (``"sigma"``, ``"vp"``, ...).
        owner: Optional operator name for the error's ``object_name``.

    Raises:
        FieldShapeError: on any shape mismatch.
    """
    if mesh.declares(StructuredMesh):
        shape = tuple(mesh.shape)
        if tuple(field.shape) != shape and tuple(field.shape) != (int(mesh.n_cells),):
            raise FieldShapeError(
                f"{name} shape must be mesh.shape or flat (n_cells,)",
                object_name=owner or "canonicalize_cell_field",
                field=name,
                expected=(shape, (int(mesh.n_cells),)),
                actual=tuple(field.shape),
            )
        return field if field.dim() == 1 else field.reshape(-1)
    # Non-structured meshes have no grid layout, so any numel-matching
    # layout is unambiguous: flatten it (Gravity3D/Magnetic3D historically
    # accepted e.g. an (n, 1) leaf vector; pure C-order flatten).
    if int(field.numel()) != int(mesh.n_cells):
        raise FieldShapeError(
            f"{name} must be flat (n_cells,) on a non-structured mesh",
            object_name=owner or "canonicalize_cell_field",
            field=name,
            expected=(int(mesh.n_cells),),
            actual=tuple(field.shape),
        )
    return field if field.dim() == 1 else field.reshape(-1)


def require_field_matches_mesh(
    mesh,
    field: torch.Tensor,
    *,
    name: str = "field",
    owner: str | None = None,
) -> torch.Tensor:
    """Validate a grid-layout field against ``mesh.shape``; return it unchanged.

    The grid-only contract for operators that keep the ``(nz, nx[, ny])``
    layout throughout (FDTD wave, Helmholtz, gravity/magnetics, structured
    EM): the field must match ``mesh.shape`` exactly; flat inputs are
    rejected too, matching the operators' historical behaviour.

    Args:
        mesh: A mesh with a grid ``shape`` (StructuredMesh-declaring).
        field: The candidate grid tensor.
        name: Field name woven into the error.
        owner: Optional operator name for the error's ``object_name``.

    Raises:
        FieldShapeError: on shape mismatch.
    """
    if tuple(field.shape) != tuple(mesh.shape):
        raise FieldShapeError(
            f"{name} shape mismatches mesh.shape",
            object_name=owner or "require_field_matches_mesh",
            field=name,
            expected=tuple(mesh.shape),
            actual=tuple(field.shape),
        )
    return field
