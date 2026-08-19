"""Capability-based mesh requirement checks (replace scattered isinstance guards).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .context import ForwardContext
from typing import TYPE_CHECKING

from .errors import GeoBrainError

if TYPE_CHECKING:  # duck-typed gate: no runtime mesh import (layer rule)
    from geobrain.mesh import Mesh


def ensure_capable_mesh(mesh: "Mesh", *caps: type, owner: str | None = None) -> "Mesh":
    """Assert ``mesh`` declares every capability in ``caps``; return the mesh.

    The mesh-level core of :func:`require_capable_mesh` for callers that
    already hold a mesh object (constructor-injected meshes, projection
    internals). Capability membership is BY DECLARATION, instance attribute
    first, class declaration as the fallback (see :meth:`Mesh.declares`,
    the single home for that rule). This is deliberately not a
    ``runtime_checkable`` structural test: capability is decided by what a
    mesh *declares*, never by which attributes happen to be present (per-
    instance refinement is invisible to structural tests, and tombstone
    properties like ``UnstructuredMesh.shape`` would fool them).

    Args:
        mesh: The mesh object to check.
        *caps: Capability marker classes the mesh must declare.
        owner: Optional name of the requiring operator/component, included
            in the error so the user sees *who* needed the capability.

    Returns:
        ``mesh`` unchanged.

    Raises:
        GeoBrainError: if the mesh lacks a required capability.
    """
    have = getattr(mesh, "mesh_capabilities", frozenset())
    for cap in caps:
        if cap not in have:
            raise GeoBrainError(
                (f"{owner} requires a mesh capability it lacks"
                 if owner else
                 "operator requires a mesh capability it lacks"),
                object_name=type(mesh).__name__,
                field="mesh",
                expected=cap.__name__,
                actual=sorted(c.__name__ for c in have),
            )
    return mesh


def require_capable_mesh(ctx: ForwardContext, *caps: type,
                         owner: str | None = None):
    """Fetch ``ctx.require_mesh()`` and assert it declares every capability in ``caps``.

    Thin ctx-fetch wrapper over :func:`ensure_capable_mesh`; see it (and
    :meth:`Mesh.declares`) for the declaration-based membership semantics.

    Args:
        ctx: The :class:`ForwardContext` from which to fetch the mesh.
        *caps: Capability marker classes that the mesh must declare.
        owner: Optional name of the requiring operator, threaded into the
            error message.

    Returns:
        The mesh object from ``ctx.require_mesh()``.

    Raises:
        GeoBrainError: if the mesh is absent or lacks a required capability.
    """
    return ensure_capable_mesh(ctx.require_mesh(), *caps, owner=owner)
