"""Discretization dispatch for the inductive EM 3-D curl-curl family.

Lives at the ``numerics`` package level (moved from ``edge_element``, review
item: the edge-element package is only ONE branch of this dispatch; the
finite-volume 'structured' branch is the other). ``edge_element.__init__``
keeps a compatibility re-export.

The one decision FDEM3D/MT3D/TEM3D each make at the top of ``_forward``: WHICH
discretization a given mesh selects. ADR 0002 pins the dual contract, a mesh
declaring :class:`~geobrain.mesh.capabilities.StructuredMesh` (TensorMesh)
takes the original Yee finite-volume path; a 3-D simplex-built mesh declaring
:class:`~geobrain.mesh.capabilities.EdgeConnectivityMesh` takes the
Nédélec edge-element (Whitney) path; anything else (e.g. octree) raises, since
inductive EM on it is not implemented. The three operators had byte-identical
copies of this ladder; this is the single source of truth.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Literal

from geobrain.core.errors import GeoBrainError
from geobrain.mesh.capabilities import EdgeConnectivityMesh, StructuredMesh

__all__ = ["resolve_inductive_mesh_path"]


def resolve_inductive_mesh_path(
    mesh,
    *,
    operator_name: str,
) -> Literal["structured", "edge"]:
    """Resolve which 3-D inductive-EM discretization a mesh selects.

    Capability dispatch is declaration-based: the INSTANCE attribute is read
    first; ``EdgeConnectivityMesh`` is declared per instance by a 3-D
    simplex-built :class:`~geobrain.mesh.unstructured.UnstructuredMesh`,
    so a class-level ``type(mesh).mesh_capabilities`` read would miss it (the
    ``require_mesh`` rule). ``StructuredMesh`` wins when both are present.

    Args:
        mesh: the forward mesh. ``StructuredMesh`` (TensorMesh) → ``"structured"``
            (Yee finite volume); ``EdgeConnectivityMesh`` (3-D simplex-built)
            without ``StructuredMesh`` → ``"edge"`` (Nédélec edge elements).
        operator_name: the calling operator's name (``"FDEM3D"``, ``"MT3D"``,
            ``"TEM3D"``), woven into the raised message and error object.

    Returns:
        ``"structured"`` for the Yee path, ``"edge"`` for the edge-element path.

    Raises:
        GeoBrainError: when the mesh declares neither path (e.g. octree);
            octree inductive EM is not implemented.
    """
    if mesh.declares(StructuredMesh):
        return "structured"
    if mesh.declares(EdgeConnectivityMesh):
        return "edge"
    raise GeoBrainError(
        f"{operator_name} supports TensorMesh (structured Yee grid) and 3-D "
        "simplex-built UnstructuredMesh (Nédélec edge elements); "
        "octree inductive EM is not implemented",
        object_name=operator_name, field="mesh",
        expected="StructuredMesh or EdgeConnectivityMesh",
        actual=sorted(
            c.__name__
            for c in getattr(mesh, "mesh_capabilities", frozenset())
        ),
    )
