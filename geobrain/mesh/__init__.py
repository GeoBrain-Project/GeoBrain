"""
Spatial-discretisation primitives.

A ``Mesh`` is a description of a discrete spatial layout. Operators read the mesh
from the typed :class:`~geobrain.core.context.ForwardContext` mesh slot
(``ctx.require_mesh()`` for required reads, ``ctx.mesh`` for optional reads) to
learn cell size, shape, and positions of cell centres.

Four concrete implementations:

- :class:`TensorMesh`: uniform Cartesian; shape ``(nz, nx)`` or ``(nz, nx, ny)``.
- :class:`OctreeMesh`: flat-leaf-list quadtree (2-D) / octree (3-D) with adaptive
  refinement via :meth:`OctreeMesh.from_tensor_mesh`. Cells carry a centre +
  per-axis half-width and an integer refinement level. Fields on an octree mesh
  are indexed by leaf-cell ID (``shape = (n_cells,)``).
- :class:`UnstructuredMesh`: first-class unstructured mesh built from
  explicit cells via :meth:`UnstructuredMesh.from_simplices` (triangles /
  tetrahedra), :meth:`UnstructuredMesh.from_polygons` (2-D polygons) or
  :meth:`UnstructuredMesh.from_mesh` (demotion of any ConnectivityMesh).
  Carries face records for the FV seam and an edge SoA for the 3-D
  edge-element family; the static-EM (DC/IP/SIP) and inductive-EM
  (FDEM3D/MT3D/TEM3D Nédélec) production paths run on it.
- :class:`CylindricalMesh`: axisymmetric ``(r, z)`` discretisation
  (``CylindricalGeometryMesh`` capability); the borehole/EM47-style
  cylindrical family runs on it.

:meth:`Mesh.projection_to` builds a differentiable :class:`MeshProjection` that
maps a field on ``self`` onto a field on the target mesh:

- Same uniform TensorMesh → identity.
- TensorMesh ↔ TensorMesh at different resolutions → bilinear (2-D) / trilinear
  (3-D) interpolation via ``F.interpolate``.
- TensorMesh ↔ OctreeMesh → ``grid_sample`` (TM→OM) or precomputed index-map
  gather (OM→TM).
- Any other GeometryMesh pair: OctreeMesh→OctreeMesh, non-uniform TensorMesh,
  UnstructuredMesh: routes to a general ``cell_centers`` kernel (inverse-distance
  weighting by default, ``k=4``; or ``'nearest'``). All paths are fully
  differentiable.

Mesh geometry is a NON-DIFFERENTIABLE constant, by policy: every mesh
canonicalizes its geometry to detached CPU float64 at construction (a
``requires_grad`` ``cell_widths`` is detached with a warning), and no gradient
ever flows through coordinates, widths, volumes or records. Differentiability
is reserved for FIELDS on the mesh. Geometry inversion (e.g. topography) is
deliberately out of scope.

Module layout:

- ``base.py``: :class:`Mesh` ABC.
- ``capabilities.py``: capability marker classes and topology/geometry records
  (the platform-wide vocabulary; the heaviest externally-consumed surface).
- ``tensor.py`` / ``octree.py`` / ``unstructured.py`` / ``cylindrical.py``,
  the four concrete meshes (``_unstructured_builders.py`` holds the private
  face/edge derivation machinery behind ``unstructured``).
- ``axes.py``: mesh-axis ↔ xyz convention helpers (see the axis-convention
  page of the documentation).
- ``fields.py``: cell-field canonicalisation / mesh-shape agreement checks.
- ``operators.py``: shared FV/staggered discrete operators built on the
  capability records (divergence, gradient, curl, transmissibility).
- ``geometry.py`` / ``geometry_triangle.py``: PLC2D input geometry and its
  Triangle-backend quality mesher (optional dependency).
- ``serialization.py``: ``mesh_from_dict`` / ``MESH_SCHEMA_VERSION``.
- ``projection/``: subpackage: :class:`MeshProjection` operator door with
  private ``_builders`` / ``_kernels`` / ``_points`` machinery.

(The mesh-capability gate ``require_capable_mesh`` lives in
``geobrain.core._require``: contract-side core code since M3, and is
re-exported here for convenience.)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .base import Mesh
from ..core._require import ensure_capable_mesh, require_capable_mesh
from .axes import apply_axes_spec, mesh_axes_to_xyz, xyz_to_mesh_axes
from .fields import canonicalize_cell_field, require_field_matches_mesh
from .capabilities import (
    BoundaryRecords,
    ConnectivityMesh,
    CylindricalGeometryMesh,
    EdgeConnectivityMesh,
    EdgeRecords,
    FaceRecords,
    GeometryMesh,
    PrismGeometryMesh,
    StructuredMesh,
    UniformMesh,
)
from .cylindrical import CylindricalMesh
from .geometry import PLC2D
from .geometry_triangle import mesh_from_triangle_output, triangulate
from .octree import OctreeMesh
from .operators import (
    EdgeCurl,
    average_cell_to_face,
    average_face_to_cell,
    boundary_divergence,
    cell_gradient,
    edge_curl,
    face_divergence,
    face_transmissibility,
    harmonic_face_values,
)
from .projection import MeshProjection
from .tensor import TensorMesh
from .unstructured import UnstructuredMesh
from .serialization import MESH_SCHEMA_VERSION, mesh_from_dict

__all__ = [
    "Mesh",
    "CylindricalMesh",
    "OctreeMesh",
    "MeshProjection",
    "TensorMesh",
    "UnstructuredMesh",
    "mesh_from_dict",
    "MESH_SCHEMA_VERSION",
    "canonicalize_cell_field",
    "apply_axes_spec",
    "mesh_axes_to_xyz",
    "xyz_to_mesh_axes",
    "ensure_capable_mesh",
    "require_capable_mesh",
    "require_field_matches_mesh",
    "BoundaryRecords",
    "ConnectivityMesh",
    "CylindricalGeometryMesh",
    "EdgeConnectivityMesh",
    "EdgeRecords",
    "FaceRecords",
    "GeometryMesh",
    "PrismGeometryMesh",
    "StructuredMesh",
    "UniformMesh",
    # operator factory (shared FV/staggered operators on records)
    "EdgeCurl",
    "average_cell_to_face",
    "average_face_to_cell",
    "boundary_divergence",
    "cell_gradient",
    "edge_curl",
    "face_divergence",
    "face_transmissibility",
    "harmonic_face_values",
    # PLC generation + markers
    "PLC2D",
    "mesh_from_triangle_output",
    "triangulate",
]
