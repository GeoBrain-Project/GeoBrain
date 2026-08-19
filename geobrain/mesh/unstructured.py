# pyright: reportPrivateImportUsage=false
"""First-class unstructured mesh: data-driven, dimension-agnostic (2-D / 3-D).

Unlike a ``TensorMesh``/``OctreeMesh`` this has no per-axis ``shape``/``spacing``;
it is defined purely by its cell-centre geometry and face connectivity (the
struct-of-arrays the finite-volume seam consumes). Constructors:

- ``UnstructuredMesh(cell_centers, cell_volumes, faces[, boundary_faces])``;
  dimension-agnostic SoA-direct (you already have the FV data).
- ``UnstructuredMesh.from_polygons(vertices, cells)``: 2-D builder that derives
  the SoA from a polygon mesh.
- ``UnstructuredMesh.from_simplices(vertices, simplices)``: 3-D builder that
  derives the SoA from a tetrahedral mesh.
- ``UnstructuredMesh.from_points(points)``: convenience builder that runs a
  ``scipy.spatial.Delaunay`` triangulation (2-D triangles / 3-D tetrahedra) and
  delegates to ``from_polygons`` / ``from_simplices``.

Coordinate-column frame (IMPORTANT; the builders do NOT transpose for you):
every vertex/point/centre coordinate is in the platform mesh-axis order,
**column 0 = z (depth, positive down), then x, then y**, matching the
``(nz, nx, ny)`` (3-D) / ``(nz, nx)`` (2-D) :class:`TensorMesh` axis order. So a
caller must pass ``from_points`` / ``from_polygons`` / ``from_simplices`` their
coordinates in ``(z, x, y)`` (3-D) or ``(z, x)`` (2-D) column order. Feeding a
standard ``(x, y, z)`` point cloud silently transposes depth into the x column
and mislabels the geometry, convert to ``(z, x, y)`` first.

Declares ``{GeometryMesh, ConnectivityMesh}``, never ``StructuredMesh``. A 3-D
simplex-built instance (``from_simplices`` / 3-D ``from_points``, or the direct
constructor fed ``node_coords=``/``cell_nodes=``) retains its node coordinates
and cell→node table and additionally declares ``EdgeConnectivityMesh``, per
INSTANCE, so 2-D ``from_polygons`` instances of this same class do not.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from typing import ClassVar, Sequence

import numpy as np
import torch

from ..core.errors import GeoBrainError
from ..core._require import ensure_capable_mesh
from .axes import resolve_axes_kwarg, xyz_to_mesh_axes
from ._unstructured_builders import (
    derive_faces_2d, derive_faces_3d, triangle_circumcenter,
)
from .base import Mesh
from .capabilities import (
    BoundaryRecords, ConnectivityMesh, EdgeConnectivityMesh, EdgeRecords,
    FaceRecords, GeometryMesh,
)

# The six local edges of a tetrahedron over its STORED node order, the pairs
# (0,1), (0,2), (0,3), (1,2), (1,3), (2,3), local direction = first listed
# node → second. This fixed ordering is part of the EdgeConnectivityMesh
# contract (downstream Whitney edge-basis assembly indexes by it).
_TET_LOCAL_EDGES: torch.Tensor = torch.tensor(
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=torch.long,
)


def _polygon_self_intersects(pts: torch.Tensor) -> bool:
    """True iff the closed vertex-loop ``pts`` (``(k, 2)``) is self-intersecting.

    Tests every pair of NON-ADJACENT edges for a *proper* crossing (strict, a
    shared endpoint or a bare touch does not count). A triangle (``k == 3``) has
    no non-adjacent edges, so it can never self-intersect and the scan returns
    immediately; the cheap O(k²) test runs only for ``k >= 4`` (quads and up),
    where a "bowtie" vertex ordering produces a nonzero signed area that the
    zero-area guard alone would miss. ``k`` is tiny for real FV cells, so this is
    negligible next to the per-cell area work in the same loop.
    """
    k = int(pts.shape[0])
    if k < 4:
        return False
    p = [(float(pts[i, 0]), float(pts[i, 1])) for i in range(k)]

    def _orient(o, u, v) -> float:
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])

    def _proper_cross(a, b, c, d) -> bool:
        d1, d2 = _orient(c, d, a), _orient(c, d, b)
        d3, d4 = _orient(a, b, c), _orient(a, b, d)
        return ((d1 > 0.0) != (d2 > 0.0)) and ((d3 > 0.0) != (d4 > 0.0))

    for i in range(k):
        a, b = p[i], p[(i + 1) % k]
        for j in range(i + 1, k):
            # Skip edges sharing a vertex: consecutive (j == i+1) or the wrap
            # pair (edge 0 and edge k-1 meet at vertex 0).
            if j == i + 1 or (i == 0 and j == k - 1):
                continue
            c, d = p[j], p[(j + 1) % k]
            if _proper_cross(a, b, c, d):
                return True
    return False


class UnstructuredMesh(Mesh):
    """A data-driven unstructured mesh (GeometryMesh + ConnectivityMesh).

    Validated under DC3D: building an :class:`UnstructuredMesh` directly from a
    :class:`TensorMesh`'s own SoA and running DC3D gives BIT-IDENTICAL voltages
    (diff 0.0) to the structured run, and a true tetrahedral Delaunay mesh
    (``from_points`` → ``from_simplices``, circumcenter cell points) reproduces
    the analytic half-space potential with the error CONVERGING under
    point-cloud refinement (pinned end-to-end by the DC3D-unstructured and
    simplex test suites).

    A 3-D simplex-built instance keeps its node coordinates + cell→node table
    and declares :class:`EdgeConnectivityMesh` on the INSTANCE (the class-level
    default below is what 2-D / SoA-direct instances fall back to), providing
    ``edge_records()`` / ``cell_edges()`` / ``cell_nodes()`` / ``node_coords()``
    for the physics-side Nédélec/Whitney edge-element assembly.

    Args:
        cell_centers: ``(n_cells, dim)`` cell-centre coordinates.
        cell_volumes: ``(n_cells,)`` positive cell volumes.
        faces: internal-face :class:`FaceRecords` (the FV seam).
        boundary_faces: optional boundary :class:`BoundaryRecords`.
        node_coords: optional ``(n_nodes, dim)`` node coordinates.
        cell_nodes: optional per-cell node index table.
        cell_markers: optional integer region markers per cell.
        boundary_markers: optional integer markers per boundary face.
    """

    # Class-level default declaration. A per-instance refinement in __init__
    # adds EdgeConnectivityMesh when simplex topology is stored; ``require_mesh``
    # reads the instance attribute with this ClassVar as the fallback.
    # Plain class-level default: instances override (capability refinement).
    mesh_capabilities: frozenset[type] = frozenset(
        {GeometryMesh, ConnectivityMesh}
    )

    def __init__(
        self,
        cell_centers: torch.Tensor,
        cell_volumes: torch.Tensor,
        faces: FaceRecords,
        *,
        boundary_faces: BoundaryRecords | None = None,
        node_coords: torch.Tensor | None = None,
        cell_nodes: torch.Tensor | None = None,
        cell_markers: torch.Tensor | None = None,
        boundary_markers: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(cell_centers, torch.Tensor):
            raise GeoBrainError(
                "cell_centers must be a Tensor",
                object_name="UnstructuredMesh", field="cell_centers",
                expected="torch.Tensor", actual=type(cell_centers).__name__,
            )
        if not isinstance(cell_volumes, torch.Tensor):
            raise GeoBrainError(
                "cell_volumes must be a Tensor",
                object_name="UnstructuredMesh", field="cell_volumes",
                expected="torch.Tensor", actual=type(cell_volumes).__name__,
            )
        if cell_centers.dim() != 2 or int(cell_centers.shape[1]) not in (2, 3):
            raise GeoBrainError(
                "cell_centers must be (n_cells, 2 or 3)",
                object_name="UnstructuredMesh", field="cell_centers",
                expected="(n_cells, 2|3)", actual=tuple(cell_centers.shape),
            )
        n_cells = int(cell_centers.shape[0])
        if tuple(cell_volumes.shape) != (n_cells,):
            raise GeoBrainError(
                "cell_volumes must be (n_cells,)",
                object_name="UnstructuredMesh", field="cell_volumes",
                expected=(n_cells,), actual=tuple(cell_volumes.shape),
            )
        centers64 = cell_centers.to(torch.float64)
        volumes64 = cell_volumes.to(torch.float64)
        if not bool(torch.isfinite(centers64).all()):
            raise GeoBrainError(
                "cell_centers must be finite",
                object_name="UnstructuredMesh", field="cell_centers",
                expected="all finite", actual="non-finite entry",
            )
        if not bool(torch.isfinite(volumes64).all()):
            raise GeoBrainError(
                "cell_volumes must be finite",
                object_name="UnstructuredMesh", field="cell_volumes",
                expected="all finite", actual="non-finite entry",
            )
        if bool((volumes64 <= 0).any()):
            raise GeoBrainError(
                "cell_volumes must be strictly positive",
                object_name="UnstructuredMesh", field="cell_volumes",
                expected="all > 0", actual="non-positive entry",
            )
        self._n_dim = int(cell_centers.shape[1])
        self._n_cells = n_cells
        self._validate_face_records(faces, n_cells=n_cells, n_dim=self._n_dim)
        if boundary_faces is not None:
            self._validate_boundary_records(
                boundary_faces, n_cells=n_cells, n_dim=self._n_dim,
            )
        # Canonicalize geometry to CPU float64 (device is a per-field/consumer
        # concern): a projection built from a CUDA-resident unstructured mesh
        # then has consistent-device precomputed geometry.
        self._cell_centers = cell_centers.detach().clone().to(device="cpu", dtype=torch.float64)
        self._cell_volumes = cell_volumes.detach().clone().to(device="cpu", dtype=torch.float64)
        self._faces = self._coerce_face_records(faces)
        self._boundary_faces = (
            self._coerce_boundary_records(boundary_faces)
            if boundary_faces is not None else None
        )
        # --- optional 3-D simplex topology (EdgeConnectivityMesh) ---------
        self._node_coords: torch.Tensor | None = None
        self._cell_nodes: torch.Tensor | None = None
        # Lazily derived edge SoA + signed cell→edge map (cached like _faces).
        self._edges: EdgeRecords | None = None
        self._cell_edge_indices: torch.Tensor | None = None
        self._cell_edge_signs: torch.Tensor | None = None
        if (node_coords is None) != (cell_nodes is None):
            raise GeoBrainError(
                "node_coords and cell_nodes must be supplied together",
                object_name="UnstructuredMesh", field="node_coords/cell_nodes",
                expected="both or neither",
                actual=(
                    "node_coords only" if cell_nodes is None else "cell_nodes only"
                ),
            )
        if node_coords is not None and cell_nodes is not None:
            self._init_simplex_topology(node_coords, cell_nodes)
        # --- optional region / boundary markers (region-marker labels) ----
        self._cell_markers: torch.Tensor | None = None
        self._boundary_markers: torch.Tensor | None = None
        if cell_markers is not None:
            self._cell_markers = self._coerce_markers(
                cell_markers, n_expected=n_cells, field="cell_markers",
            )
        if boundary_markers is not None:
            if self._boundary_faces is None:
                raise GeoBrainError(
                    "boundary_markers requires boundary_faces (markers are "
                    "aligned with the BoundaryRecords order)",
                    object_name="UnstructuredMesh", field="boundary_markers",
                    expected="boundary_faces supplied", actual=None,
                )
            self._boundary_markers = self._coerce_markers(
                boundary_markers,
                n_expected=int(self._boundary_faces.cell.numel()),
                field="boundary_markers",
            )

    @staticmethod
    def _coerce_markers(
        value, *, n_expected: int, field: str,
    ) -> torch.Tensor:
        """Validate + canonicalize an integer marker vector (CPU long)."""
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if value.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64,
        ):
            raise GeoBrainError(
                f"{field} must be an integer tensor (region/boundary labels)",
                object_name="UnstructuredMesh", field=field,
                expected="integer dtype", actual=str(value.dtype),
            )
        if value.dim() != 1 or int(value.shape[0]) != n_expected:
            raise GeoBrainError(
                f"{field} must be 1-D of length {n_expected}",
                object_name="UnstructuredMesh", field=field,
                expected=(n_expected,), actual=tuple(value.shape),
            )
        return value.detach().clone().to(device="cpu", dtype=torch.long)

    def cell_markers(self) -> torch.Tensor | None:
        """Optional per-cell integer region markers (CPU long), else None.

        Region ids: geometry regions survive meshing (gmsh
        physical groups via :meth:`from_meshio`, region seeds via the PLC
        builder) so consumers can map ``marker → parameter`` without
        geometric tests. Not a field: markers never project
        (:class:`MeshProjection` ignores them) and never enter the cell-field
        contract.
        """
        return self._cell_markers

    def boundary_markers(self) -> torch.Tensor | None:
        """Optional per-boundary-face integer markers (CPU long), else None.

        Aligned with the :meth:`boundary_faces` record order, the natural
        key for boundary-condition dispatch (e.g. which faces get Robin vs
        free-surface treatment).
        """
        return self._boundary_markers

    @staticmethod
    def _check_record_tensor(
        value: torch.Tensor,
        *,
        object_name: str,
        field: str,
        expected: tuple[int, ...],
        dtype: torch.dtype | None = None,
        positive: bool = False,
    ) -> None:
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise GeoBrainError(
                f"{field} has an invalid shape",
                object_name=object_name, field=field,
                expected=expected,
                actual=tuple(value.shape) if isinstance(value, torch.Tensor)
                else type(value).__name__,
            )
        if dtype is not None and value.dtype != dtype:
            raise GeoBrainError(
                f"{field} has an invalid dtype",
                object_name=object_name, field=field,
                expected=dtype, actual=value.dtype,
            )
        if not bool(torch.isfinite(value.to(torch.float64)).all()):
            raise GeoBrainError(
                f"{field} must be finite",
                object_name=object_name, field=field,
                expected="all finite", actual="non-finite entry",
            )
        if positive and not bool((value.to(torch.float64) > 0).all()):
            raise GeoBrainError(
                f"{field} must be strictly positive",
                object_name=object_name, field=field,
                expected="all > 0", actual="non-positive entry",
            )

    @classmethod
    def _validate_face_records(
        cls, faces: FaceRecords, *, n_cells: int, n_dim: int,
    ) -> None:
        if not isinstance(faces, FaceRecords):
            raise GeoBrainError(
                "faces must be a FaceRecords instance",
                object_name="UnstructuredMesh", field="faces",
                expected="FaceRecords", actual=type(faces).__name__,
            )
        nf = int(faces.cell_i.numel())
        object_name = "UnstructuredMesh.faces"
        cls._check_record_tensor(
            faces.cell_i, object_name=object_name, field="cell_i",
            expected=(nf,), dtype=torch.long,
        )
        cls._check_record_tensor(
            faces.cell_j, object_name=object_name, field="cell_j",
            expected=(nf,), dtype=torch.long,
        )
        for field in ("area", "dist_l", "dist_r"):
            cls._check_record_tensor(
                getattr(faces, field), object_name=object_name, field=field,
                expected=(nf,), positive=True,
            )
        cls._check_record_tensor(
            faces.normal, object_name=object_name, field="normal",
            expected=(nf, n_dim),
        )
        cls._check_record_tensor(
            faces.centroid, object_name=object_name, field="centroid",
            expected=(nf, n_dim),
        )
        if nf:
            if bool((faces.cell_i < 0).any()) or bool((faces.cell_j < 0).any()):
                raise GeoBrainError(
                    "faces reference a negative cell index",
                    object_name=object_name, field="cell_i/cell_j",
                    expected=f"indices in [0, {n_cells})", actual="negative index",
                )
            if int(faces.cell_i.max()) >= n_cells or int(faces.cell_j.max()) >= n_cells:
                raise GeoBrainError(
                    "faces reference a cell index >= n_cells",
                    object_name=object_name, field="cell_i/cell_j",
                    expected=f"indices in [0, {n_cells})", actual="out-of-range",
                )
            if bool((faces.cell_i == faces.cell_j).any()):
                raise GeoBrainError(
                    "internal faces must connect two distinct cells",
                    object_name=object_name, field="cell_i/cell_j",
                    expected="cell_i != cell_j", actual="self-face",
                )
            norm = torch.linalg.norm(faces.normal.to(torch.float64), dim=1)
            if not bool(torch.allclose(norm, torch.ones_like(norm), atol=1e-8, rtol=1e-8)):
                raise GeoBrainError(
                    "face normals must be unit vectors",
                    object_name=object_name, field="normal",
                    expected="||normal|| == 1", actual="non-unit normal",
                )

    @classmethod
    def _validate_boundary_records(
        cls, boundary: BoundaryRecords, *, n_cells: int, n_dim: int,
    ) -> None:
        if not isinstance(boundary, BoundaryRecords):
            raise GeoBrainError(
                "boundary_faces must be a BoundaryRecords instance",
                object_name="UnstructuredMesh", field="boundary_faces",
                expected="BoundaryRecords", actual=type(boundary).__name__,
            )
        nb = int(boundary.cell.numel())
        object_name = "UnstructuredMesh.boundary_faces"
        cls._check_record_tensor(
            boundary.cell, object_name=object_name, field="cell",
            expected=(nb,), dtype=torch.long,
        )
        cls._check_record_tensor(
            boundary.area, object_name=object_name, field="area",
            expected=(nb,), positive=True,
        )
        cls._check_record_tensor(
            boundary.normal, object_name=object_name, field="normal",
            expected=(nb, n_dim),
        )
        cls._check_record_tensor(
            boundary.centroid, object_name=object_name, field="centroid",
            expected=(nb, n_dim),
        )
        if nb:
            if bool((boundary.cell < 0).any()) or int(boundary.cell.max()) >= n_cells:
                raise GeoBrainError(
                    "boundary_faces reference a cell index out of range",
                    object_name=object_name, field="cell",
                    expected=f"indices in [0, {n_cells})", actual="out-of-range",
                )
            norm = torch.linalg.norm(boundary.normal.to(torch.float64), dim=1)
            if not bool(torch.allclose(norm, torch.ones_like(norm), atol=1e-8, rtol=1e-8)):
                raise GeoBrainError(
                    "boundary normals must be unit vectors",
                    object_name=object_name, field="normal",
                    expected="||normal|| == 1", actual="non-unit normal",
                )

    @staticmethod
    def _coerce_face_records(fr: FaceRecords) -> FaceRecords:
        # Geometry canonicalized to CPU (see __init__): device is a per-field
        # concern, so the connectivity SoA lives on CPU like the cell geometry.
        cpu = {"device": "cpu"}
        return FaceRecords(
            cell_i=fr.cell_i.detach().clone().to(dtype=torch.long, **cpu),
            cell_j=fr.cell_j.detach().clone().to(dtype=torch.long, **cpu),
            area=fr.area.detach().clone().to(dtype=torch.float64, **cpu),
            dist_l=fr.dist_l.detach().clone().to(dtype=torch.float64, **cpu),
            dist_r=fr.dist_r.detach().clone().to(dtype=torch.float64, **cpu),
            normal=fr.normal.detach().clone().to(dtype=torch.float64, **cpu),
            centroid=fr.centroid.detach().clone().to(dtype=torch.float64, **cpu),
        )

    @staticmethod
    def _coerce_boundary_records(br: BoundaryRecords) -> BoundaryRecords:
        cpu = {"device": "cpu"}
        return BoundaryRecords(
            cell=br.cell.detach().clone().to(dtype=torch.long, **cpu),
            area=br.area.detach().clone().to(dtype=torch.float64, **cpu),
            normal=br.normal.detach().clone().to(dtype=torch.float64, **cpu),
            centroid=br.centroid.detach().clone().to(dtype=torch.float64, **cpu),
        )

    def _init_simplex_topology(
        self, node_coords: torch.Tensor, cell_nodes: torch.Tensor,
    ) -> None:
        """Validate + store the tetrahedral node table and refine the instance
        capability declaration with :class:`EdgeConnectivityMesh`."""
        if self._n_dim != 3:
            raise GeoBrainError(
                "simplex topology (node_coords + cell_nodes) is 3-D tetrahedral "
                "only",
                object_name="UnstructuredMesh", field="node_coords/cell_nodes",
                expected="n_dim == 3", actual=f"n_dim == {self._n_dim}",
            )
        if not isinstance(node_coords, torch.Tensor):
            raise GeoBrainError(
                "node_coords must be a Tensor",
                object_name="UnstructuredMesh", field="node_coords",
                expected="torch.Tensor", actual=type(node_coords).__name__,
            )
        if not isinstance(cell_nodes, torch.Tensor):
            raise GeoBrainError(
                "cell_nodes must be a Tensor",
                object_name="UnstructuredMesh", field="cell_nodes",
                expected="torch.Tensor", actual=type(cell_nodes).__name__,
            )
        if cell_nodes.dtype != torch.long:
            raise GeoBrainError(
                "cell_nodes must use torch.long node indices",
                object_name="UnstructuredMesh", field="cell_nodes",
                expected=torch.long, actual=cell_nodes.dtype,
            )
        # Defensive copies (same discipline as the constructor's other inputs):
        # ``.to`` alone is a no-op when the dtype already matches, which would
        # store the CALLER's tensor by reference and let later in-place edits
        # silently corrupt the cached topology/derived edge records.
        coords = node_coords.detach().clone().to(device="cpu", dtype=torch.float64)
        cn = cell_nodes.detach().clone().cpu()
        if coords.dim() != 2 or int(coords.shape[1]) != 3:
            raise GeoBrainError(
                "node_coords must be (n_nodes, 3)",
                object_name="UnstructuredMesh", field="node_coords",
                expected="(n_nodes, 3)", actual=tuple(coords.shape),
            )
        if not bool(torch.isfinite(coords).all()):
            raise GeoBrainError(
                "node_coords must be finite",
                object_name="UnstructuredMesh", field="node_coords",
                expected="all finite", actual="non-finite entry",
            )
        if tuple(cn.shape) != (self._n_cells, 4):
            raise GeoBrainError(
                "cell_nodes must be (n_cells, 4) tetrahedral node ids",
                object_name="UnstructuredMesh", field="cell_nodes",
                expected=(self._n_cells, 4), actual=tuple(cn.shape),
            )
        n_nodes = int(coords.shape[0])
        if int(cn.numel()) and (int(cn.min()) < 0 or int(cn.max()) >= n_nodes):
            raise GeoBrainError(
                "cell_nodes reference a node index out of range",
                object_name="UnstructuredMesh", field="cell_nodes",
                expected=f"indices in [0, {n_nodes})", actual="out-of-range",
            )
        srt = cn.sort(dim=1).values
        if bool((srt[:, 1:] == srt[:, :-1]).any()):
            raise GeoBrainError(
                "a cell repeats a node id (degenerate tetrahedron)",
                object_name="UnstructuredMesh", field="cell_nodes",
                expected="4 distinct node ids per cell", actual="repeated id",
            )
        self._node_coords = coords
        self._cell_nodes = cn
        # Capability refinement BY INSTANCE (still explicit declaration): only
        # an instance that actually carries simplex topology declares
        # EdgeConnectivityMesh; 2-D / SoA-direct instances keep the class default.
        self.mesh_capabilities = frozenset(
            {GeometryMesh, ConnectivityMesh, EdgeConnectivityMesh}
        )

    # Tombstones raise AttributeError (NOT NotImplementedError) so that
    # ``hasattr(mesh, "shape")`` is False and ``getattr(mesh, "shape", None)``
    # returns the default: standard attribute-protocol semantics, while the
    # message still guides users toward the capability idiom.
    @property
    def shape(self) -> tuple[int, ...]:
        raise AttributeError(
            "UnstructuredMesh has no structured shape; it is not a StructuredMesh"
        )

    @property
    def spacing(self) -> tuple[float, ...]:
        raise AttributeError(
            "UnstructuredMesh has no structured spacing; it is not a StructuredMesh"
        )

    def cell_centers(self) -> torch.Tensor:
        return self._cell_centers.clone()

    def cell_volumes(self) -> torch.Tensor:
        return self._cell_volumes.clone()

    # --- ConnectivityMesh ------------------------------------------------
    def face_neighbors(self) -> FaceRecords:
        return self._faces.clone()

    def boundary_faces(self) -> BoundaryRecords:
        if self._boundary_faces is not None:
            return self._boundary_faces.clone()
        return BoundaryRecords(
            cell=torch.empty(0, dtype=torch.long),
            area=torch.empty(0, dtype=torch.float64),
            normal=torch.empty(0, self._n_dim, dtype=torch.float64),
            centroid=torch.empty(0, self._n_dim, dtype=torch.float64),
        )

    def to_dict(self) -> dict:
        """Declarative, JSON-native serialization (round-trips via ``mesh_from_dict``).

        Schema (``version`` is :data:`~geobrain.mesh.serialization.MESH_SCHEMA_VERSION`)::

            {"type": "UnstructuredMesh", "version": 1,
             "n_dim": 2 | 3,
             "cell_centers": <tensor-envelope (n_cells, n_dim)>,
             "cell_volumes": <tensor-envelope (n_cells,)>,
             "faces": {<FaceRecords SoA: cell_i/cell_j/area/dist_l/dist_r/
                        normal/centroid, each a tensor-envelope>},
             "boundary_faces": {<BoundaryRecords SoA>} | null,
             "node_coords": <tensor-envelope (n_nodes, 3)> | null,
             "cell_nodes": <tensor-envelope (n_cells, 4) int64> | null}

        Stores exactly the SoA the direct constructor consumes, so the factory
        rebuilds an identical mesh: ``cell_centers`` / ``cell_volumes`` /
        ``face_neighbors()`` / ``boundary_faces()`` all match to float64, and a
        3-D simplex-built mesh's ``node_coords`` / ``cell_nodes`` (hence its
        :class:`EdgeConnectivityMesh` edge records) round-trip too. ``null`` is
        emitted for an absent boundary or absent simplex topology so those
        optional slots are faithfully restored (not silently materialised).
        """
        from .serialization import (
            MESH_SCHEMA_VERSION, _ser_boundary_records, _ser_face_records,
            _ser_tensor,
        )

        return {
            "type": "UnstructuredMesh",
            "version": MESH_SCHEMA_VERSION,
            "n_dim": int(self._n_dim),
            "cell_centers": _ser_tensor(self._cell_centers),
            "cell_volumes": _ser_tensor(self._cell_volumes),
            "faces": _ser_face_records(self._faces),
            "boundary_faces": (
                _ser_boundary_records(self._boundary_faces)
                if self._boundary_faces is not None else None
            ),
            "node_coords": (
                _ser_tensor(self._node_coords)
                if self._node_coords is not None else None
            ),
            "cell_nodes": (
                _ser_tensor(self._cell_nodes)
                if self._cell_nodes is not None else None
            ),
            "cell_markers": (
                _ser_tensor(self._cell_markers)
                if self._cell_markers is not None else None
            ),
            "boundary_markers": (
                _ser_tensor(self._boundary_markers)
                if self._boundary_markers is not None else None
            ),
        }

    def append_connections(self, extra: FaceRecords) -> "UnstructuredMesh":
        """Return a new mesh with ``extra`` internal faces appended (NNC hook)."""
        merged = FaceRecords.concat(self._faces, extra)
        return UnstructuredMesh(
            self._cell_centers, self._cell_volumes, merged,
            boundary_faces=self._boundary_faces,
            # NNC adds FV connections only: node topology is untouched, so a
            # simplex-built mesh keeps its EdgeConnectivityMesh declaration
            # (and cells/boundary keep their markers).
            node_coords=self._node_coords, cell_nodes=self._cell_nodes,
            cell_markers=self._cell_markers,
            boundary_markers=self._boundary_markers,
        )

    # --- EdgeConnectivityMesh (3-D simplex-built instances only) ---------
    def _require_simplex_topology(
        self, method: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._node_coords is None or self._cell_nodes is None:
            raise GeoBrainError(
                f"{method} needs stored simplex topology, which this instance "
                "does not carry. Edge connectivity is available only on 3-D "
                "tetrahedral meshes built via from_simplices / from_points on a "
                "3-D cloud (or the direct constructor fed node_coords= and "
                "cell_nodes=); 2-D from_polygons meshes have no tetrahedral "
                "edge structure.",
                object_name="UnstructuredMesh", field="node_coords/cell_nodes",
                expected="3-D simplex-built mesh", actual="topology absent",
            )
        return self._node_coords, self._cell_nodes

    def node_coords(self) -> torch.Tensor:
        """``(n_nodes, 3)`` float64 node coordinates (stored builder input).

        Part of the :class:`EdgeConnectivityMesh` contract, Whitney edge-basis
        assembly needs them for the barycentric-coordinate gradients.

        Raises:
            GeoBrainError: on an instance without simplex topology (2-D
                ``from_polygons`` or SoA-direct without ``node_coords=``).
        """
        coords, _ = self._require_simplex_topology("node_coords()")
        return coords.clone()

    def cell_nodes(self) -> torch.Tensor:
        """``(n_cells, 4)`` long cell→node table, rows in STORED builder order.

        Part of the :class:`EdgeConnectivityMesh` contract. The stored order
        defines the local-edge convention (see :meth:`cell_edges`), so it is
        returned exactly as built, never sorted.

        Raises:
            GeoBrainError: on an instance without simplex topology.
        """
        _, cn = self._require_simplex_topology("cell_nodes()")
        return cn.clone()

    def edge_records(self) -> EdgeRecords:
        """Unique-edge struct-of-arrays (lazily derived, cached like the face SoA).

        Every edge appears once, keyed by its canonical node pair
        ``small id → large id`` (so ``nodes[:, 0] < nodes[:, 1]`` always);
        ``tangent`` is the unit vector along that canonical direction, with
        ``length`` and ``midpoint`` from the same node coordinates. Edge order
        is deterministic: lexicographic in the canonical node pair
        (``torch.unique(dim=0)``).

        Raises:
            GeoBrainError: on an instance without simplex topology, or on a
                degenerate zero-length edge (two nodes of a cell coincide).
        """
        if self._edges is None:
            self._derive_edges()
        assert self._edges is not None
        return self._edges.clone()

    def cell_edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Signed cell→edge incidence: ``(n_cells, 6)`` long ids + float64 signs.

        Local-edge convention (fixed contract, Whitney assembly indexes by
        it): with the cell's four nodes in STORED order, local edges are the
        node pairs ``(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)``, local
        direction = first listed node → second. Column ``k`` of both returned
        tensors refers to local edge ``k``; the id indexes into
        :meth:`edge_records`, and the sign is ``+1`` where the local direction
        agrees with the edge's canonical direction (small node id → large) and
        ``-1`` where it is reversed.

        Raises:
            GeoBrainError: on an instance without simplex topology.
        """
        if self._cell_edge_indices is None or self._cell_edge_signs is None:
            self._derive_edges()
        assert self._cell_edge_indices is not None
        assert self._cell_edge_signs is not None
        return self._cell_edge_indices.clone(), self._cell_edge_signs.clone()

    def _derive_edges(self) -> None:
        """Vectorised unique-edge derivation (canonical simplex edge numbering;
        cached on first use).

        Pipeline: cell→node ``(nc, 4)`` → six local node pairs ``(nc, 6, 2)``
        → per-pair sort to the canonical ``small id → large id`` form while
        recording which pairs flipped → flatten + ``torch.unique(dim=0)`` for
        the global edge table (deterministic lexicographic order) with the
        inverse map reshaped to the ``(nc, 6)`` cell→edge ids → signs from the
        flip mask. Geometry (tangent/length/midpoint) indexes ``node_coords``.
        """
        coords, cn = self._require_simplex_topology("edge_records()")
        local = _TET_LOCAL_EDGES.to(cn.device)
        pairs = cn[:, local]                                  # (nc, 6, 2)
        flipped = pairs[..., 0] > pairs[..., 1]               # (nc, 6) bool
        canon = torch.stack(
            [pairs.amin(dim=-1), pairs.amax(dim=-1)], dim=-1,
        ).reshape(-1, 2)                                      # (nc*6, 2)
        nodes, inverse = torch.unique(canon, dim=0, return_inverse=True)
        vec = coords[nodes[:, 1]] - coords[nodes[:, 0]]       # canonical dir
        length = torch.linalg.norm(vec, dim=1)
        if bool((length <= 0).any()):
            raise GeoBrainError(
                "degenerate zero-length edge, two nodes of a cell share "
                "coordinates",
                object_name="UnstructuredMesh", field="node_coords",
                expected="all edge lengths > 0", actual="zero-length edge",
            )
        self._edges = EdgeRecords(
            nodes=nodes,
            tangent=vec / length.unsqueeze(1),
            length=length,
            midpoint=0.5 * (coords[nodes[:, 0]] + coords[nodes[:, 1]]),
        )
        self._cell_edge_indices = inverse.reshape(self._n_cells, 6)
        # +1 where the local direction (first listed node → second) agrees with
        # the canonical direction, -1 where the pair was flipped.
        self._cell_edge_signs = 1.0 - 2.0 * flipped.to(torch.float64)

    # --- 2-D polygon builder --------------------------------------------
    @classmethod
    def from_mesh(
        cls,
        mesh,
        *,
        extra_faces: FaceRecords | None = None,
    ) -> "UnstructuredMesh":
        """Build an :class:`UnstructuredMesh` from any ConnectivityMesh's own SoA.

        The first-class form of the validated demotion pattern (see the class
        docstring: bit-identical DC3D voltages when built from a TensorMesh's
        records): copies ``cell_centers`` / ``cell_volumes`` /
        ``face_neighbors`` (with ``extra_faces`` concatenated when given, the
        NNC case) **and** ``boundary_faces``, the hand-rolled constructor
        calls this replaces silently dropped the boundary records. A 3-D
        simplex-built source (declares ``EdgeConnectivityMesh``) additionally
        carries its node topology into the copy.

        Args:
            mesh: Any mesh declaring
                :class:`~geobrain.mesh.capabilities.ConnectivityMesh`.
            extra_faces: Optional extra internal-face records appended to the
                source's ``face_neighbors()`` (non-neighbour connections).

        Raises:
            GeoBrainError: when ``mesh`` lacks the ConnectivityMesh capability.
        """
        ensure_capable_mesh(mesh, ConnectivityMesh,
                            owner="UnstructuredMesh.from_mesh")
        faces = mesh.face_neighbors()
        if extra_faces is not None:
            faces = FaceRecords.concat(faces, extra_faces)
        topo: dict = {}
        if mesh.declares(EdgeConnectivityMesh):
            topo = {"node_coords": mesh.node_coords(),
                    "cell_nodes": mesh.cell_nodes()}
        return cls(
            mesh.cell_centers(),
            mesh.cell_volumes(),
            faces,
            boundary_faces=mesh.boundary_faces(),
            **topo,
        )

    @classmethod
    def from_polygons(
        cls,
        vertices: torch.Tensor,
        cells: Sequence[Sequence[int]] | torch.Tensor,
        *,
        cell_center: str = "centroid",
        axes: str = "zxy",
        cell_markers: torch.Tensor | None = None,
    ) -> "UnstructuredMesh":
        """Build a 2-D mesh from vertices + polygon vertex-loops (derives the SoA).

        Args:
            vertices: ``(n_vert, 2)`` float64 vertex coordinates in platform
                column order **``(z, x)``**, column 0 = z (depth, positive
                down), column 1 = x, matching the ``(nz, nx)`` TensorMesh axis
                order. Pass a plain ``(x, z)`` cloud only after transposing.
            cells: sequence of polygon vertex-loops (CCW or CW); each loop is a
                sequence of vertex indices, ``>= 3`` per cell. An ``(m, k)``
                int tensor is accepted too (normalised on entry, mirroring
                :meth:`from_simplices`).
            cell_center: which point to use as the cell's FV centre.

                - ``"centroid"`` (default, back-compat): the area centroid.
                  Robust on arbitrary polygons but the TPFA two-point flux is
                  only *approximately* consistent (the centroid→face line need
                  not be perpendicular to the face).
                - ``"circumcenter"``: the triangle circumcentre (solved from the
                  standard 2×2 perpendicular-bisector system). On a **Delaunay**
                  triangulation the circumcentres are the Voronoi sites, so the
                  cell-point→face segment is perpendicular to the shared face
                  (Voronoi–Delaunay orthogonality) and TPFA becomes consistent,
                  but ONLY where the triangulation is additionally
                  *self-centered* (each circumcentre inside its own triangle, so
                  the two cell points straddle their shared edge). For a
                  non-self-centered face (an obtuse triangle's circumcentre on the
                  wrong side of a shared edge) ``dist_l``/``dist_r`` are rescaled
                  to keep ``dist_l+dist_r`` equal to the projected separation and
                  a one-time ``UserWarning`` is emitted; the two-point flux there
                  is only approximate, so ``"centroid"`` (always straddling, the
                  robust default) is preferred unless the mesh is verified
                  self-centered. **Restricted to pure-triangle meshes**, every
                  cell must have exactly 3 vertices, else a
                  :class:`GeoBrainError` is raised.
            axes: coordinate-column order of the input coordinates:
                'zxy' (default; platform mesh order, z first) or
                'xyz' (world order, permuted internally via
                xyz_to_mesh_axes).

        Raises:
            GeoBrainError: on non-2-D vertices, a cell with ``< 3`` vertices, an
                out-of-range vertex index, a degenerate (zero-area) polygon, a
                self-intersecting (non-simple) polygon loop, e.g. a "bowtie"
                quad, a ``circumcenter`` request on a non-triangle cell, or a
                degenerate circumcentre that lands on a face plane (sliver
                triangle; see the per-face guard; switch to ``"centroid"`` or
                supply a Delaunay-quality mesh).
        """
        if cell_center not in ("centroid", "circumcenter"):
            raise GeoBrainError(
                "cell_center must be 'centroid' or 'circumcenter'",
                object_name="UnstructuredMesh.from_polygons", field="cell_center",
                expected="'centroid' | 'circumcenter'", actual=cell_center,
            )
        vertices = resolve_axes_kwarg(vertices, axes,
                                      "UnstructuredMesh.from_polygons")
        verts = vertices.to(torch.float64)
        if verts.dim() != 2 or int(verts.shape[1]) != 2:
            raise GeoBrainError(
                "from_polygons is 2-D: vertices must be (n_vert, 2)",
                object_name="UnstructuredMesh.from_polygons", field="vertices",
                expected="(n_vert, 2)", actual=tuple(verts.shape),
            )
        if isinstance(cells, torch.Tensor):
            cells = cells.to(torch.long).tolist()
        n_vert = int(verts.shape[0])
        centers, vols = [], []
        for ci, loop in enumerate(cells):
            if len(loop) < 3:
                raise GeoBrainError(
                    "each cell needs >= 3 vertices",
                    object_name="UnstructuredMesh.from_polygons", field="cells",
                    expected=">= 3 vertices", actual=f"cell {ci} has {len(loop)}",
                )
            if cell_center == "circumcenter" and len(loop) != 3:
                raise GeoBrainError(
                    "cell_center='circumcenter' requires a pure-triangle mesh",
                    object_name="UnstructuredMesh.from_polygons", field="cells",
                    expected="3 vertices per cell",
                    actual=f"cell {ci} has {len(loop)}",
                )
            if any(v < 0 or v >= n_vert for v in loop):
                raise GeoBrainError(
                    "cell vertex index out of range",
                    object_name="UnstructuredMesh.from_polygons", field="cells",
                    expected=f"indices in [0, {n_vert})", actual=f"cell {ci}",
                )
            p = verts[torch.tensor(loop, dtype=torch.long)]
            x, y = p[:, 0], p[:, 1]
            xn, yn = torch.roll(x, -1), torch.roll(y, -1)
            cross = x * yn - xn * y
            a2 = cross.sum()
            area = 0.5 * torch.abs(a2)
            if float(area) <= 0.0:
                raise GeoBrainError(
                    "degenerate (zero-area) polygon",
                    object_name="UnstructuredMesh.from_polygons", field="cells",
                    expected="positive area", actual=f"cell {ci} area=0",
                )
            # Reject a self-intersecting (non-simple) loop: e.g. a "bowtie" quad
            # whose two lobes give a nonzero signed area that slips past the
            # zero-area guard above. Cheap (only k>=4 loops are scanned); a valid
            # non-convex-but-simple polygon (a U-shaped cell) is accepted.
            if _polygon_self_intersects(p):
                raise GeoBrainError(
                    "self-intersecting (non-simple) polygon loop",
                    object_name="UnstructuredMesh.from_polygons", field="cells",
                    expected="a simple (non-self-intersecting) polygon",
                    actual=f"cell {ci} has crossing edges",
                )
            if cell_center == "circumcenter":
                centers.append(triangle_circumcenter(p[0], p[1], p[2], ci))
            else:
                cx = ((x + xn) * cross).sum() / (3.0 * a2)
                cy = ((y + yn) * cross).sum() / (3.0 * a2)
                centers.append(torch.stack([cx, cy]))
            vols.append(area)
        cell_centers = torch.stack(centers)
        cell_volumes = torch.stack(vols)
        faces, boundary = derive_faces_2d(verts, cells, cell_centers)
        return cls(cell_centers, cell_volumes, faces, boundary_faces=boundary,
                   cell_markers=cell_markers)

    # --- 3-D tetrahedron builder ----------------------------------------
    @classmethod
    def from_simplices(
        cls,
        vertices: torch.Tensor,
        simplices: Sequence[Sequence[int]] | torch.Tensor,
        *,
        cell_center: str = "centroid",
        axes: str = "zxy",
        cell_markers: torch.Tensor | None = None,
    ) -> "UnstructuredMesh":
        """Build a 3-D mesh from vertices + tetrahedra (derives the SoA).

        The 3-D analogue of :meth:`from_polygons`: each tetrahedron becomes one
        cell; its four triangular faces are keyed by the ``frozenset`` of their
        three vertex ids so a face shared by two tets is detected and emitted as
        one internal :class:`FaceRecords` entry (an unshared face becomes a
        :class:`BoundaryRecords` entry).

        Args:
            vertices: ``(n_vert, 3)`` float64 vertex coordinates in platform
                column order **``(z, x, y)``**, column 0 = z (depth, positive
                down), then x, then y, matching the ``(nz, nx, ny)`` TensorMesh
                axis order. Pass a plain ``(x, y, z)`` cloud only after
                permuting its columns to ``(z, x, y)``.
            simplices: ``(m, 4)`` int vertex indices (sequence or tensor), one
                row per tetrahedron ``(a, b, c, d)``.
            cell_center: which point to use as the cell's FV centre.

                - ``"centroid"`` (default): the mean of the four vertices.
                  Robust on low-quality elements but the TPFA two-point flux is
                  only *approximately* consistent (the centroid→face segment is
                  not in general perpendicular to the face).
                - ``"circumcenter"``: the circumsphere centre (solved from the
                  3×3 system ``2(B−A)·x = |B|²−|A|²`` and its B→C, B→D rows). The
                  Voronoi–Delaunay-orthogonality argument (circumcentres are the
                  Voronoi sites, so the cell-point→face segment is perpendicular
                  to the shared face, making the harmonic-σ TPFA transmissibility
                  in :func:`assemble_poisson_fv` *consistent*) holds ONLY when the
                  tetrahedralisation is additionally **self-centered**, every
                  circumcentre lies inside its own element, so the two cell points
                  of a shared face straddle it. A plain Delaunay mesh does NOT
                  guarantee this: 3-D Delaunay output from realistic clouds
                  routinely has non-self-centered (obtuse) tets whose circumcentre
                  falls on the *wrong* side of a shared face, and there
                  ``dist_l + dist_r`` no longer equals the true cell-point
                  separation. Those faces are RESCALED (split of the projected
                  separation ``|(c_j-c_i)·n|`` so ``dist_l+dist_r`` stays exact)
                  and a one-time ``UserWarning`` is emitted, but the two-point
                  flux there is only approximate. Prefer the ``"centroid"``
                  default (robust: centroids always straddle their shared faces);
                  use ``"circumcenter"`` only for a verified self-centered mesh.
            axes: coordinate-column order of the input coordinates:
                'zxy' (default; platform mesh order, z first) or
                'xyz' (world order, permuted internally via
                xyz_to_mesh_axes).

        Per-tet geometry: volume ``= |det([b−a, c−a, d−a])| / 6``; each face area
        ``= |cross| / 2`` with the unit cross-product normal oriented from
        ``cell_i`` to ``cell_j``. ``dist_l``/``dist_r`` are the perpendicular
        distances from each side's cell point to the face plane when the two
        points straddle it (the self-centered case, ``dist_l+dist_r`` equals the
        projected separation exactly); for a non-straddling (non-self-centered)
        face they are instead the ``|(c_j-c_i)·n|`` split described above.

        The built mesh retains ``vertices``/``simplices`` as its node
        coordinates and cell→node table and declares
        :class:`EdgeConnectivityMesh` (see :meth:`edge_records` /
        :meth:`cell_edges`).

        Raises:
            GeoBrainError: on non-3-D vertices, a simplex without exactly 4
                indices, an out-of-range vertex index, a degenerate
                (zero-volume) tet, a degenerate circumsphere system, or, the
                **degenerate-geometry guard**, a cell point that lands ON a
                face plane (``dist_l``/``dist_r`` ``<= 1e-12``, which happens
                with circumcentres of sliver/obtuse elements). In the last case
                supply a Delaunay-quality mesh or use ``cell_center="centroid"``.
        """
        if cell_center not in ("centroid", "circumcenter"):
            raise GeoBrainError(
                "cell_center must be 'centroid' or 'circumcenter'",
                object_name="UnstructuredMesh.from_simplices", field="cell_center",
                expected="'centroid' | 'circumcenter'", actual=cell_center,
            )
        vertices = resolve_axes_kwarg(vertices, axes,
                                      "UnstructuredMesh.from_simplices")
        verts = vertices.to(torch.float64)
        if verts.dim() != 2 or int(verts.shape[1]) != 3:
            raise GeoBrainError(
                "from_simplices is 3-D: vertices must be (n_vert, 3)",
                object_name="UnstructuredMesh.from_simplices", field="vertices",
                expected="(n_vert, 3)", actual=tuple(verts.shape),
            )
        n_vert = int(verts.shape[0])
        # Vectorized validation + per-tet geometry: the per-cell scalar
        # arithmetic is retained as the test-suite oracle; values match it
        # to ~1 ulp and error messages keep their first-offender cell index.
        if isinstance(simplices, torch.Tensor):
            # Vertex device wins (the historical ``.tolist()`` normalization
            # was device-agnostic; mixed-device inputs must keep working).
            cn = simplices.to(dtype=torch.long, device=verts.device)
            if cn.dim() != 2 or int(cn.shape[1]) != 4:
                raise GeoBrainError(
                    "each simplex needs exactly 4 vertices",
                    object_name="UnstructuredMesh.from_simplices",
                    field="simplices",
                    expected="(n_cells, 4)", actual=tuple(cn.shape),
                )
        else:
            for ci, tet in enumerate(simplices):
                if len(tet) != 4:
                    raise GeoBrainError(
                        "each simplex needs exactly 4 vertices",
                        object_name="UnstructuredMesh.from_simplices",
                        field="simplices",
                        expected="4 vertices", actual=f"cell {ci} has {len(tet)}",
                    )
            cn = torch.tensor(
                [[int(v) for v in tet] for tet in simplices], dtype=torch.long,
            )
        oob = (cn < 0) | (cn >= n_vert)
        if bool(oob.any()):
            ci0 = int(oob.any(dim=1).long().argmax())
            raise GeoBrainError(
                "simplex vertex index out of range",
                object_name="UnstructuredMesh.from_simplices", field="simplices",
                expected=f"indices in [0, {n_vert})", actual=f"cell {ci0}",
            )
        tv = verts[cn]                                    # (nc, 4, 3)
        a = tv[:, 0]
        m = torch.stack(
            [tv[:, 1] - a, tv[:, 2] - a, tv[:, 3] - a], dim=1)   # (nc, 3, 3)
        cell_volumes = torch.abs(torch.linalg.det(m)) / 6.0
        zero = cell_volumes <= 0.0
        if bool(zero.any()):
            ci0 = int(zero.long().argmax())
            raise GeoBrainError(
                "degenerate (zero-volume) tetrahedron",
                object_name="UnstructuredMesh.from_simplices", field="simplices",
                expected="positive volume", actual=f"cell {ci0} volume=0",
            )
        if cell_center == "circumcenter":
            m2 = 2.0 * m
            sq = (tv * tv).sum(dim=2)                     # (nc, 4) |v|² rows
            rhs = torch.stack(
                [sq[:, 1] - sq[:, 0], sq[:, 2] - sq[:, 0], sq[:, 3] - sq[:, 0]],
                dim=1)
            deg = torch.abs(torch.linalg.det(m2)) <= 1e-30
            if bool(deg.any()):
                ci0 = int(deg.long().argmax())
                raise GeoBrainError(
                    "degenerate (coplanar) tetrahedron, no circumsphere centre",
                    object_name="UnstructuredMesh.from_simplices",
                    field="simplices",
                    expected="non-coplanar tetrahedron", actual=f"cell {ci0}",
                )
            cell_centers = torch.linalg.solve(m2, rhs)
        else:
            cell_centers = (tv[:, 0] + tv[:, 1] + tv[:, 2] + tv[:, 3]) / 4.0
        faces, boundary = derive_faces_3d(verts, cn, cell_centers)
        # Retain the node coordinates + cell→node table (STORED row order, the
        # local-edge convention of cell_edges() is defined over it): they are
        # the EdgeConnectivityMesh substrate for the physics-side Whitney
        # edge-element assembly.
        cell_nodes = cn.to(device=verts.device)
        return cls(
            cell_centers, cell_volumes, faces, boundary_faces=boundary,
            node_coords=verts, cell_nodes=cell_nodes,
            cell_markers=cell_markers,
        )

    # --- scipy Delaunay convenience builder -----------------------------
    # Conservative threshold for the non-destructive mesh-quality warning in
    # ``from_points``: the spread of the per-face transmissibility geometric
    # factor ``area / (dist_l + dist_r)`` (the σ=1 TPFA coupling), measured as
    # the 99.9th-percentile-over-median ratio, above which the point cloud is
    # flagged as too harshly graded for a reliable centroid-TPFA solve.
    #
    # Why p99.9/median and not max/median: a single worst circumcentre outlier
    # (3-D Delaunay tets are frequently NOT self-centred, so one near-face
    # circumcentre gives a tiny dist_l+dist_r) pushes max/median to ~400-500×
    # even on a perfectly healthy quasi-uniform cloud: too noisy to threshold.
    # The 99.9th percentile is robust to that lone outlier while still tracking
    # the BODY of the sliver population. This threshold is calibrated for the
    # CENTROID default (the case the from_points default and this warning target)
    #, centroids always straddle their shared faces, so dist_l+dist_r is a clean
    # transmissibility distance. Measured separation (DC3D half-space diagnostic,
    # ~8-11 k faces):
    #     quasi-uniform / gently-densified : p99.9/median ≈ 3 (centroid)
    #     harsh electrode-refinement shells: p99.9/median ≈ 71-77 (centroid)
    # so 50× sits comfortably above a healthy centroid mesh yet below the
    # pathological gradation the failed probe used.
    #
    # NOTE (post TPFA-distance fix): with cell_center='circumcenter' on a
    # non-self-centered mesh the corrected dist_l/dist_r (which now sum to the
    # true |(c_j-c_i)·n| instead of the old unsigned over-estimate) make this
    # spread far larger (p99.9/median ~1000× even on a quasi-uniform cloud). That
    # reflects the non-self-centered circumcentre geometry, NOT harsh spacing, and
    # is surfaced precisely by the dedicated non-self-centered warning in
    # derive_faces_2d/3d; the coarse spread heuristic here is centroid-calibrated
    # and will over-report on circumcentre meshes, so prefer the centroid default.
    #
    # Chosen as a WARNING (not a drop / not an error): an aspect-based DROP
    # filter strong enough to remove the offending slivers also DISCONNECTS the
    # mesh (the DC system goes singular at frac ≳ 0.08), a strictly worse
    # failure mode than the bounded ~2× accuracy hit the slivers cause.
    _GRADATION_TSPREAD_WARN: ClassVar[float] = 50.0
    # Below this internal-face count the p99.9 percentile is too granular to be
    # a meaningful population statistic, so the gradation probe is skipped.
    _GRADATION_MIN_FACES: ClassVar[int] = 200

    @classmethod
    def from_meshio(
        cls,
        mesh_or_path,
        *,
        axes: str = "xyz",
        cell_center: str = "centroid",
    ) -> "UnstructuredMesh":
        """Build from a :class:`meshio.Mesh` or any path meshio can read.

        The ingestion door for externally-generated quality meshes (Gmsh /
        TetGen / .vtu / 30+ formats): ``tetra`` blocks (concatenated if
        several) build a 3-D simplex mesh via :meth:`from_simplices`;
        ``triangle`` blocks build a 2-D mesh via :meth:`from_polygons`
        (3-D-embedded triangles with a constant world-z plane are squeezed
        to 2-D). Other cell families raise a structured error.

        Args:
            mesh_or_path: A ``meshio.Mesh`` instance, or a ``str`` /
                ``os.PathLike`` handed to ``meshio.read``.
            axes: ``"xyz"`` (DEFAULT; note this deliberately differs from
                the in-memory sibling builders, whose default is ``"zxy"``):
                external mesher files are universally world-ordered, so the
                points are column-permuted world ``(x, y, z)`` → mesh
                ``(z, x, y)`` on entry. Pass ``"zxy"`` for a file already in
                platform order.
            cell_center: Forwarded to the underlying builder.

        Raises:
            GeoBrainError: If meshio is not installed, the mesh holds no
                supported cell block, or ``axes`` is invalid.
        """
        try:
            import meshio  # optional extra (roadmap optional-dependency policy)
        except ImportError as exc:  # pragma: no cover - env without meshio
            raise GeoBrainError(
                "UnstructuredMesh.from_meshio requires the optional 'meshio' "
                "package, install with `pip install meshio`",
                object_name="UnstructuredMesh.from_meshio",
                field="meshio",
                expected="meshio importable",
                actual=None,
            ) from exc

        if axes not in ("xyz", "zxy"):
            raise GeoBrainError(
                "from_meshio axes must be 'xyz' (world-order file, default) "
                "or 'zxy' (platform-order file)",
                object_name="UnstructuredMesh.from_meshio",
                field="axes",
                expected="'xyz' | 'zxy'",
                actual=axes,
            )

        mm = mesh_or_path
        if not isinstance(mm, meshio.Mesh):
            mm = meshio.read(mesh_or_path)

        blocks = getattr(mm, "cells_dict", {})
        # gmsh *physical groups* → markers: cell_data_dict concatenates the
        # per-block "gmsh:physical" tags in the same per-type order as
        # cells_dict, so tags stay row-aligned with the cell blocks below.
        phys = getattr(mm, "cell_data_dict", {}).get("gmsh:physical", {})
        tets = blocks.get("tetra")
        tris = blocks.get("triangle")
        if tets is None and tris is None:
            raise GeoBrainError(
                "from_meshio supports 'tetra' (3-D) and 'triangle' (2-D) cell "
                "blocks; none found",
                object_name="UnstructuredMesh.from_meshio",
                field="cells",
                expected="'tetra' or 'triangle' blocks",
                actual=sorted(blocks),
            )

        points = torch.as_tensor(mm.points, dtype=torch.float64)

        if tets is not None:
            if points.shape[1] != 3:
                raise GeoBrainError(
                    "tetra blocks need (n, 3) points",
                    object_name="UnstructuredMesh.from_meshio",
                    field="points",
                    expected="(n, 3)",
                    actual=tuple(points.shape),
                )
            if axes == "xyz":
                points = xyz_to_mesh_axes(points)
            cells = torch.as_tensor(tets, dtype=torch.long)
            cm = phys.get("tetra")
            mesh = cls.from_simplices(
                points, cells, cell_center=cell_center,
                cell_markers=(
                    torch.as_tensor(cm.copy(), dtype=torch.long)
                    if cm is not None else None
                ),
            )
            # tagged surface triangles → boundary markers (internal tagged
            # surfaces simply find no matching boundary face and are skipped)
            return cls._lift_boundary_markers(
                mesh, points, blocks.get("triangle"), phys.get("triangle"),
            )

        # triangle path (2-D). A 3-D-embedded constant plane is squeezed:
        # the constant world column is dropped BEFORE the axes permutation.
        if points.shape[1] == 3:
            spans = (points.max(dim=0).values - points.min(dim=0).values).abs()
            flat = torch.nonzero(spans < 1e-12).reshape(-1)
            if flat.numel() != 1:
                raise GeoBrainError(
                    "3-D-embedded triangle mesh must lie in a constant "
                    "coordinate plane to squeeze to 2-D",
                    object_name="UnstructuredMesh.from_meshio",
                    field="points",
                    expected="exactly one constant world column",
                    actual=spans.tolist(),
                )
            keep = [i for i in range(3) if i != int(flat)]
            points = points[:, keep]
        if axes == "xyz":
            points = xyz_to_mesh_axes(points)
        polys = [list(map(int, row)) for row in tris]
        cm = phys.get("triangle")
        mesh = cls.from_polygons(
            points, polys, cell_center=cell_center,
            cell_markers=(
                torch.as_tensor(cm.copy(), dtype=torch.long)
                if cm is not None else None
            ),
        )
        return cls._lift_boundary_markers(
            mesh, points, blocks.get("line"), phys.get("line"),
        )

    @staticmethod
    def _lift_boundary_markers(
        mesh: "UnstructuredMesh",
        points: torch.Tensor,
        elements,
        tags,
    ) -> "UnstructuredMesh":
        """Attach gmsh boundary physical tags to a freshly built mesh.

        ``elements`` are the tagged lower-dimensional cells (``line`` in 2-D,
        ``triangle`` in 3-D) in the SAME transformed ``points`` frame the mesh
        was built from; each is matched to a :class:`BoundaryRecords` entry by
        centroid (nearest, tol 1e-9; both sides are means of the same float64
        vertices). Untagged boundary faces keep marker 0; tagged elements that
        match no boundary face (internal tagged surfaces, e.g. faults) are
        skipped. No-op when there is nothing to lift.
        """
        if elements is None or tags is None or mesh._boundary_faces is None:
            return mesh
        if len(tags) != len(elements):
            raise GeoBrainError(
                "boundary marker tags must pair one-to-one with elements",
                object_name="UnstructuredMesh", field="tags",
                expected=f"len(tags) == len(elements) == {len(elements)}",
                actual=len(tags),
            )
        elems = torch.as_tensor(
            np.asarray(elements).copy(), dtype=torch.long,
        )
        cent_elems = points[elems].mean(dim=1)          # (m, n_dim)
        br_cent = mesh._boundary_faces.centroid          # (nb, n_dim)
        dist = torch.cdist(cent_elems, br_cent)
        min_d, idx = dist.min(dim=1)
        hit = min_d < 1e-9
        if not bool(hit.any()):
            return mesh
        marks = torch.zeros(br_cent.shape[0], dtype=torch.long)
        hit_faces = idx[hit]
        hit_tags = torch.as_tensor(np.asarray(tags).copy(), dtype=torch.long)[hit]
        # Two tagged elements matching the SAME boundary face must agree;
        # a silent last-write-wins would hide a mesher/tagging inconsistency.
        uniq_faces, inverse = torch.unique(hit_faces, return_inverse=True)
        first_tag = torch.zeros(uniq_faces.shape[0], dtype=torch.long)
        first_tag[inverse] = hit_tags  # last write per face...
        if not bool(torch.all(first_tag[inverse] == hit_tags)):
            bad = int(uniq_faces[(first_tag[inverse] != hit_tags).nonzero()[0, 0]])
            raise GeoBrainError(
                "conflicting boundary markers for one boundary face",
                object_name="UnstructuredMesh", field="tags",
                expected="one marker value per matched boundary face",
                actual=f"face {bad} received differing markers",
            )
        marks[hit_faces] = hit_tags
        # Same-class post-construction attach (validated shape/dtype): a
        # rebuild would have to re-thread the optional simplex topology for
        # nothing.
        mesh._boundary_markers = mesh._coerce_markers(
            marks, n_expected=br_cent.shape[0], field="boundary_markers",
        )
        return mesh

    @classmethod
    def from_points(
        cls,
        points: torch.Tensor,
        *,
        cell_center: str = "centroid",
        quality_warn: bool = True,
        axes: str = "zxy",
    ) -> "UnstructuredMesh":
        """Build a mesh from a point cloud via ``scipy.spatial.Delaunay``.

        2-D points ``(n, 2)`` are triangulated and delegated to
        :meth:`from_polygons`; 3-D points ``(n, 3)`` are tetrahedralised and
        delegated to :meth:`from_simplices`. ``scipy`` is a hard GeoBrain
        dependency, so no optional-import guard is needed.

        Args:
            points: ``(n, 2)`` or ``(n, 3)`` float64 point coordinates in
                platform column order, **``(z, x)``** (2-D) or **``(z, x, y)``**
                (3-D), depth first, matching the ``(nz, nx[, ny])`` TensorMesh
                axis order. A standard ``(x, y, z)`` cloud MUST be column-permuted
                to ``(z, x, y)`` before calling, else depth is silently treated
                as x.
            cell_center: forwarded to the underlying builder. Default
                ``"centroid"``: the empirically robust choice. The textbook
                Voronoi–Delaunay orthogonality argument favours
                ``"circumcenter"``, but a consistent TPFA additionally requires
                a *self-centered* triangulation, and 3-D Delaunay output from
                realistic point clouds routinely violates that (circumcentres
                fall outside their tetrahedra): the DC3D half-space benchmark
                measured centroid ≈ 2× more accurate (rel-L2 0.084 vs 0.155).
                Pass ``"circumcenter"`` only for high-quality, near-self-centered
                Delaunay meshes where the duality argument actually applies.
            axes: coordinate-column order of the input coordinates:
                'zxy' (default; platform mesh order, z first) or
                'xyz' (world order, permuted internally via
                xyz_to_mesh_axes).
            quality_warn: when ``True`` (default), emit a ``UserWarning`` if the
                derived mesh is too harshly graded for a reliable TPFA solve,
                specifically when the per-face transmissibility geometric factor
                ``area / (dist_l + dist_r)`` spread (99.9th-percentile / median)
                exceeds :data:`_GRADATION_TSPREAD_WARN`×. This is a
                **non-destructive** diagnostic: nothing is dropped, the mesh is
                returned as built. Set ``False`` to silence it (e.g. when you
                have independently vetted the cloud).

        Sliver simplices: those with volume below ``1e-12 ×`` the median
        simplex volume; are dropped before delegating; scipy's Delaunay can
        emit near-degenerate cells on the convex hull whose circumcentres would
        trip the degenerate-geometry guard.

        Mesh quality is **garbage-in / garbage-out** here: scipy's unconstrained
        Delaunay does no quality optimisation, so a harshly-graded cloud (sharp
        jumps in point spacing, e.g. dense electrode-refinement shells dropped
        straight into a coarse base grid) yields high-aspect slivers whose
        *volume* is not tiny, so the volume sliver-filter passes them, but
        whose two-point flux carries an O(1) consistency error. The builder does
        not silently repair this (a drop filter aggressive enough to remove the
        slivers disconnects the mesh); instead ``quality_warn`` surfaces it.
        Supply a quasi-uniform or gently-densified cloud (smooth spacing
        gradients) for a trustworthy solve.

        Raises:
            GeoBrainError: on non-2-D/3-D points, or whatever the delegated
                builder raises.
        """
        from scipy.spatial import Delaunay, QhullError

        points = resolve_axes_kwarg(points, axes,
                                    "UnstructuredMesh.from_points")
        pts = points.to(torch.float64)
        if pts.dim() != 2 or int(pts.shape[1]) not in (2, 3):
            raise GeoBrainError(
                "from_points needs (n, 2) or (n, 3) points",
                object_name="UnstructuredMesh.from_points", field="points",
                expected="(n, 2|3)", actual=tuple(pts.shape),
            )
        n_dim = int(pts.shape[1])
        min_points = n_dim + 1
        if int(pts.shape[0]) < min_points:
            raise GeoBrainError(
                "from_points needs enough points for a Delaunay simplex",
                object_name="UnstructuredMesh.from_points", field="points",
                expected=f">= {min_points} points", actual=int(pts.shape[0]),
            )
        pts_np = pts.detach().cpu().numpy()
        try:
            tri = Delaunay(pts_np)
        except (QhullError, ValueError) as exc:
            raise GeoBrainError(
                "scipy Delaunay triangulation failed; points may be degenerate "
                "(duplicate, collinear/coplanar, or otherwise not full-dimensional)",
                object_name="UnstructuredMesh.from_points", field="points",
                expected="full-dimensional non-degenerate point cloud",
                actual=type(exc).__name__,
            ) from exc
        simplices = np.asarray(tri.simplices, dtype=np.int64)  # (m, n_dim+1)

        # Drop slivers: volume < 1e-12 × median volume.
        p = pts_np[simplices]  # (m, n_dim+1, n_dim)
        if n_dim == 2:
            e1 = p[:, 1] - p[:, 0]
            e2 = p[:, 2] - p[:, 0]
            vol = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2.0
        else:
            e1 = p[:, 1] - p[:, 0]
            e2 = p[:, 2] - p[:, 0]
            e3 = p[:, 3] - p[:, 0]
            vol = np.abs(np.einsum("ij,ij->i", np.cross(e1, e2), e3)) / 6.0
        med = float(np.median(vol))
        if med <= 0.0 or not np.isfinite(med):
            raise GeoBrainError(
                "Delaunay triangulation produced degenerate simplices",
                object_name="UnstructuredMesh.from_points", field="points",
                expected="positive finite simplex volumes", actual=med,
            )
        keep = vol >= 1e-12 * med
        simplices = simplices[keep]
        if int(simplices.shape[0]) == 0:
            raise GeoBrainError(
                "Delaunay triangulation produced only sliver simplices",
                object_name="UnstructuredMesh.from_points", field="points",
                expected="at least one retained simplex", actual=0,
            )

        verts = torch.from_numpy(pts_np)
        if n_dim == 2:
            cells = [list(int(v) for v in row) for row in simplices]
            mesh = cls.from_polygons(verts, cells, cell_center=cell_center)
        else:
            mesh = cls.from_simplices(
                verts, torch.from_numpy(simplices), cell_center=cell_center,
            )

        if quality_warn:
            mesh._warn_if_harshly_graded()
        return mesh

    def quality_report(self) -> dict:
        """Physics-relevant mesh-quality metrics.

        The companion of :meth:`from_meshio`: assess an ingested mesh BEFORE
        running physics on it. Generic metrics come from the face SoA and
        apply to ANY instance; 3-D simplex meshes (node tables present) add
        shape metrics. All values are plain floats/ints (JSON-safe).

        Returns keys:
            ``n_cells``, ``n_internal_faces``, ``n_boundary_faces``;
            ``volume``: ``{min, max, max_over_min}``;
            ``tpfa_orthogonality``: ``{min, mean}`` of ``|n̂_f · d̂_f|`` per
            internal face (``d_f`` = neighbour-centre line; 1.0 = the
            two-point flux assumption is exact, low values flag skewed
            cells the TPFA assembler will mis-couple);
            ``t_geom_spread``: 99.9th-percentile / median of the assembler's
            own geometric coupling ``area / (dist_l + dist_r)`` (the
            :meth:`from_points` warning signal, exposed);
            and, on 3-D simplex meshes only:
            ``dihedral_deg``: ``{min, mean}`` over every tet's 6 dihedrals;
            ``radius_ratio``: ``{min, mean}`` of ``3·r_in / R_circ``
            (1.0 = regular tetrahedron, → 0 for slivers).
        """
        fr = self._faces
        rep: dict = {
            "n_cells": self.n_cells,
            "n_internal_faces": int(fr.area.numel()),
            "n_boundary_faces": int(self._boundary_faces.area.numel())
            if self._boundary_faces is not None else 0,
        }
        vols = self._cell_volumes
        vmin, vmax = float(vols.min()), float(vols.max())
        rep["volume"] = {"min": vmin, "max": vmax,
                        "max_over_min": vmax / max(vmin, 1e-300)}

        if fr.area.numel():
            centers = self._cell_centers
            d = centers[fr.cell_j] - centers[fr.cell_i]
            d_hat = d / d.norm(dim=1, keepdim=True).clamp(min=1e-300)
            n_hat = fr.normal / fr.normal.norm(dim=1, keepdim=True).clamp(min=1e-300)
            ortho = (n_hat * d_hat).sum(dim=1).abs()
            rep["tpfa_orthogonality"] = {"min": float(ortho.min()),
                                         "mean": float(ortho.mean())}
            t_geom = fr.area / (fr.dist_l + fr.dist_r).clamp(min=1e-300)
            med = float(t_geom.median())
            p999 = float(torch.quantile(t_geom, 0.999))
            rep["t_geom_spread"] = p999 / max(med, 1e-300)

        if self._node_coords is not None and self.n_dim == 3:
            v = self._node_coords[self._cell_nodes]          # (n, 4, 3)
            # --- outward face normals (face k = opposite vertex k) ---------
            idx = torch.tensor([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
            a = v[:, idx[:, 0]]
            b = v[:, idx[:, 1]]
            c = v[:, idx[:, 2]]
            n = torch.cross(b - a, c - a, dim=2)             # (n, 4, 3)
            # orient away from the opposite vertex
            opp = v                                          # vertex k
            flip = ((n * (opp - a)).sum(dim=2) > 0)
            n = torch.where(flip.unsqueeze(-1), -n, n)
            areas = 0.5 * n.norm(dim=2)                      # (n, 4)
            n_hat = n / n.norm(dim=2, keepdim=True).clamp(min=1e-300)
            # --- dihedrals: faces i, j share an edge; θ = π − angle(n_i, n_j)
            pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
            cosd = -(n_hat[:, pairs[:, 0]] * n_hat[:, pairs[:, 1]]).sum(dim=2)
            dihed = torch.rad2deg(torch.arccos(cosd.clamp(-1.0, 1.0)))
            rep["dihedral_deg"] = {"min": float(dihed.min()),
                                   "mean": float(dihed.mean())}
            # --- radius ratio: r_in = 3V / ΣA;  R_circ from the 3×3 solve ---
            r_in = 3.0 * vols / areas.sum(dim=1).clamp(min=1e-300)
            rhs = 0.5 * ((v[:, 1:] ** 2).sum(dim=2) - (v[:, :1] ** 2).sum(dim=2))
            A = v[:, 1:] - v[:, :1]                          # (n, 3, 3)
            centre = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
            R = (centre - v[:, 0]).norm(dim=1)
            ratio = 3.0 * r_in / R.clamp(min=1e-300)
            rep["radius_ratio"] = {"min": float(ratio.min()),
                                   "mean": float(ratio.mean())}
        return rep

    def _warn_if_harshly_graded(self) -> None:
        """Emit a ``UserWarning`` when the mesh is too harshly graded for TPFA.

        Non-destructive quality probe used by :meth:`from_points`. The signal is
        the spread (99.9th-percentile / median) of the per-face transmissibility
        geometric factor ``T_geom = area / (dist_l + dist_r)``, the σ=1
        two-point flux coupling the FV assembler stamps on each internal face.
        High-aspect slivers from a harshly-graded cloud produce extreme
        ``T_geom`` outliers even after the volume sliver-filter; a spread above
        :data:`_GRADATION_TSPREAD_WARN`× flags the cloud as unreliable. The
        99.9th percentile (rather than the max) is used so a lone non-self-
        centred circumcentre on an otherwise-healthy mesh does not false-trip
        it. The message is actionable: it names the cause (harsh spacing
        gradient) and the remedy (quasi-uniform / gently-densified cloud).
        """
        import warnings

        fr = self._faces
        n_faces = int(fr.area.numel())
        if n_faces < self._GRADATION_MIN_FACES:
            return
        denom = fr.dist_l + fr.dist_r
        t_geom = fr.area / denom
        med = float(torch.median(t_geom))
        if med <= 0.0:
            return
        spread = float(torch.quantile(t_geom, 0.999)) / med
        if spread > self._GRADATION_TSPREAD_WARN:
            warnings.warn(
                "UnstructuredMesh.from_points: the point cloud is harshly "
                "graded, the per-face transmissibility geometric factor "
                f"(area / (dist_l + dist_r)) spreads {spread:.0f}× "
                f"(99.9th pct / median), above the "
                f"{self._GRADATION_TSPREAD_WARN:.0f}× advisory threshold. "
                "scipy's unconstrained Delaunay turns sharp "
                "spacing jumps into high-aspect slivers whose two-point flux "
                "carries an O(1) error, so a TPFA solve (DC3D, flow, ...) on "
                "this mesh may be unreliable. Supply a quasi-uniform or gently "
                "densified cloud (smooth spacing gradients), or vet the mesh and "
                "pass quality_warn=False to silence this.",
                UserWarning,
                stacklevel=3,
            )
