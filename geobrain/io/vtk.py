# pyright: reportPrivateImportUsage=false
"""
VTK / pyvista export for ParaView-ready 3-D field visualization.

For :class:`~geobrain.mesh.TensorMesh`-based scalar / vector cell-data
in 3D, writes a RectilinearGrid (``.vtr``) file that any VTK-compatible
viewer reads.

Public:
    mesh_to_pyvista          TensorMesh + cell_data → pv.RectilinearGrid
    write_tensormesh_vtk      one-call writer (``.vtr`` by default)
    write_points_vtk          point-cloud writer (stations / receivers)
    write_octree_points_vtk   OctreeMesh cell-center cloud writer
    write_unstructured_vtk    3-D simplex UnstructuredMesh writer (meshio)

Node coordinate arrays come from the mesh's :meth:`TensorMesh.node_lines`
(origin-aware, exact on graded meshes). The 3-D cell-data shape convention
is ``(nz, nx, ny)``, the platform-wide axis order (depth slowest, y
fastest), while VTK expects x-fastest flattening, so
``transpose(1, 2, 0).ravel(order="F")`` reorders to match.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from geobrain.core.errors import GeoBrainError

from collections.abc import Mapping
from typing import Any

import numpy as np

from ._common import _to_numpy

try:
    import pyvista as pv
    _HAS_PV = True
except ImportError:  # pragma: no cover
    _HAS_PV = False
    pv = None  # type: ignore[assignment]


__all__ = [
    "mesh_to_pyvista",
    "write_tensormesh_vtk",
    "write_points_vtk",
    "write_octree_points_vtk",
    "write_unstructured_vtk",
]


def _require_pv() -> None:
    if not _HAS_PV:
        raise ImportError(
            "pyvista is required for VTK export. "
            "Install with: pip install pyvista"
        )


# ---------------------------------------------------------------------------
# TensorMesh ↔ RectilinearGrid
# ---------------------------------------------------------------------------


def _tensormesh_node_arrays(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract ``(x_nodes, y_nodes, z_nodes)`` 1-D arrays from a 3-D TensorMesh.

    The :class:`TensorMesh` stores ``shape = (nz, nx, ny)`` in 3D (platform
    axis order) and per-axis ``cell_widths`` in the same order. Node arrays
    come from :meth:`TensorMesh.node_lines`, origin-aware and exact on
    graded meshes.
    """
    shape = tuple(mesh.shape)
    if len(shape) != 3:
        raise GeoBrainError(
            f"VTK export expects a 3-D TensorMesh; got shape {shape}."
        )
    nl_z, nl_x, nl_y = mesh.node_lines()   # mesh axis order (z, x, y)
    x_nodes = _to_numpy(nl_x, dtype=np.float64)
    y_nodes = _to_numpy(nl_y, dtype=np.float64)
    z_nodes = _to_numpy(nl_z, dtype=np.float64)
    return x_nodes, y_nodes, z_nodes


def mesh_to_pyvista(
    mesh,
    cell_data: Mapping[str, Any] | None = None,
):
    """
    Convert a 3-D TensorMesh + cell-data dict to a ``pv.RectilinearGrid``.

    Args:
        mesh: A 3-D :class:`geobrain.mesh.TensorMesh`.
        cell_data: Per-cell scalar (or 3-vector) fields keyed by name.
            Each value must have shape ``(nz, nx, ny)`` (scalars, platform
            axis order) or ``(nz, nx, ny, 3)`` (vectors).

    Returns:
        :class:`pyvista.RectilinearGrid`.

    Raises:
        GeoBrainError: if any ``cell_data`` entry has an unrecognized
            shape, or if the mesh is not 3-D.
        ImportError: if pyvista is not installed.
    """
    _require_pv()
    x_nodes, y_nodes, z_nodes = _tensormesh_node_arrays(mesh)
    grid = pv.RectilinearGrid(x_nodes, y_nodes, z_nodes)

    if cell_data:
        mesh_shape = tuple(getattr(mesh, "shape", ()))  # (nz, nx, ny)
        for name, val in cell_data.items():
            arr = _to_numpy(val)
            if arr.ndim == 3:
                if mesh_shape and tuple(arr.shape) != mesh_shape:
                    raise GeoBrainError(
                        f"cell_data['{name}'] shape {tuple(arr.shape)} "
                        f"does not match mesh.shape {mesh_shape} "
                        f"(expected scalar field (nz, nx, ny))."
                    )
                # (nz, nx, ny) layout → VTK x-fastest order.
                grid.cell_data[name] = arr.transpose(1, 2, 0).ravel(
                    order="F"
                )
            elif arr.ndim == 4 and arr.shape[-1] == 3:
                if mesh_shape and tuple(arr.shape[:3]) != mesh_shape:
                    raise GeoBrainError(
                        f"cell_data['{name}'] vector-field spatial "
                        f"shape {tuple(arr.shape[:3])} does not match "
                        f"mesh.shape {mesh_shape} (expected "
                        f"(nz, nx, ny, 3))."
                    )
                grid.cell_data[name] = arr.transpose(1, 2, 0, 3).reshape(
                    -1, 3, order="F"
                )
            else:
                raise GeoBrainError(
                    f"cell_data['{name}'] has shape {arr.shape}; "
                    f"expected (nz, nx, ny) or (nz, nx, ny, 3)."
                )
    return grid


def write_tensormesh_vtk(
    path: str,
    mesh,
    cell_data: Mapping[str, Any] | None = None,
) -> None:
    """
    One-call writer: TensorMesh + cell data → ``.vtr``.

    The file format is inferred from the extension; ``.vtr``
    (rectilinear) is the natural choice for variable-cell TensorMeshes.

    Args:
        path: output file; extension selects the VTK format (``.vtr``).
        mesh: 3-D :class:`~geobrain.mesh.TensorMesh`.
        cell_data: ``{name: (nz, nx, ny) array}`` scalar fields (platform
            axis order; vectors as ``(nz, nx, ny, 3)``).
    """
    grid = mesh_to_pyvista(mesh, cell_data=cell_data)
    grid.save(path)


# ---------------------------------------------------------------------------
# Point clouds (stations, receivers, source positions)
# ---------------------------------------------------------------------------


def write_points_vtk(
    path: str,
    points: Any,
    point_data: Mapping[str, Any] | None = None,
) -> None:
    """
    Write a point cloud as ``.vtp``.

    Args:
        path: Output path; ``.vtp`` extension recommended.
        points: ``(n, 3)`` array-like of XYZ coordinates.
        point_data: Per-point scalar (``(n,)``) or vector (``(n, 3)``)
            fields.

    Raises:
        GeoBrainError: if ``points`` does not have shape ``(n, 3)``.
    """
    _require_pv()
    pts = _to_numpy(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise GeoBrainError(
            f"points must have shape (n, 3); got {pts.shape}."
        )
    n = int(pts.shape[0])
    poly = pv.PolyData(pts)
    if point_data:
        for k, v in point_data.items():
            arr = _to_numpy(v)
            if arr.ndim == 1 and int(arr.shape[0]) == n:
                poly.point_data[k] = arr
            elif arr.ndim == 2 and arr.shape == (n, 3):
                poly.point_data[k] = arr
            else:
                raise GeoBrainError(
                    f"point_data['{k}'] has shape {arr.shape}; "
                    f"expected ({n},) for a scalar field or "
                    f"({n}, 3) for a vector field."
                )
    poly.save(path)


# ---------------------------------------------------------------------------
# OctreeMesh point-cloud export
# ---------------------------------------------------------------------------


def write_octree_points_vtk(
    path: str,
    octree_mesh,
    cell_data: Mapping[str, Any] | None = None,
) -> None:
    """
    Export an OctreeMesh's cell centers as a point cloud (``.vtp``).

    Writes a point cloud only, not an UnstructuredGrid of hex cells.
    Use ParaView's "Glyph" filter to render each point as a box sized
    to the cell if you need a hex-like appearance.

    Raises:
        GeoBrainError: if ``octree_mesh.cell_centers`` does not look like
            an ``(n_cells, 3)`` array.

    Args:
        path: output ``.vtp`` file.
        octree_mesh: :class:`~geobrain.mesh.OctreeMesh` whose leaf centres
            are exported as a point cloud.
        cell_data: optional ``{name: (n_cells,) array}`` leaf values.
    """
    _require_pv()
    centers_attr = getattr(octree_mesh, "cell_centers", None)
    # ``OctreeMesh.cell_centers`` is a method (returns a tensor);
    # accept either a callable or an attribute for robustness.
    if callable(centers_attr):
        centers = centers_attr()
    else:
        centers = centers_attr
    centres = _to_numpy(centers) if centers is not None else None
    if centres is None or centres.ndim != 2 or centres.shape[1] != 3:
        raise GeoBrainError(
            "octree_mesh must expose cell_centers as an (n_cells, 3) "
            "array."
        )
    # OctreeMesh.cell_centers() columns follow the platform mesh-axis
    # order (z, x, y); write_points_vtk expects physical XYZ. Reorder the
    # columns to (x, y, z), mirroring Scene3D._octree_corners.
    centres = np.ascontiguousarray(centres[:, [1, 2, 0]])
    write_points_vtk(path, centres, point_data=cell_data)


def write_unstructured_vtk(
    path,
    mesh,
    cell_data: Mapping[str, Any] | None = None,
):
    """Write a 3-D simplex :class:`UnstructuredMesh` (+ per-cell data) to ``.vtu``.

    The ParaView door for tet meshes (the 3-D EM inversion workflow).
    Backed by ``meshio`` (optional
    extra), no pyvista required. Node coordinates are emitted in WORLD
    ``(x, y, z)`` order (``mesh_axes_to_xyz`` of the stored mesh-order
    nodes), matching every other writer in this module; the export is the
    exact inverse of ``UnstructuredMesh.from_meshio``'s default ``axes="xyz"``
    ingestion, so write → read round-trips.

    Args:
        mesh: An :class:`~geobrain.mesh.unstructured.UnstructuredMesh`
            WITH simplex topology (built by ``from_simplices`` /
            ``from_meshio`` tetra, or the SoA constructor fed
            ``node_coords=``/``cell_nodes=``). Meshes without node tables
            (2-D ``from_polygons``, SoA-only) cannot be expressed as a VTK
            unstructured grid, export their cell centres with
            :func:`write_points_vtk` instead.
        path: Output path (``.vtu`` recommended; any meshio-writable
            unstructured format works).
        cell_data: Optional ``{name: (n_cells,)}`` per-cell scalars.

    Raises:
        GeoBrainError: If meshio is missing, the mesh has no simplex
            topology, or a ``cell_data`` entry has the wrong length.
    """
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - env without meshio
        raise GeoBrainError(
            "write_unstructured_vtk requires the optional 'meshio' package, "
            "install with `pip install meshio`"
        ) from exc

    from ..mesh import mesh_axes_to_xyz

    try:
        nodes = mesh.node_coords()
        cells = mesh.cell_nodes()
    except Exception as exc:
        raise GeoBrainError(
            "write_unstructured_vtk needs a 3-D simplex mesh with node "
            "tables (from_simplices / from_meshio tetra); for meshes without "
            "them export the cell centres with write_points_vtk instead"
        ) from exc

    n_cells = int(cells.shape[0])
    data = {}
    if cell_data:
        for name, val in cell_data.items():
            arr = _to_numpy(val).reshape(-1)
            if arr.shape[0] != n_cells:
                raise GeoBrainError(
                    f"cell_data['{name}'] must have one value per cell "
                    f"({n_cells}); got shape {arr.shape}"
                )
            data[name] = [arr]

    mm = meshio.Mesh(
        points=mesh_axes_to_xyz(nodes).cpu().numpy(),
        cells=[("tetra", cells.cpu().numpy())],
        cell_data=data or None,
    )
    meshio.write(path, mm)
