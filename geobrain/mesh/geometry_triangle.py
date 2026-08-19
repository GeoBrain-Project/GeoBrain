"""Triangle backend: PLC → quality triangle mesh with markers.

Turns a :class:`~geobrain.mesh.geometry.PLC2D` into an
:class:`~geobrain.mesh.unstructured.UnstructuredMesh` via Shewchuk's
Triangle, carrying region
seeds → ``cell_markers()`` and segment markers → ``boundary_markers()``.

The ``triangle`` package is an OPTIONAL dependency (same policy as meshio).
Where it cannot be installed, run Triangle in a separate environment
against :mod:`geometry`'s numpy arrays (the test suite carries a
fixture-generator script for exactly this pattern) and convert the saved
output here with :func:`mesh_from_triangle_output`; the conversion path is
torch-only and fully testable without Triangle.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.errors import GeoBrainError
from .geometry import PLC2D
from .unstructured import UnstructuredMesh

__all__ = ["triangulate", "mesh_from_triangle_output"]


def _opts(quality: float | None, max_area: float | None,
          has_regions: bool, extra: str) -> str:
    o = "p"
    if quality is not None:
        o += f"q{float(quality):g}"
    if has_regions:
        o += "A"
    if max_area is not None:
        o += f"a{float(max_area):.17g}"
    elif has_regions:
        o += "a"  # activate per-region area ceilings
    return o + extra


def triangulate(
    plc: PLC2D,
    *,
    quality: float | None = 30.0,
    max_area: float | None = None,
    extra_opts: str = "",
) -> UnstructuredMesh:
    """Quality-mesh a PLC (Triangle ``p q<quality> A a`` switches).

    Args:
        plc: the geometry (with region/hole seeds already added).
        quality: minimum-angle constraint in degrees (``None`` disables).
        max_area: global triangle-area ceiling; per-region ceilings from
            ``add_region(..., max_area=...)`` apply when this is ``None``.
        extra_opts: appended verbatim to the Triangle switch string.

    Raises:
        GeoBrainError: when the optional ``triangle`` package is missing.
    """
    try:
        import triangle as _triangle
    except ImportError as exc:
        raise GeoBrainError(
            "triangulate requires the optional 'triangle' package "
            "(pip install triangle). If it cannot be installed in this "
            "environment, run Triangle in a separate environment and "
            "convert its saved output with mesh_from_triangle_output",
            object_name="geometry_triangle.triangulate",
            field="triangle",
            expected="triangle importable",
            actual=None,
        ) from exc

    arrays = plc.to_arrays()
    out = _triangle.triangulate(
        arrays,
        _opts(quality, max_area, "regions" in arrays, extra_opts),
    )
    return mesh_from_triangle_output(out)


def mesh_from_triangle_output(out: dict) -> UnstructuredMesh:
    """Convert a Triangle output dict into a marked UnstructuredMesh.

    Expects the keys ``triangle.triangulate`` produces: ``vertices (n, 2)``
    in the PLC's (x, z) section frame, ``triangles (m, 3)``, and optionally
    ``triangle_attributes`` (regional attributes → ``cell_markers``, rounded
    to long) and ``segments``/``segment_markers`` (constrained segments →
    ``boundary_markers``, centroid-matched; internal constraint segments
    match no boundary face and are skipped).
    """
    verts_xz = np.asarray(out["vertices"], dtype=np.float64)
    tris = np.asarray(out["triangles"], dtype=np.int64)
    if verts_xz.ndim != 2 or verts_xz.shape[1] != 2:
        raise GeoBrainError(
            "triangle output vertices must be (n, 2)",
            object_name="mesh_from_triangle_output",
            field="vertices",
            expected="(n, 2)",
            actual=tuple(verts_xz.shape),
        )
    # PLC section frame (x, z) → platform mesh axes (z, x)
    verts_mesh = torch.from_numpy(
        np.ascontiguousarray(verts_xz[:, ::-1])
    )

    cell_markers = None
    attrs = out.get("triangle_attributes")
    if attrs is not None and np.asarray(attrs).size:
        cell_markers = torch.as_tensor(
            np.rint(np.asarray(attrs, dtype=np.float64)).reshape(-1),
        ).to(torch.long)

    mesh = UnstructuredMesh.from_polygons(
        verts_mesh,
        [list(map(int, row)) for row in tris],
        cell_markers=cell_markers,
    )
    segments = out.get("segments")
    markers = out.get("segment_markers")
    if segments is not None and markers is not None:
        mesh = UnstructuredMesh._lift_boundary_markers(
            mesh, verts_mesh,
            np.asarray(segments, dtype=np.int64),
            np.asarray(markers, dtype=np.int64).reshape(-1),
        )
    return mesh
