"""PLC geometry builder: the polytools-style construction layer of the platform.

A :class:`PLC2D` collects geometric pieces (outer world box, polygons,
circles, free segments), *region seeds* (an interior point + integer region
marker + optional per-region max triangle area) and *hole seeds*, and emits
the piecewise-linear-complex arrays a quality mesher consumes. The Triangle
backend (:mod:`geobrain.mesh.geometry_triangle`) turns it into an
:class:`~geobrain.mesh.unstructured.UnstructuredMesh` whose
``cell_markers()`` / ``boundary_markers()`` carry the region/boundary labels
through meshing, geometry-level intent survives into the solve, so
consumers map ``marker → parameter`` without geometric tests.

Coordinates are the platform's 2-D **section frame** ``(x, z)`` with z
positive DOWN (the Triangle plane is orientation-agnostic; the backend
permutes columns to mesh axes ``(z, x)`` on conversion).

Deliberately numpy-only (no torch import): the module is loadable standalone
(``importlib`` by file path) in a bare numpy environment, so Triangle can be
run in a separate environment and its output consumed here as committed
fixtures.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["PLC2D"]

_QUANT = 9  # vertex dedup rounding (decimals), shared piece corners merge


class PLC2D:
    """Piecewise-linear complex in the (x, z-down) section frame.

    Pieces append vertices (deduplicated exactly at 1e-9) and marked
    segments; :meth:`to_arrays` emits the Triangle-format dict.
    """

    def __init__(self) -> None:
        self._verts: list[tuple[float, float]] = []
        self._vert_lut: dict[tuple[float, float], int] = {}
        self._segments: list[tuple[int, int]] = []
        self._segment_markers: list[int] = []
        self._regions: list[tuple[float, float, int, float]] = []
        self._holes: list[tuple[float, float]] = []

    # ------------------------------------------------------------- vertices

    def _vertex(self, x: float, z: float) -> int:
        key = (round(float(x), _QUANT), round(float(z), _QUANT))
        idx = self._vert_lut.get(key)
        if idx is None:
            idx = len(self._verts)
            self._vert_lut[key] = idx
            self._verts.append((float(x), float(z)))
        return idx

    def _chain(self, points, *, closed: bool, marker: int) -> "PLC2D":
        ids = [self._vertex(px, pz) for px, pz in points]
        pairs = list(zip(ids[:-1], ids[1:]))
        if closed and len(ids) > 2:
            pairs.append((ids[-1], ids[0]))
        for a, b in pairs:
            if a == b:
                raise ValueError(
                    "PLC2D: degenerate zero-length segment (duplicate "
                    f"consecutive point near vertex {a})"
                )
            self._segments.append((a, b))
            self._segment_markers.append(int(marker))
        return self

    # --------------------------------------------------------------- pieces

    def add_polygon(self, points, *, boundary_marker: int = 0,
                    closed: bool = True) -> "PLC2D":
        """Add a vertex chain ``[(x, z), ...]`` (closed loop by default)."""
        if len(points) < (3 if closed else 2):
            raise ValueError("PLC2D.add_polygon needs >= 3 points "
                             "(2 for an open chain)")
        return self._chain(points, closed=closed, marker=boundary_marker)

    def add_rectangle(self, x: tuple[float, float], z: tuple[float, float],
                      *, boundary_marker: int = 0) -> "PLC2D":
        """Axis-aligned box ``x=(x0, x1)``, ``z=(z0, z1)`` (z down)."""
        x0, x1 = sorted(map(float, x))
        z0, z1 = sorted(map(float, z))
        if x0 == x1 or z0 == z1:
            raise ValueError("PLC2D.add_rectangle: zero-size box")
        return self._chain(
            [(x0, z0), (x1, z0), (x1, z1), (x0, z1)],
            closed=True, marker=boundary_marker,
        )

    def add_circle(self, center: tuple[float, float], radius: float, *,
                   n_segments: int = 32,
                   boundary_marker: int = 0) -> "PLC2D":
        """Regular-polygon approximation of a circle."""
        if radius <= 0 or n_segments < 3:
            raise ValueError("PLC2D.add_circle: need radius > 0 and "
                             "n_segments >= 3")
        cx, cz = float(center[0]), float(center[1])
        pts = [
            (cx + radius * math.cos(2 * math.pi * k / n_segments),
             cz + radius * math.sin(2 * math.pi * k / n_segments))
            for k in range(n_segments)
        ]
        return self._chain(pts, closed=True, marker=boundary_marker)

    def add_line(self, p0: tuple[float, float], p1: tuple[float, float], *,
                 boundary_marker: int = 0) -> "PLC2D":
        """Free internal constraint segment (e.g. a fault trace)."""
        return self._chain([p0, p1], closed=False, marker=boundary_marker)

    # ------------------------------------------------------- seeds / holes

    def add_region(self, point: tuple[float, float], *, marker: int,
                   max_area: float | None = None) -> "PLC2D":
        """Region seed: interior point + region marker (+ area ceiling)."""
        self._regions.append(
            (float(point[0]), float(point[1]), int(marker),
             float(max_area) if max_area is not None else 0.0)
        )
        return self

    def add_hole(self, point: tuple[float, float]) -> "PLC2D":
        """Hole seed: the enclosing closed piece is left unmeshed."""
        self._holes.append((float(point[0]), float(point[1])))
        return self

    # --------------------------------------------------------------- export

    @property
    def n_vertices(self) -> int:
        return len(self._verts)

    @property
    def n_segments(self) -> int:
        return len(self._segments)

    def to_arrays(self) -> dict:
        """Triangle-format PSLG dict (float64 / int32 numpy arrays).

        Keys: ``vertices (n,2)`` in (x, z), ``segments (m,2)``,
        ``segment_markers (m,)``; plus ``regions (r,4)`` rows
        ``[x, z, marker, max_area]`` and ``holes (h,2)`` when present,
        exactly what ``triangle.triangulate`` consumes.
        """
        if len(self._verts) < 3 or not self._segments:
            raise ValueError("PLC2D.to_arrays: geometry is empty; add an "
                             "outer boundary piece first")
        out = {
            "vertices": np.asarray(self._verts, dtype=np.float64),
            "segments": np.asarray(self._segments, dtype=np.int32),
            "segment_markers": np.asarray(self._segment_markers,
                                          dtype=np.int32),
        }
        if self._regions:
            out["regions"] = np.asarray(self._regions, dtype=np.float64)
        if self._holes:
            out["holes"] = np.asarray(self._holes, dtype=np.float64)
        return out
