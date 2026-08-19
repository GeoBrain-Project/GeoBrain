"""Axisymmetric cylindrical mesh (n_theta = 1): records-only surface.

The axisymmetric ring mesh in the platform's frame: coordinates ``(z, r)`` with z positive DOWN (axis 0,
core convention) and r from the symmetry axis (axis 1, r₀ = 0 always).
Every cell is a full ring, and the RECORDS carry the true ring geometry,
volumes ``π(r₂²−r₁²)Δz``, radial face areas ``2πrΔz``, vertical annulus
areas ``π(r₂²−r₁²)``, so every records consumer (the operator factory,
records TPFA assemblers, projections) is correct on cylinders unchanged.

Declares GeometryMesh + ConnectivityMesh + CylindricalGeometryMesh, and
deliberately NOT ``StructuredMesh``/``UniformMesh``: a cylinder must never
satisfy a Cartesian-stencil gate (FDTD, DC2D) even though it carries the
same structured accessor names (``shape`` / ``cell_widths`` /
``center_lines`` / ``node_lines``, the surface CylindricalGeometryMesh
names for staggered consumers such as the axisymmetric FDEM E_φ operator).
Cross-mesh projection routes cyl↔cyl through the exact rectilinear kernel
with ring-exact conservative coarsening; cyl↔Cartesian pairs are rejected
(see the projection package).

The symmetry axis produces NO boundary faces: the r = 0 face of the inner
ring has zero area (no flux crosses the axis), so emitting it would only
plant zero-area records in every boundary consumer.

Cell ordering is C-order over ``(z, r)`` (r fastest) matching the
flattening every structured mesh in the platform uses.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
import numbers
from typing import Sequence

import torch

from ..core.errors import GeoBrainError
from .base import Mesh
from .capabilities import (
    BoundaryRecords,
    ConnectivityMesh,
    CylindricalGeometryMesh,
    FaceRecords,
    GeometryMesh,
)

__all__ = ["CylindricalMesh"]

_PI = math.pi


class CylindricalMesh(Mesh):
    """Axisymmetric ring mesh: shape ``(nz, nr)``, coords ``(z, r)``."""

    mesh_capabilities: frozenset[type] = frozenset(
        {GeometryMesh, ConnectivityMesh, CylindricalGeometryMesh}
    )

    def __init__(
        self,
        shape: Sequence[int],
        spacing: Sequence[float] | None = None,
        *,
        cell_widths: Sequence[torch.Tensor] | None = None,
        origin: Sequence[float] | None = None,
    ) -> None:
        """
        Args:
            shape: ``(nz, nr)`` ring counts (z first, core convention).
            spacing: constant ``(dz, dr)``. Mutually exclusive with
                ``cell_widths``.
            cell_widths: per-axis 1-D width tensors ``(wz, wr)``.
            origin: ``(z0, r0)``; ``r0`` MUST be 0 (the mesh always starts
                at the symmetry axis). ``None`` → ``(0, 0)``.
        """
        if (spacing is None) == (cell_widths is None):
            raise GeoBrainError(
                "pass exactly one of spacing or cell_widths",
                object_name="CylindricalMesh", field="spacing/cell_widths",
                expected="exactly one", actual="both" if spacing else "neither",
            )
        if len(shape) != 2:
            raise GeoBrainError(
                "CylindricalMesh shape must be (nz, nr)",
                object_name="CylindricalMesh", field="shape",
                expected="length 2", actual=tuple(shape),
            )
        for axis, n in enumerate(shape):
            if isinstance(n, bool) or not isinstance(n, numbers.Integral) or int(n) <= 0:
                raise GeoBrainError(
                    "CylindricalMesh cell count must be a positive int",
                    object_name="CylindricalMesh", field=f"shape[{axis}]",
                    expected="positive int", actual=n,
                )
        nz, nr = int(shape[0]), int(shape[1])

        if cell_widths is not None:
            if len(cell_widths) != 2:
                raise GeoBrainError(
                    "cell_widths must be (wz, wr)",
                    object_name="CylindricalMesh", field="cell_widths",
                    expected="length 2", actual=len(cell_widths),
                )
            wz = torch.as_tensor(cell_widths[0], dtype=torch.float64)
            wr = torch.as_tensor(cell_widths[1], dtype=torch.float64)
        else:
            wz = torch.full((nz,), float(spacing[0]), dtype=torch.float64)
            wr = torch.full((nr,), float(spacing[1]), dtype=torch.float64)
        for name, w, n in (("wz", wz, nz), ("wr", wr, nr)):
            if w.shape != (n,) or not bool(torch.isfinite(w).all()) \
                    or bool((w <= 0).any()):
                raise GeoBrainError(
                    f"{name} must be ({n},) of positive finite widths",
                    object_name="CylindricalMesh", field=name,
                    expected=(n,), actual=tuple(w.shape),
                )

        if origin is None:
            origin = (0.0, 0.0)
        if len(origin) != 2 or float(origin[1]) != 0.0:
            raise GeoBrainError(
                "origin must be (z0, 0.0), the r axis always starts at the "
                "symmetry axis",
                object_name="CylindricalMesh", field="origin",
                expected="(z0, 0.0)", actual=tuple(origin),
            )

        self._n_dim = 2
        self._n_cells = nz * nr
        self._shape = (nz, nr)
        self._wz = wz.detach().clone().cpu()
        self._wr = wr.detach().clone().cpu()
        self._z0 = float(origin[0])
        # node lines (exact cumulative sums, origin-shifted for z)
        self._zn = torch.cat(
            [torch.zeros(1, dtype=torch.float64), torch.cumsum(self._wz, 0)]
        ) + self._z0
        self._rn = torch.cat(
            [torch.zeros(1, dtype=torch.float64), torch.cumsum(self._wr, 0)]
        )
        self._faces: FaceRecords | None = None
        self._boundary: BoundaryRecords | None = None

    # ------------------------------------------------------------ structured
    # accessors for staggered consumers (NOT a StructuredMesh declaration;
    # capability membership is by declaration, so Cartesian gates still reject)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def cell_widths(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``(wz, wr)`` per-axis width vectors (fresh clones).

        Part of the :class:`CylindricalGeometryMesh` accessor set, a
        deliberately 2-tuple shape distinct from TensorMesh's per-axis tuple
        property and OctreeMesh's per-cell method (see the OctreeMesh
        ``cell_widths`` docstring for the three-way note).
        """
        return self._wz.clone(), self._wr.clone()

    @property
    def origin(self) -> tuple[float, float]:
        return (self._z0, 0.0)

    def center_lines(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-axis centre coordinates: z centres, ring MIDPOINT radii.

        The ring centre is the radial midpoint ``(r₁+r₂)/2`` (the standard
        convention), not the volume centroid, the TPFA half-distances below
        use the same rule, keeping centre-to-face distances consistent.
        """
        zc = self._zn[:-1] + 0.5 * self._wz
        rc = self._rn[:-1] + 0.5 * self._wr
        return zc, rc

    def node_lines(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._zn.clone(), self._rn.clone()

    # -------------------------------------------------------------- geometry

    def cell_centers(self) -> torch.Tensor:
        zc, rc = self.center_lines()
        zz, rr = torch.meshgrid(zc, rc, indexing="ij")
        return torch.stack([zz.reshape(-1), rr.reshape(-1)], dim=1)

    def cell_volumes(self) -> torch.Tensor:
        ring = _PI * (self._rn[1:] ** 2 - self._rn[:-1] ** 2)   # (nr,)
        return (self._wz[:, None] * ring[None, :]).reshape(-1)

    # ---------------------------------------------------------- connectivity

    def _cell_flat(self, iz: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        return iz * self._shape[1] + ir

    def _build_faces(self) -> FaceRecords:
        nz, nr = self._shape
        zc, rc = self.center_lines()
        rows: list[dict] = []

        # radial faces: between rings (iz, ir) and (iz, ir+1) at r-node ir+1
        iz, ir = torch.meshgrid(
            torch.arange(nz), torch.arange(nr - 1), indexing="ij"
        )
        iz, ir = iz.reshape(-1), ir.reshape(-1)
        r_node = self._rn[ir + 1]
        rad = dict(
            cell_i=self._cell_flat(iz, ir),
            cell_j=self._cell_flat(iz, ir + 1),
            area=2.0 * _PI * r_node * self._wz[iz],
            dist_l=0.5 * self._wr[ir],
            dist_r=0.5 * self._wr[ir + 1],
            normal=torch.stack(
                [torch.zeros_like(r_node), torch.ones_like(r_node)], dim=1
            ),
            centroid=torch.stack([zc[iz], r_node], dim=1),
        )
        rows.append(rad)

        # vertical faces: between (iz, ir) and (iz+1, ir) at z-node iz+1
        iz, ir = torch.meshgrid(
            torch.arange(nz - 1), torch.arange(nr), indexing="ij"
        )
        iz, ir = iz.reshape(-1), ir.reshape(-1)
        ring = _PI * (self._rn[ir + 1] ** 2 - self._rn[ir] ** 2)
        vert = dict(
            cell_i=self._cell_flat(iz, ir),
            cell_j=self._cell_flat(iz + 1, ir),
            area=ring,
            dist_l=0.5 * self._wz[iz],
            dist_r=0.5 * self._wz[iz + 1],
            normal=torch.stack(
                [torch.ones_like(ring), torch.zeros_like(ring)], dim=1
            ),
            centroid=torch.stack([self._zn[iz + 1], rc[ir]], dim=1),
        )
        rows.append(vert)

        return FaceRecords(
            **{
                k: torch.cat([r[k] for r in rows]).to(
                    torch.long if k in ("cell_i", "cell_j") else torch.float64
                )
                for k in ("cell_i", "cell_j", "area", "dist_l", "dist_r",
                          "normal", "centroid")
            }
        )

    def _build_boundary(self) -> BoundaryRecords:
        nz, nr = self._shape
        zc, rc = self.center_lines()
        ring = _PI * (self._rn[1:] ** 2 - self._rn[:-1] ** 2)   # (nr,)
        cells, areas, normals, cents = [], [], [], []

        # outer shell r = R (+r outward). The AXIS (r = 0) emits nothing:
        # its face area is exactly zero.
        iz = torch.arange(nz)
        r_out = self._rn[-1]
        cells.append(self._cell_flat(iz, torch.full_like(iz, nr - 1)))
        areas.append(2.0 * _PI * r_out * self._wz)
        normals.append(torch.stack(
            [torch.zeros(nz, dtype=torch.float64),
             torch.ones(nz, dtype=torch.float64)], dim=1))
        cents.append(torch.stack(
            [zc, torch.full((nz,), float(r_out), dtype=torch.float64)], dim=1))

        # top (z = z0, outward −z) and bottom (outward +z) annuli
        ir = torch.arange(nr)
        for iz_row, node, sign in ((0, self._zn[0], -1.0),
                                   (nz - 1, self._zn[-1], +1.0)):
            cells.append(self._cell_flat(torch.full_like(ir, iz_row), ir))
            areas.append(ring.clone())
            normals.append(torch.stack(
                [torch.full((nr,), sign, dtype=torch.float64),
                 torch.zeros(nr, dtype=torch.float64)], dim=1))
            cents.append(torch.stack(
                [torch.full((nr,), float(node), dtype=torch.float64), rc],
                dim=1))

        return BoundaryRecords(
            cell=torch.cat(cells).to(torch.long),
            area=torch.cat(areas).to(torch.float64),
            normal=torch.cat(normals).to(torch.float64),
            centroid=torch.cat(cents).to(torch.float64),
        )

    def face_neighbors(self) -> FaceRecords:
        """Return interior face connectivity (ring-true record geometry).

        Cached on the (immutable) mesh; returns a fresh clone each call,
        the same ownership contract as :meth:`TensorMesh.face_neighbors` /
        :meth:`OctreeMesh.face_neighbors`, so a consumer mutating the
        returned records cannot corrupt the mesh's cache.
        """
        if self._faces is None:
            self._faces = self._build_faces()
        return self._faces.clone()

    def boundary_faces(self) -> BoundaryRecords:
        """Return boundary face records (axis faces excluded by construction).

        Cached on the (immutable) mesh; returns a fresh clone each call;
        see :meth:`face_neighbors`.
        """
        if self._boundary is None:
            self._boundary = self._build_boundary()
        return self._boundary.clone()

    def append_connections(self, extra: FaceRecords) -> "Mesh":
        """Return a connectivity-equivalent mesh with ``extra`` faces appended (NNC).

        Part of the :class:`ConnectivityMesh` contract. The result is an
        :class:`~geobrain.mesh.unstructured.UnstructuredMesh` built
        directly from this mesh's own SoA (``cell_centers``/``cell_volumes``/
        ``face_neighbors() + extra``/``boundary_faces()``): once non-neighbour
        connections (NNC) are appended the ring-adjacency invariants
        (``shape``/``cell_widths``/``center_lines``) no longer describe the
        connectivity, so the returned mesh correctly does NOT declare
        :class:`CylindricalGeometryMesh`: the same demotion policy as
        :meth:`TensorMesh.append_connections` /
        :meth:`OctreeMesh.append_connections`. The return semantics match
        :meth:`UnstructuredMesh.append_connections`: a new mesh whose
        ``face_neighbors()`` is the original face SoA with ``extra``
        concatenated and whose boundary faces are unchanged.
        """
        from .unstructured import UnstructuredMesh

        return UnstructuredMesh.from_mesh(self, extra_faces=extra)

    # --------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        from .serialization import MESH_SCHEMA_VERSION, _ser_tensor

        return {
            "type": "CylindricalMesh",
            "version": MESH_SCHEMA_VERSION,
            "shape": list(self._shape),
            "wz": _ser_tensor(self._wz),
            "wr": _ser_tensor(self._wr),
            "z0": self._z0,
        }
