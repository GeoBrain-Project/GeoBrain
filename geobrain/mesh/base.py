"""Mesh ABC: minimal interface every mesh implementation satisfies.

A ``Mesh`` is a description of a discrete spatial layout. Operators read the mesh
from the typed :class:`~geobrain.core.context.ForwardContext` mesh slot
(``ctx.require_mesh()`` for required reads, ``ctx.mesh`` for optional reads) to
learn cell size, shape, and positions of cell centres. Concrete implementations
live in sibling modules:

- :class:`~geobrain.mesh.tensor.TensorMesh`: Cartesian (uniform or graded).
- :class:`~geobrain.mesh.octree.OctreeMesh`: flat-leaf quadtree/octree.
- :class:`~geobrain.mesh.unstructured.UnstructuredMesh`: explicit-cell
  unstructured (simplex / polygon / SoA-demoted).
- :class:`~geobrain.mesh.cylindrical.CylindricalMesh`: axisymmetric
  ``(z, r)`` ring cells.

The :meth:`Mesh.projection_to` factory returns a differentiable
:class:`~geobrain.mesh.projection.MeshProjection` that maps a field on
``self`` to a field on the target mesh.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # avoid circular import: MeshProjection depends on Mesh
    from .projection import MeshProjection


class Mesh(ABC):
    """A spatial discretisation.

    Concrete subclasses MUST set ``self._n_dim`` (int) and ``self._n_cells``
    (int) in ``__init__``; these flat counts are the GeometryMesh primitives
    and are read directly (no longer derived from ``shape``). ``shape``/
    ``spacing`` are NOT part of this base interface; they belong to the
    ``StructuredMesh`` capability and are provided only by meshes that declare it.
    """

    @property
    def n_dim(self) -> int:
        """Number of spatial dimensions (stored, not derived from shape)."""
        return self._n_dim

    @property
    def n_cells(self) -> int:
        """Total number of cells (stored, not derived from shape)."""
        return self._n_cells

    def declares(self, *caps: type) -> bool:
        """``True`` iff this mesh declares every capability marker in ``caps``.

        The single home for the capability-membership rule: the lookup reads
        ``mesh_capabilities`` off the INSTANCE first (per-instance capability
        refinement, e.g. only a uniform :class:`TensorMesh` carries
        ``UniformMesh``), falling back to the class declaration. Both are
        explicit declarations; membership is never structural
        (method-presence) matching. Consumers holding a mesh object should
        call this instead of hand-spelling
        ``Cap in getattr(mesh, "mesh_capabilities", frozenset())``.
        """
        have = getattr(self, "mesh_capabilities", frozenset())
        return all(cap in have for cap in caps)

    @abstractmethod
    def cell_centers(self) -> torch.Tensor:
        """``(n_cells, n_dim)`` tensor of cell-centre coordinates."""
        ...

    @abstractmethod
    def cell_volumes(self) -> torch.Tensor:
        """``(n_cells,)`` cell volumes (area in 2-D, volume in 3-D)."""
        ...

    def projection_to(self, other: "Mesh", *,
                      field_name: "str | Sequence[str]",
                      method: str | None = None, k_neighbors: int = 4,
                      idw_power: float = 1.0,
                      conservative: bool = True,
                      padding: str = "raise") -> "MeshProjection":
        """Build a differentiable operator that maps field(s) on ``self`` to ``other``.

        ``field_name`` is one field name (a plain string) or a sequence of
        names, a sequence projects every named field through the SAME
        precomputed weights, each written under its own name.

        ``method=None`` (default) auto-picks the interpolant for the mesh pair:
        bilinear/trilinear (``'linear'``) on structured (uniform or graded)
        TensorMesh pairs, volume-weighted ``'overlap'`` on OctreeMesh↔OctreeMesh
        pairs, and inverse-distance (``'idw'``) on the general (unstructured)
        path. Pass an explicit method to override, e.g. ``'barycentric'`` for a
        linear-exact interpolant from a simplex UnstructuredMesh source; a method
        that conflicts with the resolved path raises (e.g. ``'linear'`` on an
        unstructured source).

        ``k_neighbors`` sizes the general k-NN IDW stencil and ``idw_power``
        its Shepard exponent (default 1.0, the historical weight; classical
        Shepard is 2.0); ``conservative`` (default ``True``) makes the
        genuinely-coarsening STRUCTURED kernels mass/mean-preserving,
        equal-extent structured downsampling area-averages; unequal/graded
        TM and cyl↔cyl pairs use exact overlap-volume weights (ring-exact in
        ``(z, r²)`` for cylinders); octree paths volume-weight. The general
        k-NN path is point-interpolating and has NO conservative branch:
        pass ``method='volume'`` there for the volume-binned conservative
        restriction (a construction-time warning flags conservative
        coarsening requests left on plain IDW). ``padding``
        ('raise' | 'border' | 'zeros', default ``'raise'``) sets the
        out-of-coverage policy when the target extends beyond the source
        domain, non-'raise' downgrades warn with the uncovered fraction
        (see ``uncovered`` / ``coverage_report()``). Everything forwards
        straight to :class:`MeshProjection`.
        """
        # Lazy import to break the Mesh ↔ MeshProjection cycle.
        from .projection import MeshProjection

        return MeshProjection(source=self, target=other, field_name=field_name,
                                  method=method, k_neighbors=k_neighbors,
                                  idw_power=idw_power,
                                  conservative=conservative, padding=padding)
