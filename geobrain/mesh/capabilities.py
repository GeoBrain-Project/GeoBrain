"""
Mesh capability markers + face/edge/boundary connectivity records.

A *capability* is an interface contract: "a mesh with this capability provides
these methods/attributes". The markers below are plain abstract classes used
ONLY as identity keys in each mesh's ``mesh_capabilities`` frozenset and each
operator's ``requires_mesh_capabilities`` tuple.

Declaration semantics: ``requires_mesh_capabilities`` is the CONJUNCTIVE
(necessary) set; the mesh must declare every listed marker. Operators whose
sufficiency is a CHOICE between engine paths additionally declare
``requires_mesh_capabilities_any: tuple[tuple[type, ...], ...]``: at least
one group must be fully declared on top of the flat set (e.g. inductive EM:
``((StructuredMesh,), (EdgeConnectivityMesh,))``: Yee finite volume or
Nédélec edge elements; an octree satisfies the flat ``ConnectivityMesh``
requirement but neither group, and is rejected by the operator's own gate).
A dedicated contract test audits that every such declaration matches its
runtime behaviour. Membership is **by explicit
declaration**, never by structural (method-presence) matching, that is why
these are NOT ``runtime_checkable`` Protocols. Two reasons: (1) per-instance
refinement (a uniform TensorMesh carries ``UniformMesh``; a 3-D simplex-built
UnstructuredMesh carries ``EdgeConnectivityMesh``) cannot be expressed by a
structural test on the class; (2) structural matching admits ACCIDENTAL
conformance, any object that happens to grow a ``shape`` attribute would
silently pass a probe and let grid-stencil operators accept it, whereas an
explicit declaration keeps membership auditable in one place. (``UnstructuredMesh``'s former tombstone properties were exactly such a
trap; they raise ``AttributeError`` so ``hasattr`` stays honest; the
principle is general.) The docstrings record the interface
each capability requires; the classes carry no implementation.

Declarations normally live on the class (a plain class-level frozenset); a
dimension-polymorphic mesh class whose capability depends on the data it was
built from (UnstructuredMesh is one class for 2-D and 3-D) refines the
declaration on the INSTANCE, still explicit declaration, never structural
matching. ``require_mesh`` reads the instance attribute, falling back to the
class.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch


class _RecordsBase:
    """Generic clone / concat for the SoA connectivity-record dataclasses below.

    Both walk :func:`dataclasses.fields`, so adding a new tensor field to a record
    needs **no** change here (avoids the per-field hand-rolled clone footgun).

    - :meth:`clone` deep-copies (every tensor field cloned): used to hand out a
      defensive copy of a mesh's cached geometry without exposing the cache.
    - :meth:`concat` joins two same-type records along the leading (record) dim,
      the single definition behind every mesh's ``append_connections`` (NNC).
    """

    def clone(self):  # type: ignore[no-untyped-def]
        return replace(self, **{
            f.name: getattr(self, f.name).clone() for f in fields(self)  # type: ignore[arg-type]
        })

    @classmethod
    def concat(cls, a, b):  # type: ignore[no-untyped-def]
        return replace(a, **{
            f.name: torch.cat([getattr(a, f.name), getattr(b, f.name)])
            for f in fields(a)
        })


class _CapabilityMarkerMeta(type):
    """Metaclass for capability markers: fail LOUDLY on the two natural misuses.

    Membership is declaration-based (``mesh.declares(Marker)``), never
    structural or inheritance-based, so ``isinstance(mesh, Marker)`` would
    silently return ``False`` for every mesh, and instantiating a marker is
    meaningless. Both raise with the correct idiom.
    ``issubclass`` between markers (and ``cap in mesh_capabilities`` hashing)
    is untouched.
    """

    def __instancecheck__(cls, instance) -> bool:  # noqa: ANN001
        raise TypeError(
            f"capability membership is declaration-based: use "
            f"mesh.declares({cls.__name__}) or ensure_capable_mesh(...), "
            f"not isinstance"
        )

    def __call__(cls, *args: object, **kwargs: object):
        raise TypeError(
            f"{cls.__name__} is a capability marker and cannot be "
            f"instantiated; meshes DECLARE it via mesh_capabilities"
        )


class GeometryMesh(metaclass=_CapabilityMarkerMeta):
    """Base capability: flat geometry + flat counts, no structured assumptions.

    A mesh declaring this capability provides::

        n_dim: int
        n_cells: int
        cell_centers() -> Tensor  # (n_cells, n_dim)
        cell_volumes() -> Tensor  # (n_cells,)

    Coordinate-column frame: the columns of ``cell_centers()`` (and of any
    coordinate a builder consumes) are in the platform mesh-axis order,
    **column 0 = z (depth, positive down), then x, then y**, matching the
    ``(nz, nx, ny)`` (3-D) / ``(nz, nx)`` (2-D) TensorMesh axis order. Callers
    feeding an :class:`~geobrain.mesh.unstructured.UnstructuredMesh` builder
    a raw ``(x, y, z)`` point cloud must permute it to ``(z, x, y)`` first.

    Every mesh declares this capability.
    """


class StructuredMesh(GeometryMesh):
    """Tensor-product / regular-grid capability. Only TensorMesh declares it.

    Adds to :class:`GeometryMesh`::

        shape: tuple[int, ...]
        spacing: tuple[float, ...]            # per-axis; exact only when is_uniform
        cell_widths: tuple[Tensor, ...]       # property: one 1-D width vector per axis

    ``spacing`` degrades to a per-axis **mean** on a non-uniform mesh, an operator
    that needs exact per-cell geometry must read ``cell_widths`` (the full width
    vectors), not ``spacing``. An operator whose stencil assumes *constant* spacing
    must require :class:`UniformMesh`, not this; ``StructuredMesh`` alone admits
    graded meshes.

    Required by: io/vtk · vis/slicer · reshape-only consumers (Gravity2D).
    Graded-capable per-cell consumers (gravity/magnetics cell_bounds, MT3D
    cell_widths) also declare it.
    """


class UniformMesh(StructuredMesh):
    """Regular-grid capability: **constant** cell spacing per axis (``is_uniform``).

    A uniform mesh's ``spacing`` is exact (not a per-axis mean), so finite-
    difference / Yee stencils that divide by a single constant ``dz``/``dx`` per
    axis are valid. Declared per INSTANCE by :class:`TensorMesh`: only uniform
    instances carry it; a graded (``cell_widths``) TensorMesh declares only
    :class:`StructuredMesh` and is rejected by a ``(UniformMesh,)`` gate.

    Required by: FDTD wave (all engine ops + propagator facades) · Helmholtz2D ·
    the 2-D structured EM solvers DC2D / IP2D. Dual-path operators whose *other*
    branch accepts graded/unstructured meshes (MT2D, DC3D, FDEM3D, TEM3D) must
    NOT put this in their ``requires_mesh_capabilities`` (it would block that
    branch); they gate on a broader capability and check ``mesh.is_uniform``
    inside the structured branch instead.
    """


class ConnectivityMesh(GeometryMesh):
    """Finite-volume connectivity capability: internal face cell-pairs + geometry.

    Adds to :class:`GeometryMesh`::

        face_neighbors() -> FaceRecords
        boundary_faces() -> BoundaryRecords
        append_connections(extra: FaceRecords) -> ConnectivityMesh   # NNC

    Required by: em cell-centred DC/IP/SP and the finite-volume numerics.
    (The flow subsystem runs on its own CartGrid stack and bridges via
    adapters; it does not consume FaceRecords directly.)
    """


class EdgeConnectivityMesh(ConnectivityMesh):
    """Edge-connectivity capability: unique-edge records + signed cell→edge maps.

    Adds to :class:`ConnectivityMesh`::

        edge_records() -> EdgeRecords            # global unique edges (SoA)
        cell_edges()   -> tuple[Tensor, Tensor]  # (n_cells, 6) long edge ids,
                                                 # (n_cells, 6) float64 signs
        cell_nodes()   -> Tensor                 # (n_cells, 4) long, stored order
        node_coords()  -> Tensor                 # (n_nodes, 3) float64

    Records-based by design (the platform's seam philosophy): the mesh hands out
    topology/geometry RECORDS only, which edges exist, how each cell references
    them, and the node table Whitney barycentric-coordinate gradients need.
    Operator algebra (lowest-order Nédélec/Whitney edge-basis assembly, edge
    curl, mass/stiffness inner products) is NEVER implemented on the mesh
    CLASSES; it lives either in the shared core operator factory
    (:mod:`~geobrain.mesh.operators`, free functions consuming records,
    parity-tested against a reference implementation) or physics-side, exactly as
    :class:`ConnectivityMesh` hands :class:`FaceRecords` to the physics-side
    FV assembler.

    Local-edge convention (downstream Whitney assembly depends on it): with the
    cell's four nodes in STORED order, the six local edges are the node pairs
    ``(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)``; the local direction runs from
    the first listed node to the second. The global canonical direction of every
    edge runs from its smaller node id to its larger one, and the ``cell_edges()``
    sign is ``+1`` where the local direction agrees with the canonical one.

    Required by: the em 3-D edge-element curl-curl family, the FDEM3D/TEM3D/MT3D
    unstructured (tetrahedral) branches consume it today (each operator's
    ``_forward_edge_fem`` path builds an ``EdgeAssemblyPlan`` from it). Declared
    per INSTANCE by UnstructuredMesh: only 3-D simplex-built instances carry the
    simplex topology this contract needs (2-D ``from_polygons`` instances do not
    declare it).
    """


class CylindricalGeometryMesh(ConnectivityMesh):
    """Axisymmetric ring-cell capability: structured ``(z, r)`` lines on top
    of ring-true records.

    Adds to :class:`ConnectivityMesh` the structured accessors an
    axisymmetric staggered consumer builds its θ-edge system from::

        shape         -> (nz, nr)
        cell_widths   -> (wz, wr)
        center_lines() / node_lines() -> per-axis (z, r) coordinates

    with the geometric contract that cells are FULL RINGS (records carry
    ``π(r₂²−r₁²)Δz`` volumes and ``2πrΔz`` radial areas) and r starts at
    the symmetry axis. Declared by
    :class:`~geobrain.mesh.cylindrical.CylindricalMesh`; required by
    the axisymmetric EM consumers (FDEMCyl). Deliberately DISJOINT from
    :class:`StructuredMesh`: a cylinder must never satisfy a Cartesian
    stencil gate even though it carries the same accessor names.
    """


class PrismGeometryMesh(GeometryMesh):
    """Per-cell analytic-kernel capability: axis-aligned cell bounds.

    Adds to :class:`GeometryMesh`::

        cell_bounds() -> Tensor   # (n_cells, 2*n_dim): [lo, hi] per axis

    Required by: potential Gravity/Magnetic.
    """


@dataclass(frozen=True, eq=False)
class FaceRecords(_RecordsBase):
    """Struct-of-arrays over the ``nf`` internal faces of a mesh.

    Dtype/device contract: geometry fields are produced CPU float64 and
    index fields ``torch.long`` by every mesh; CONSUMERS move fields to
    their own device/dtype (records offer no ``.to()`` by design).


    ``eq=False``: equality/hash are by object identity, a generated
    field-wise ``__eq__`` would call ``bool()`` on multi-element tensor
    comparisons and raise ``RuntimeError`` (same for the other record
    dataclasses below).

    Each field is a tensor whose leading dim is ``nf``:

    - ``cell_i`` / ``cell_j`` ``(nf,)`` long: the two cells sharing the face.
    - ``area`` ``(nf,)``: face area (length in 2-D).
    - ``dist_l`` / ``dist_r`` ``(nf,)``: cell-centre→face distances on each side.
    - ``normal`` ``(nf, n_dim)``: unit face normal (i→j).
    - ``centroid`` ``(nf, n_dim)``: face centroid coordinates.

    Attributes:
        cell_i: ``(nf,)`` first adjacent cell per internal face.
        cell_j: ``(nf,)`` second adjacent cell.
        area: ``(nf,)`` face areas.
        dist_l: ``(nf,)`` centre-to-face distance on the ``cell_i`` side.
        dist_r: ``(nf,)`` centre-to-face distance on the ``cell_j`` side.
        normal: ``(nf, dim)`` unit normals oriented ``cell_i -> cell_j``.
        centroid: ``(nf, dim)`` face centroids.
    """

    cell_i: torch.Tensor
    cell_j: torch.Tensor
    area: torch.Tensor
    dist_l: torch.Tensor
    dist_r: torch.Tensor
    normal: torch.Tensor
    centroid: torch.Tensor


@dataclass(frozen=True, eq=False)
class EdgeRecords(_RecordsBase):
    """Struct-of-arrays over the ``ne`` unique edges of a mesh.

    Dtype/device contract: geometry fields are produced CPU float64 and
    index fields ``torch.long`` by every mesh; CONSUMERS move fields to
    their own device/dtype (records offer no ``.to()`` by design).


    Each field is a tensor whose leading dim is ``ne``. Every edge has one
    canonical direction, from its smaller global node id to its larger one
    (``nodes[:, 0] < nodes[:, 1]``); per-cell orientation relative to it is
    carried by the ``cell_edges()`` signs, not duplicated here.

    - ``nodes`` ``(ne, 2)`` long: global node pair in canonical order.
    - ``tangent`` ``(ne, n_dim)``: unit tangent along the canonical direction.
    - ``length`` ``(ne,)``: edge length.
    - ``midpoint`` ``(ne, n_dim)``: edge midpoint coordinates.

    Attributes:
        nodes: ``(ne, 2)`` endpoint node indices per edge.
        tangent: ``(ne, dim)`` unit tangents.
        length: ``(ne,)`` edge lengths.
        midpoint: ``(ne, dim)`` edge midpoints.
    """

    nodes: torch.Tensor
    tangent: torch.Tensor
    length: torch.Tensor
    midpoint: torch.Tensor


@dataclass(frozen=True, eq=False)
class BoundaryRecords(_RecordsBase):
    """Struct-of-arrays over the ``nb`` boundary faces of a mesh.

    Dtype/device contract: geometry fields are produced CPU float64 and
    index fields ``torch.long`` by every mesh; CONSUMERS move fields to
    their own device/dtype (records offer no ``.to()`` by design).


    - ``cell`` ``(nb,)`` long: the single owning cell.
    - ``area`` ``(nb,)``: face area.
    - ``normal`` ``(nb, n_dim)``: outward unit normal.
    - ``centroid`` ``(nb, n_dim)``: face centroid coordinates.

    Attributes:
        cell: ``(nb,)`` owning-cell index of each boundary face.
        area: ``(nb,)`` face areas.
        normal: ``(nb, dim)`` outward unit normals.
        centroid: ``(nb, dim)`` face centroids.
    """

    cell: torch.Tensor
    area: torch.Tensor
    normal: torch.Tensor
    centroid: torch.Tensor


__all__ = [
    "GeometryMesh",
    "CylindricalGeometryMesh",
    "StructuredMesh",
    "UniformMesh",
    "ConnectivityMesh",
    "EdgeConnectivityMesh",
    "PrismGeometryMesh",
    "FaceRecords",
    "EdgeRecords",
    "BoundaryRecords",
]
