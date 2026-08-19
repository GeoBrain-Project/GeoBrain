"""TensorMesh: Cartesian tensor-product mesh (uniform or per-axis non-uniform).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
import numbers
import warnings
from typing import Sequence

import torch

from ..core.errors import GeoBrainError
from .base import Mesh
from .capabilities import (
    BoundaryRecords, ConnectivityMesh, FaceRecords, GeometryMesh,
    PrismGeometryMesh, StructuredMesh, UniformMesh,
)


class TensorMesh(Mesh):
    """Cartesian tensor-product mesh: uniform or per-axis non-uniform.

    The convention is ``shape = (nz, nx)`` for 2D or ``(nz, nx, ny)`` for 3D;
    depth is the slow axis, then x, with y the fast axis (the platform-wide
    axis order shared by the wave, EM, potential-field and vis families).

    Two construction modes:

    - ``spacing=(dz, dx[, dy])``: each axis uses a single constant cell width.
    - ``cell_widths=(wz, wx[, wy])`` where each ``w*`` is a 1-D tensor with
      ``len = shape[axis]``. Useful for padded / geometrically-growing boundary
      layers.

    ``cell_widths`` and ``spacing`` are mutually exclusive. After construction,
    :attr:`cell_widths` is always populated (the ``spacing=`` path expands to
    constant 1-D tensors).

    Uniformity (:attr:`is_uniform`, the :class:`UniformMesh` capability, and
    ``__eq__``) is decided by GEOMETRY, not by which keyword was used: a mesh is
    uniform iff every axis's ``cell_widths`` are all-equal (per axis;
    ``torch.allclose`` against the axis's first width). So a ``cell_widths``
    mesh built with a per-axis constant is uniform and compares ``==`` to its
    ``spacing=`` twin, while a genuinely-graded mesh is non-uniform, matching
    the value-based uniformity that the Yee ``curl_curl`` assembly assumes.
    """# Plain class-level default: instances OVERRIDE it (capability
    # refinement), which a ClassVar annotation would mistype.
    mesh_capabilities: frozenset[type] = frozenset({
        GeometryMesh, StructuredMesh, ConnectivityMesh, PrismGeometryMesh,
    })

    def __init__(
        self,
        shape: Sequence[int],
        spacing: Sequence[float] | None = None,
        *,
        cell_widths: Sequence[torch.Tensor] | None = None,
        origin: Sequence[float] | None = None,
    ) -> None:
        """
        Build a uniform or non-uniform Cartesian mesh.

        Args:
            shape: Cell counts per axis, length 2 (``(nz, nx)``) or 3
                (``(nz, nx, ny)``).
            spacing: Constant per-axis cell width. Mutually exclusive with
                ``cell_widths``.
            cell_widths: Per-axis 1-D width tensors (length ``shape[axis]``).
                Mutually exclusive with ``spacing``.
            origin: Per-axis coordinate offset of the mesh's near (index-0)
                corner, in mesh-axis order ``(oz, ox[, oy])``; a scalar
                broadcasts to every axis (like ``spacing``). ``None`` (the
                default) means a zero origin, cell centres start at
                ``0.5 * width`` on each axis, byte-identical to the pre-origin
                behaviour. A non-zero origin shifts every coordinate-producing
                method (:meth:`cell_centers`, :meth:`cell_bounds`, and the face /
                boundary centroids) by that offset; widths, volumes and areas are
                origin-invariant.

        Raises:
            GeoBrainError: On bad rank, non-positive sizes, passing both /
                neither of ``spacing`` and ``cell_widths``, or an ``origin`` whose
                length mismatches the mesh dimension or holds a non-finite value.
        """
        if spacing is not None and cell_widths is not None:
            raise GeoBrainError(
                "pass at most one of spacing or cell_widths",
                object_name="TensorMesh",
                field="spacing/cell_widths",
                expected="exactly one of {spacing, cell_widths}",
                actual="both",
            )
        if spacing is None and cell_widths is None:
            raise GeoBrainError(
                "must pass one of spacing or cell_widths",
                object_name="TensorMesh",
                field="spacing/cell_widths",
                expected="exactly one of {spacing, cell_widths}",
                actual="neither",
            )
        if not (2 <= len(shape) <= 3):
            raise GeoBrainError(
                "TensorMesh supports only 2D and 3D",
                object_name="TensorMesh",
                field="shape",
                expected="length 2 or 3",
                actual=tuple(shape),
            )
        for axis, n in enumerate(shape):
            # Accept any integral type (incl. numpy integers) but NOT bool, which
            # is an int subclass that would silently become a cell count of 0/1.
            if isinstance(n, bool) or not isinstance(n, numbers.Integral) or int(n) <= 0:
                raise GeoBrainError(
                    "TensorMesh cell count must be a positive int",
                    object_name="TensorMesh",
                    field=f"shape[{axis}]",
                    expected="positive int",
                    actual=n,
                )
        self._shape = tuple(int(s) for s in shape)
        self._n_dim = len(self._shape)
        self._n_cells = 1
        for s in self._shape:
            self._n_cells *= s

        # Coordinate origin (per-axis offset of the mesh's near corner). ``None``
        # -> zeros, which reproduces the pre-origin geometry bit-for-bit. Stored
        # as a plain float tuple in mesh-axis order; only the coordinate-producing
        # methods read it (widths/volumes/areas are origin-invariant).
        if isinstance(origin, (int, float)) and not isinstance(origin, bool):
            origin = tuple(float(origin) for _ in shape)  # scalar broadcast
        if origin is None:
            self._origin: tuple[float, ...] = (0.0,) * self._n_dim
        else:
            if len(origin) != self._n_dim:
                raise GeoBrainError(
                    "origin length must match mesh dimension",
                    object_name="TensorMesh",
                    field="origin",
                    expected=self._n_dim,
                    actual=len(origin),
                )
            origin_f = tuple(float(o) for o in origin)
            for axis, o in enumerate(origin_f):
                if not math.isfinite(o):
                    raise GeoBrainError(
                        "origin entries must be finite",
                        object_name="TensorMesh",
                        field=f"origin[{axis}]",
                        expected="finite float",
                        actual=o,
                    )
            self._origin = origin_f

        if isinstance(spacing, (int, float)) and not isinstance(spacing, bool):
            # Scalar spacing = the ubiquitous equal-spacing case; broadcast
            # to every axis.
            spacing = tuple(float(spacing) for _ in shape)
        if spacing is not None:
            if len(shape) != len(spacing):
                raise GeoBrainError(
                    "shape and spacing must have the same length",
                    object_name="TensorMesh",
                    field="spacing",
                    expected=len(shape),
                    actual=len(spacing),
                )
            for axis, h in enumerate(spacing):
                if h <= 0:
                    raise GeoBrainError(
                        "TensorMesh spacing must be positive",
                        object_name="TensorMesh",
                        field=f"spacing[{axis}]",
                        expected="> 0",
                        actual=h,
                    )
            self._spacing = tuple(float(h) for h in spacing)
            self._cell_widths = tuple(
                torch.full((self._shape[i],), self._spacing[i], dtype=torch.float64)
                for i in range(len(self._shape))
            )
            self._uniform = True
        else:
            assert cell_widths is not None  # narrow for mypy
            if len(shape) != len(cell_widths):
                raise GeoBrainError(
                    "shape and cell_widths must have the same length",
                    object_name="TensorMesh",
                    field="cell_widths",
                    expected=len(shape),
                    actual=len(cell_widths),
                )
            stored: list[torch.Tensor] = []
            for axis, w in enumerate(cell_widths):
                if not isinstance(w, torch.Tensor):
                    raise GeoBrainError(
                        "cell_widths entries must be torch.Tensor",
                        object_name="TensorMesh",
                        field=f"cell_widths[{axis}]",
                        expected="torch.Tensor",
                        actual=type(w).__name__,
                    )
                if w.ndim != 1:
                    raise GeoBrainError(
                        "cell_widths entries must be 1-D",
                        object_name="TensorMesh",
                        field=f"cell_widths[{axis}]",
                        expected="1-D tensor",
                        actual=f"{w.ndim}-D",
                    )
                if int(w.shape[0]) != self._shape[axis]:
                    raise GeoBrainError(
                        "cell_widths length must match shape on each axis",
                        object_name="TensorMesh",
                        field=f"cell_widths[{axis}]",
                        expected=self._shape[axis],
                        actual=int(w.shape[0]),
                    )
                if w.requires_grad:
                    warnings.warn(
                        f"TensorMesh cell_widths[{axis}] has requires_grad=True, but "
                        "mesh geometry is a non-differentiable constant: the widths "
                        "are detached and no gradient flows through them.",
                        UserWarning,
                        stacklevel=2,
                    )
                # Canonicalize graded geometry to CPU float64 (device is a
                # per-field/consumer concern), matching how the uniform path's
                # torch.full lands on CPU, so a projection built from a
                # CUDA-resident graded mesh has consistent-device geometry.
                w_f64 = w.detach().cpu().to(dtype=torch.float64)
                if not bool(torch.isfinite(w_f64).all()):
                    raise GeoBrainError(
                        "cell_widths must be finite",
                        object_name="TensorMesh",
                        field=f"cell_widths[{axis}]",
                        expected="all finite",
                        actual="non-finite entries present",
                    )
                if not bool((w_f64 > 0).all()):
                    raise GeoBrainError(
                        "cell_widths must all be positive",
                        object_name="TensorMesh",
                        field=f"cell_widths[{axis}]",
                        expected="> 0",
                        actual="<= 0 entries present",
                    )
                # ``.clone()`` so the mesh OWNS its widths: ``.to(float64)`` is a
                # no-op share when the caller already passes float64 (e.g.
                # ``torch.from_numpy(wz)``), and an un-cloned alias would let the
                # caller corrupt the mesh geometry by mutating their tensor,
                # breaking the "immutable after construction" invariant the lazy
                # caches rely on. (The uniform path is already safe via torch.full.)
                stored.append(w_f64.clone())
            self._cell_widths = tuple(stored)
            self._spacing = tuple(float(w.mean().item()) for w in self._cell_widths)
            # Value-based uniformity: uniform iff EVERY axis's widths are all-equal
            # (per axis), NOT merely because ``cell_widths=`` was the constructor.
            # A per-axis constant (e.g. ``torch.full``) is therefore uniform and
            # picks up the ``UniformMesh`` capability + compares ``==`` to its
            # ``spacing=`` twin; a genuinely-graded axis makes the mesh non-uniform.
            # ``rtol=1e-9`` treats last-ULP float noise as constant while still
            # rejecting real gradation (which varies by orders more).
            self._uniform = all(
                bool(torch.allclose(w, w[0], rtol=1e-9, atol=1e-12))
                for w in self._cell_widths
            )

        # Lazy geometry/topology caches. A TensorMesh is immutable after
        # construction, so cell_centers / cell_bounds / cell_volumes /
        # face_neighbors / boundary_faces are loop-invariant across an inversion
        # (only the property field changes). Compute once, hand out a fresh clone
        # each call so callers keep the "returns a new object" contract.
        self._cc_cache: torch.Tensor | None = None
        self._cb_cache: torch.Tensor | None = None
        self._cv_cache: torch.Tensor | None = None
        self._fn_cache: FaceRecords | None = None
        self._bf_cache: BoundaryRecords | None = None

        # Per-instance capability refinement: a value-UNIFORM mesh additionally
        # declares ``UniformMesh`` so operators whose stencils assume constant
        # spacing (FDTD wave, Helmholtz2D, 2-D structured EM) can require it
        # declaratively; a genuinely-graded (non-constant cell_widths) mesh declares
        # only StructuredMesh and is rejected by a ``(UniformMesh,)`` gate. Because
        # ``self._uniform`` is now geometric, a constant-``cell_widths`` mesh gets
        # ``UniformMesh`` too. Mirrors UnstructuredMesh's per-instance refinement of
        # EdgeConnectivityMesh; ``require_mesh`` reads the instance attribute first,
        # the ClassVar as fallback.
        _base = {GeometryMesh, StructuredMesh, ConnectivityMesh, PrismGeometryMesh}
        self.mesh_capabilities = (
            frozenset(_base | {UniformMesh}) if self._uniform else frozenset(_base)
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Cell counts per axis, ``(nz, nx)`` in 2-D or ``(nz, nx, ny)`` in 3-D."""
        return self._shape

    @property
    def spacing(self) -> tuple[float, ...]:
        """Uniform spacing per axis.

        For uniform meshes, returns the spacing passed at construction.
        For non-uniform meshes, returns the per-axis MEAN cell width, a
        lossy stand-in kept for backward compat with code that assumes
        uniform. Reading it on a graded mesh emits a one-time UserWarning
        (the classic h-averaging footgun); gate on
        ``declares(UniformMesh)`` or consume ``cell_widths`` instead.
        """
        if not self._uniform and not getattr(self, "_spacing_warned", False):
            import warnings

            warnings.warn(
                "TensorMesh.spacing on a non-uniform mesh returns the per-axis "
                "MEAN cell width, not a true spacing; gate on "
                "declares(UniformMesh) or use cell_widths for graded meshes",
                UserWarning,
                stacklevel=2,
            )
            self._spacing_warned = True
        return self._spacing

    @property
    def cell_widths(self) -> tuple[torch.Tensor, ...]:
        """Per-axis 1-D width tensors, length ``shape[axis]`` each.

        Always populated: uniform meshes expand to constant tensors.
        (Name-sharing note: OctreeMesh's ``cell_widths()`` is a METHOD
        returning per-cell ``(n_cells, n_dim)`` widths and CylindricalMesh's
        is a ``(wz, wr)`` pair; see the OctreeMesh docstring.) Hands out
        a fresh ``.clone()`` per axis (like ``cell_centers`` / ``cell_bounds``),
        so a consumer cannot mutate the mesh's internal widths through the
        returned tensors.
        """
        return tuple(w.clone() for w in self._cell_widths)

    @property
    def is_uniform(self) -> bool:
        """``True`` iff the geometry is uniform: every axis has a constant cell
        width (``torch.allclose`` per axis), regardless of which constructor
        (``spacing=`` or a constant ``cell_widths=``) was used."""
        return self._uniform

    @property
    def origin(self) -> tuple[float, ...]:
        """Per-axis coordinate offset of the mesh's near corner.

        Mesh-axis order ``(oz, ox[, oy])``; defaults to zeros. The offset added
        to :meth:`cell_centers` / :meth:`cell_bounds` / face & boundary
        centroids. Widths, volumes and areas are origin-invariant.
        """
        return self._origin

    @property
    def dz(self) -> float:
        """First-axis (depth) spacing."""
        return self._spacing[0]

    @property
    def dx(self) -> float:
        """``x``-axis spacing.

        Under the platform ``(nz, nx)`` / ``(nz, nx, ny)`` convention ``x`` is
        axis 1 (the last axis in 2-D, the *middle* axis in 3-D), so this returns
        ``spacing[1]`` in both cases.
        """
        return self._spacing[1]

    @property
    def dy(self) -> float:
        """``y``-axis spacing: only defined for 3-D meshes.

        Under the ``(nz, nx, ny)`` 3-D convention ``y`` is the last/fast axis
        (index 2), so this returns ``spacing[-1]``.
        """
        if len(self._spacing) < 3:
            raise GeoBrainError(
                "dy is only defined for 3D meshes",
                object_name="TensorMesh", field="dy",
                expected="3D mesh", actual=f"{len(self._spacing)}D",
            )
        return self._spacing[-1]

    def center_lines(self) -> tuple[torch.Tensor, ...]:
        """Per-axis 1-D cell-CENTRE coordinates (length ``shape[axis]`` each).

        Float64, origin-shifted, exact on graded meshes. BITWISE identical to
        the per-axis projections of :meth:`cell_centers`; it mirrors the same
        branch: ``origin + (arange + 0.5)·h`` on a uniform mesh,
        ``origin + cumsum(w) − 0.5·w`` on a graded one. The single home for
        the per-axis centre vectors consumers used to rebuild by hand (some
        silently assuming a zero origin).
        """
        if self._uniform:
            return tuple(
                (torch.arange(n, dtype=torch.float64) + 0.5) * h + o
                for n, h, o in zip(self._shape, self._spacing, self._origin)
            )
        return tuple(
            (torch.cumsum(w, dim=0) - 0.5 * w).to(dtype=torch.float64) + o
            for w, o in zip(self._cell_widths, self._origin)
        )

    def node_lines(self) -> tuple[torch.Tensor, ...]:
        """Per-axis 1-D NODE (cell-edge) coordinates (length ``shape[axis]+1``).

        Float64, origin-shifted, exact on graded meshes. Arithmetic contract
        (deliberately mirrors :meth:`center_lines`'s branching):

        - uniform mesh: the exact multiplicative form ``origin + k·h``
          (node ``k`` carries no accumulated rounding; bitwise identical to
          the ``arange(n+1)·d`` arithmetic of the structured Yee engines);
        - graded mesh: ``origin + [0, cumsum(w)]``, bitwise identical to the
          edges :meth:`cell_bounds` is built from.

        The two forms differ by ~1 ulp on a uniform mesh with a
        non-representable spacing; use :meth:`cell_bounds` (or the projection
        module's ``_tm_axis_edges``) when box-consistency across the whole
        mesh matters more than node-k exactness.
        """
        if self._uniform:
            return tuple(
                torch.arange(n + 1, dtype=torch.float64) * h + o
                for n, h, o in zip(self._shape, self._spacing, self._origin)
            )
        return tuple(
            torch.cat([w.new_zeros(1), torch.cumsum(w, dim=0)]).to(
                dtype=torch.float64) + o
            for w, o in zip(self._cell_widths, self._origin)
        )

    def cell_centers(self) -> torch.Tensor:
        """``(n_cells, n_dim)`` tensor of cell-centre coordinates.

        Each axis starts at ``origin[axis] + 0.5 * width`` (``origin`` defaults to
        zeros, giving the legacy ``0``-start geometry). Cached on the (immutable)
        mesh; returns a fresh clone each call. Per-axis 1-D variants:
        :meth:`center_lines` / :meth:`node_lines`.
        """
        if self._cc_cache is None:
            if self._uniform:
                axes = [
                    (torch.arange(n, dtype=torch.float64) + 0.5) * h + o
                    for n, h, o in zip(self._shape, self._spacing, self._origin)
                ]
            else:
                # cumsum(widths) - widths/2 + origin → cell-centre coordinates
                # starting at ``origin[axis]`` along each axis.
                axes = [
                    (torch.cumsum(w, dim=0) - 0.5 * w).to(dtype=torch.float64) + o
                    for w, o in zip(self._cell_widths, self._origin)
                ]
            grids = torch.meshgrid(*axes, indexing="ij")
            self._cc_cache = torch.stack([g.reshape(-1) for g in grids], dim=-1)
        return self._cc_cache.clone()

    def cell_bounds(self) -> torch.Tensor:
        """``(n_cells, 2*n_dim)`` axis-aligned cell extents ``[lo, hi]`` per axis.

        Layout: ``[lo_0, hi_0, lo_1, hi_1, ...]`` (one lo/hi pair per axis,
        axis order = mesh axis order). Cell order matches :meth:`cell_centers`
        exactly (axis-0 slowest, x/y faster). Computed from per-axis cumulative
        cell widths, **exact for non-uniform meshes**, where ``spacing`` (the
        per-axis mean) would be wrong.
        """
        if self._cb_cache is None:
            edges = [
                torch.cat([w.new_zeros(1), torch.cumsum(w, dim=0)]) + o
                for w, o in zip(self._cell_widths, self._origin)
            ]
            los = [e[:-1] for e in edges]
            his = [e[1:] for e in edges]
            lo_grids = torch.meshgrid(*los, indexing="ij")
            hi_grids = torch.meshgrid(*his, indexing="ij")
            cols: list[torch.Tensor] = []
            for lo_g, hi_g in zip(lo_grids, hi_grids):
                cols.append(lo_g.reshape(-1))
                cols.append(hi_g.reshape(-1))
            self._cb_cache = torch.stack(cols, dim=-1)
        return self._cb_cache.clone()

    def cell_volumes(self) -> torch.Tensor:
        """``(n_cells,)`` cell volumes (area in 2-D, volume in 3-D).

        For uniform meshes each cell has the same volume; for non-uniform meshes
        the volume is the tensor product of the per-axis widths at that grid
        position. Cell ordering matches :meth:`cell_centers` (``indexing="ij"``).
        """
        if self._cv_cache is None:
            per_axis = [w.to(dtype=torch.float64) for w in self._cell_widths]
            grids = torch.meshgrid(*per_axis, indexing="ij")
            vol = grids[0]
            for g in grids[1:]:
                vol = vol * g
            self._cv_cache = vol.reshape(-1)
        return self._cv_cache.clone()

    def face_neighbors(self) -> FaceRecords:
        """Return interior face connectivity for this tensor mesh.

        One :class:`FaceRecords` entry per interior face (shared by two cells).
        Faces are enumerated axis by axis in mesh-axis order (axis 0 → 2:
        z-faces, then x-faces, then y-faces within each axis group).

        Flat cell indices follow the LAST-AXIS-fastest convention that matches
        :meth:`cell_centers` (the strides come from ``self._shape[a+1:]``):
        the last axis varies fastest, the first axis slowest. For the canonical
        3-D shape ``(nz, nx, ny)`` this is
        ``flat = iy + ix*ny + iz*nx*ny`` (ny fastest, then nx, then nz).

        Returns:
            :class:`FaceRecords` with ``nf`` rows where
            ``nf = Σ_a (shape[a]-1) · Π_{b≠a} shape[b]``.

        Cached on the (immutable) mesh; returns a fresh clone each call.
        """
        if self._fn_cache is not None:
            return self._fn_cache.clone()
        n_dim = len(self._shape)
        widths = [w.to(torch.float64) for w in self._cell_widths]
        # Cell-centre coordinates along each axis (shifted by the mesh origin).
        axes_c = [
            torch.cumsum(w, 0) - 0.5 * w + o
            for w, o in zip(widths, self._origin)
        ]

        # Strides for last-axis-fastest flat indexing.
        # shape = (nz, nx, ny): stride[2]=1, stride[1]=ny, stride[0]=nx*ny.
        strides: list[int] = [1] * n_dim
        for a in range(n_dim - 2, -1, -1):
            strides[a] = strides[a + 1] * self._shape[a + 1]

        i_list: list[torch.Tensor] = []
        j_list: list[torch.Tensor] = []
        area_list: list[torch.Tensor] = []
        dl_list: list[torch.Tensor] = []
        dr_list: list[torch.Tensor] = []
        axis_list: list[torch.Tensor] = []
        centroid_list: list[torch.Tensor] = []

        ranges = [torch.arange(s, dtype=torch.long) for s in self._shape]

        for axis in range(n_dim):
            if self._shape[axis] < 2:
                continue
            # Grid ranges: full on all axes except the face axis which runs
            # 0..n-2 (interior faces only: left cell index).
            grid_ranges = [
                ranges[a] if a != axis else ranges[a][:-1]
                for a in range(n_dim)
            ]
            mesh_idx = torch.meshgrid(*grid_ranges, indexing="ij")

            # Flat index of the left cell.
            left_flat = torch.zeros(mesh_idx[0].numel(), dtype=torch.long)
            for a in range(n_dim):
                left_flat = left_flat + mesh_idx[a].reshape(-1) * strides[a]
            right_flat = left_flat + strides[axis]

            # Face-axis index of the left cell (for per-cell width look-up).
            ia = mesh_idx[axis].reshape(-1)

            # Half-distances from cell centres to the shared face.
            w_axis = widths[axis]
            dl = 0.5 * w_axis[ia]
            dr = 0.5 * w_axis[ia + 1]

            # Face area = product of widths on all axes OTHER than the face axis.
            area = torch.ones(left_flat.shape[0], dtype=torch.float64)
            for a in range(n_dim):
                if a == axis:
                    continue
                area = area * widths[a][mesh_idx[a].reshape(-1)]

            nf = left_flat.shape[0]

            # Face centroid: left-cell centre shifted half a left-cell width
            # along the face normal.
            centroid = torch.zeros(nf, n_dim, dtype=torch.float64)
            for a in range(n_dim):
                centroid[:, a] = axes_c[a][mesh_idx[a].reshape(-1)]
            centroid[:, axis] = axes_c[axis][ia] + 0.5 * w_axis[ia]

            i_list.append(left_flat)
            j_list.append(right_flat)
            area_list.append(area)
            dl_list.append(dl)
            dr_list.append(dr)
            axis_list.append(torch.full((nf,), axis, dtype=torch.long))
            centroid_list.append(centroid)

        if not i_list:
            # No interior faces (e.g. a single-cell mesh): return empty records
            # rather than crashing on torch.cat of an empty list.
            self._fn_cache = FaceRecords(
                cell_i=torch.empty(0, dtype=torch.long),
                cell_j=torch.empty(0, dtype=torch.long),
                area=torch.empty(0, dtype=torch.float64),
                dist_l=torch.empty(0, dtype=torch.float64),
                dist_r=torch.empty(0, dtype=torch.float64),
                normal=torch.empty(0, n_dim, dtype=torch.float64),
                centroid=torch.empty(0, n_dim, dtype=torch.float64),
            )
            return self._fn_cache.clone()

        cell_i = torch.cat(i_list)
        cell_j = torch.cat(j_list)
        axis_t = torch.cat(axis_list)
        nf_total = cell_i.shape[0]

        normal = torch.zeros(nf_total, n_dim, dtype=torch.float64)
        normal[torch.arange(nf_total), axis_t] = 1.0

        self._fn_cache = FaceRecords(
            cell_i=cell_i,
            cell_j=cell_j,
            area=torch.cat(area_list),
            dist_l=torch.cat(dl_list),
            dist_r=torch.cat(dr_list),
            normal=normal,
            centroid=torch.cat(centroid_list),
        )
        return self._fn_cache.clone()

    def boundary_faces(self) -> BoundaryRecords:
        """Return outer (domain-boundary) face records for this tensor mesh.

        One :class:`BoundaryRecords` entry per exterior face, the faces of the
        first/last cell along each axis that have no neighbour. Each axis
        contributes two boundary slabs: the ``lo`` slab (index 0 along that
        axis, **outward** normal ``-axis``) and the ``hi`` slab (index
        ``shape[axis]-1``, outward normal ``+axis``). The outward convention
        (normal points OUT of the owning cell, away from the domain) matches
        :meth:`UnstructuredMesh.boundary_faces`.

        Flat cell indices follow the same last-axis-fastest convention as
        :meth:`face_neighbors` and :meth:`cell_centers`
        (``flat = iy + ix*ny + iz*nx*ny`` for the 3-D shape ``(nz, nx, ny)``).
        The face centroid is the owning cell's centre projected onto the
        boundary plane (its boundary-axis coordinate set to the domain edge,
        ``0`` for the lo slab, the cumulative width for the hi slab).

        Returns:
            :class:`BoundaryRecords` with ``nb`` rows where
            ``nb = Σ_a 2 · Π_{b≠a} shape[b]`` (faces enumerated axis by axis,
            lo slab then hi slab within each axis group).

        Cached on the (immutable) mesh; returns a fresh clone each call.
        """
        if self._bf_cache is not None:
            return self._bf_cache.clone()
        n_dim = len(self._shape)
        widths = [w.to(torch.float64) for w in self._cell_widths]
        # Cell-centre coordinates along each axis (shifted by the mesh origin) and
        # the total extent (sum of widths) used to place the hi-slab face plane.
        axes_c = [
            torch.cumsum(w, 0) - 0.5 * w + o
            for w, o in zip(widths, self._origin)
        ]
        extents = [float(w.sum()) for w in widths]

        # Strides for x-fastest flat indexing (matches face_neighbors).
        strides: list[int] = [1] * n_dim
        for a in range(n_dim - 2, -1, -1):
            strides[a] = strides[a + 1] * self._shape[a + 1]

        ranges = [torch.arange(s, dtype=torch.long) for s in self._shape]

        cell_list: list[torch.Tensor] = []
        area_list: list[torch.Tensor] = []
        normal_list: list[torch.Tensor] = []
        centroid_list: list[torch.Tensor] = []

        for axis in range(n_dim):
            for side, fixed in ((-1.0, 0), (+1.0, self._shape[axis] - 1)):
                # Grid over all axes; the boundary axis is pinned to ``fixed``.
                grid_ranges = [
                    ranges[a] if a != axis else ranges[a][fixed:fixed + 1]
                    for a in range(n_dim)
                ]
                mesh_idx = torch.meshgrid(*grid_ranges, indexing="ij")
                flat = torch.zeros(mesh_idx[0].numel(), dtype=torch.long)
                for a in range(n_dim):
                    flat = flat + mesh_idx[a].reshape(-1) * strides[a]
                nb = flat.shape[0]

                # Face area = product of widths on all axes OTHER than the
                # boundary axis (each of those cells contributes its own width).
                area = torch.ones(nb, dtype=torch.float64)
                for a in range(n_dim):
                    if a == axis:
                        continue
                    area = area * widths[a][mesh_idx[a].reshape(-1)]

                # Outward normal: -axis on the lo slab, +axis on the hi slab.
                normal = torch.zeros(nb, n_dim, dtype=torch.float64)
                normal[:, axis] = side

                # Face centroid: owning-cell centre projected onto the boundary
                # plane (boundary-axis coord at the domain edge).
                centroid = torch.zeros(nb, n_dim, dtype=torch.float64)
                for a in range(n_dim):
                    centroid[:, a] = axes_c[a][mesh_idx[a].reshape(-1)]
                centroid[:, axis] = (
                    self._origin[axis] if side < 0
                    else self._origin[axis] + extents[axis]
                )

                cell_list.append(flat)
                area_list.append(area)
                normal_list.append(normal)
                centroid_list.append(centroid)

        self._bf_cache = BoundaryRecords(
            cell=torch.cat(cell_list),
            area=torch.cat(area_list),
            normal=torch.cat(normal_list),
            centroid=torch.cat(centroid_list),
        )
        return self._bf_cache.clone()

    def append_connections(self, extra: FaceRecords) -> "Mesh":
        """Return a connectivity-equivalent mesh with ``extra`` faces appended (NNC).

        Part of the :class:`ConnectivityMesh` contract. The result is an
        :class:`~geobrain.mesh.unstructured.UnstructuredMesh` built
        directly from this mesh's own SoA (``cell_centers``/``cell_volumes``/
        ``face_neighbors() + extra``/``boundary_faces()``): once non-neighbour
        connections (NNC) are appended the tensor-product invariants (``shape``/
        ``spacing``/``cell_widths``) no longer describe the connectivity, so the
        returned mesh correctly does NOT declare :class:`StructuredMesh` (nor
        :class:`PrismGeometryMesh`). The return semantics match
        :meth:`UnstructuredMesh.append_connections`: a new mesh whose
        ``face_neighbors()`` is the original face SoA with ``extra``
        concatenated and whose boundary faces are unchanged.
        """
        from .unstructured import UnstructuredMesh

        return UnstructuredMesh.from_mesh(self, extra_faces=extra)

    def to_dict(self) -> dict:
        """Declarative, JSON-native serialization (round-trips via ``mesh_from_dict``).

        Schema (``version`` is :data:`~geobrain.mesh.serialization.MESH_SCHEMA_VERSION`)::

            {"type": "TensorMesh", "version": 1,
             "shape": [nz, nx[, ny]],
             "uniform": bool,
             "cell_widths": [<tensor-envelope per axis>]}

        ``cell_widths`` is always stored (the per-axis width vectors, constant
        for a uniform mesh); the ``uniform`` flag lets the factory rebuild a
        uniform mesh via ``spacing=`` so the reconstruction compares ``==`` to
        this mesh (``TensorMesh.__eq__`` distinguishes uniform from graded).
        ``mesh_from_dict(m.to_dict()) == m`` for both uniform and graded meshes.
        """
        from .serialization import MESH_SCHEMA_VERSION, _ser_tensor

        return {
            "type": "TensorMesh",
            "version": MESH_SCHEMA_VERSION,
            "shape": list(self._shape),
            "uniform": bool(self._uniform),
            "cell_widths": [_ser_tensor(w) for w in self._cell_widths],
            "origin": list(self._origin),
        }

    def __repr__(self) -> str:
        origin_part = (
            "" if all(o == 0.0 for o in self._origin)
            else f", origin={self._origin}"
        )
        if self._uniform:
            return (
                f"TensorMesh(shape={self._shape}, spacing={self._spacing}"
                f"{origin_part})"
            )
        return (
            f"TensorMesh(shape={self._shape}, "
            f"cell_widths=<non-uniform, mean={self._spacing}>{origin_part})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorMesh):
            return False
        if self._shape != other._shape:
            return False
        if self._origin != other._origin:
            return False
        # Compare the actual per-axis cell_widths (the true geometry), NOT the
        # ``_uniform`` construction flag: a constant-``cell_widths`` mesh and its
        # ``spacing=`` twin store byte-identical ``torch.full`` widths, so they
        # compare equal; a genuinely-graded mesh differs from a uniform one on the
        # widths. ``spacing`` (per-axis mean) and origin are preserved implicitly;
        # equal widths give equal means, and origin is checked above.
        return all(
            a.shape == b.shape and torch.equal(a, b)
            for a, b in zip(self._cell_widths, other._cell_widths)
        )

    def __hash__(self) -> int:
        # Hash from the width geometry (consistent with the width-based __eq__):
        # shape + per-axis (length, sum, first, last) fingerprint + origin. Two
        # equal meshes have byte-identical widths -> identical fingerprints ->
        # equal hash (this covers the constant-``cell_widths`` == ``spacing=`` twin
        # case, since both are the same ``torch.full`` tensor). Deriving the hash
        # from the mean ``spacing`` instead would be UNSAFE: ``float(w.mean())`` can
        # differ in the last ULP from the ``spacing=`` scalar even for identical
        # widths, breaking eq/hash consistency.
        fingerprints = tuple(
            (int(w.shape[0]), float(w.sum().item()),
             float(w[0].item()), float(w[-1].item()))
            for w in self._cell_widths
        )
        return hash(("TensorMesh", self._shape, fingerprints, self._origin))
