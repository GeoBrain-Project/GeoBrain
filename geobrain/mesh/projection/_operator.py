"""MeshProjection: differentiable field projection between two meshes.

Operator module of the ``projection`` subpackage: the public
MeshProjection / adjoint surface, mode selection, method validation and the
padding policy. Geometry builders live in ``_builders``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch

from ...core.differentiability import DifferentiabilityLevel, DifferentiabilitySpec
from ...core.errors import GeoBrainError
from ...core.operator import PropertyTransform
from ...core.containers import ModelState
from ...core.context import ForwardContext
from ..base import Mesh
from ..capabilities import UniformMesh
from ..octree import OctreeMesh
from ..tensor import TensorMesh
from ..unstructured import UnstructuredMesh
from ._builders import (
    _drop_rows,
    _validate_field_names,
    _validate_idw_power,
    _validate_k_neighbors,
)
from ._kernels import projection_kernel


class MeshProjection(PropertyTransform):
    """
    Differentiable field projection between two ``Mesh`` instances.

    Reads each projected field from ``state.tensors`` shaped like the source
    mesh and writes a tensor shaped like the target mesh under the same name.
    ``field_name`` is one name (a plain string, the common case) or a
    sequence of names: a sequence projects EVERY named field through the SAME
    precomputed weights, each written under its own name (one geometry
    precompute for vp/vs/rho instead of three modules); every other state
    entry passes through untouched. The pull-back weights are torch
    operations:

    - ``TensorMesh → TensorMesh``: ``F.interpolate`` (bilinear / nearest).
    - ``TensorMesh → OctreeMesh``: ``grid_sample`` at the octree centres.
    - ``OctreeMesh → TensorMesh``: precomputed index-map gather (each TM cell reads
      its containing OM leaf).
    - ``OctreeMesh → OctreeMesh``: exact box-overlap volume weights
      ``w(T, S) = vol(T ∩ S) / norm(T)`` (mode ``om_to_om``, auto method
      ``'overlap'``: see below; ``'nearest'`` gives the legacy nearest-centre
      gather instead).

    Autograd flows back through every branch.

    Method names (auto-picked when ``method=None``): ``'linear'``
    (bi/trilinear) on the structured kernels, ``'overlap'`` on OM↔OM,
    ``'idw'`` (k-NN inverse distance) on the general path, ``'nearest'``
    everywhere. ``method='barycentric'`` is an OPT-IN alternative on the
    general path for an :class:`UnstructuredMesh` SOURCE: the source CELL
    CENTRES are Delaunay-triangulated (scipy) and each target point takes the
    barycentric weights of its containing centre-simplex, exact on linear
    fields inside the centre-triangulation hull (IDW is not), falling back to
    the k-NN IDW row outside the hull. It never becomes a default: auto on
    the general path stays ``'idw'``.

    Dtype policy: REAL dtypes only, outputs preserve the input field's
    dtype; internal compute is float64 on EVERY kernel (weights/grids are
    precomputed float64 and the field is cast in and back per forward, a
    no-op for float64 fields; the om_to_tm single-leaf gather is an exact
    index copy with no arithmetic). A complex field is refused loudly: the
    float64 cast would silently discard its imaginary part, project real
    and imaginary parts as two separate fields instead.

    Adjointness (IMPORTANT): each projection is a ONE-WAY interpolant. Its
    autograd backward is the EXACT transpose of ITSELF (``Wᵀ`` for the
    sparse-weight kernels), but the forward ``A → B`` and the reverse ``B → A``
    are DISTINCT interpolants, NOT adjoints of each other, so a round-trip
    ``proj_BA(proj_AB(x))`` is LOSSY (empirically ~23 % RMS on a smooth field)
    and ``A → B`` is not the inverse of ``B → A``. For the true mathematical
    adjoint use :meth:`transpose`: it returns the exact ``Wᵀ`` of the
    MATERIALIZED forward: ``W`` reproduces ``_forward`` including the
    conservative blends and the ``padding='zeros'`` masking, for EVERY
    kernel (the sparse-weight modes ``general`` / ``tm_linear`` / ``om_to_om``
    transpose their stored ``W`` directly; the grid_sample / index-map kernels
    materialize an equivalent sparse ``W`` on demand from the same
    precomputed geometry). Applying it to a target-shaped field gives the
    source-shaped ``Wᵀ x``, identical to the forward's autograd backward.

    Conservative coarsening: with ``conservative=True`` (the default) every
    GENUINELY-COARSENING projection is mass/mean-preserving instead of
    point-sampled (point samples alias integral quantities such as a gravity
    source density). Activation per kernel:

    - equal-extent uniform tm_to_tm downsampling keeps the fast
      ``F.interpolate(mode='area')`` path (identical to overlap-volume weights
      for integer ratios);
    - unequal-extent / shifted-origin uniform tm_to_tm pairs with a coarser
      target use exact overlap-volume weights (``w(T,S) = vol(T∩S)/norm(T)``);
    - a coarser graded (``tm_linear``) target likewise uses overlap-volume
      weights;
    - octree→TM cells overlapping ≥ 2 leaves take the overlap-volume weighted
      mean of those leaves (cells inside a single leaf keep the exact gather);
    - TM→octree leaves coarser than the source are overlap-volume averaged.

    Finer-or-equal targets always keep the pointwise interpolation. The
    ``om_to_om`` ``'overlap'`` kernel is inherently volume-weighted, exact
    containment gather for a target leaf inside one source leaf, conservative
    mean when coarsening, so it ignores ``conservative`` (pass
    ``method='nearest'`` for a pointwise gather instead).

    Mass preservation guarantee: over any fully-covered target region the
    projected field carries the source integral exactly
    (``Σ_T out_T·vol(T) == Σ_S f_S·vol(S ∩ region)``) and constants are
    reproduced (covered rows are a partition of unity). Partially-covered
    target cells follow ``padding``: ``'border'`` / ``'raise'`` normalize by
    the COVERED volume; the mean over the covered part, so constants survive
    at the boundary; ``'zeros'`` normalizes by the FULL target-cell volume,
    the uncovered fraction contributes 0, so the covered MASS (not the mean)
    is what is preserved. The two coincide on fully-covered cells.

    Args:
        source: mesh the input field lives on.
        target: mesh the output field is produced on.
        field_name: ModelState tensor name(s) this transform projects.
        method: projection kernel override (``None`` = auto-select by mesh
            pair: identity / bi-/trilinear / grid_sample / index-map /
            ``'idw'`` / ``'nearest'``).
        k_neighbors: neighbour count for the generic ``'idw'`` kernel.
        idw_power: inverse-distance weighting exponent.
        conservative: renormalise weights so constant fields are preserved.
        padding: out-of-domain policy, ``'raise'``, ``'border'``, or
            ``'zeros'``.
    """

    def __init__(self, source: Mesh, target: Mesh, *,
                 field_name: str | Sequence[str],
                 method: str | None = None, k_neighbors: int = 4,
                 idw_power: float = 1.0,
                 conservative: bool = True, padding: str = "raise") -> None:
        super().__init__()
        # ``method=None`` (the default) auto-picks per mode below: bilinear/
        # trilinear ``'linear'`` on structured-uniform pairs, box-overlap
        # volume weighting ``'overlap'`` on OctreeMesh↔OctreeMesh, inverse-
        # distance ``'idw'`` on the general (unstructured/non-uniform) path. An
        # explicit method is still validated and still rejected if it conflicts
        # with the resolved mode (so the zero-arg factory works for every mesh
        # pair while an unknown/mismatched explicit method fails loudly).
        # ``'barycentric'`` is opt-in ONLY (never auto-picked): linear-exact
        # centre-triangulation interpolation for an UnstructuredMesh source.
        if method is not None and method not in (
            "linear", "nearest", "idw", "overlap", "barycentric", "volume",
        ):
            raise GeoBrainError(
                "MeshProjection method must be 'linear', 'nearest', 'idw', "
                "'overlap', 'barycentric', or 'volume'",
                object_name="MeshProjection",
                field="method",
                expected="'linear', 'nearest', 'idw', 'overlap', "
                         "'barycentric', 'volume', or None (auto)",
                actual=method,
            )
        # Out-of-coverage policy for target cells that fall outside the source
        # domain: 'raise' (default: fail loudly on partial coverage), 'border'
        # (clamp/extrapolate at the source boundary, the legacy behaviour), or
        # 'zeros' (fill uncovered target cells with 0). See the per-kernel
        # coverage check at the end of __init__.
        if padding not in ("raise", "border", "zeros"):
            raise GeoBrainError(
                "MeshProjection padding must be 'raise', 'border', or 'zeros'",
                object_name="MeshProjection",
                field="padding",
                expected="'raise', 'border', or 'zeros'",
                actual=padding,
            )
        # One field name (a plain string: the common case) or a sequence of
        # names: a sequence projects EVERY named field through the SAME
        # precomputed weights, each written under its own name.
        self._field_names = _validate_field_names(field_name)
        if source.n_dim != target.n_dim:
            raise GeoBrainError(
                "source and target must have equal n_dim",
                object_name="MeshProjection",
                field="n_dim",
                expected=source.n_dim,
                actual=target.n_dim,
            )

        self.source = source
        self.target = target
        # ``field_name`` keeps the user's plain string for the single-field
        # case (heavily pinned) and the normalized tuple for a sequence.
        self.field_name = (
            field_name if isinstance(field_name, str) else self._field_names
        )
        # Conservative (mass/mean-preserving) fine->coarse resampling: when the
        # target is genuinely COARSER than the source, average the covered
        # source cells (area / overlap-volume weights, see the class docstring)
        # instead of point-sampling (which aliases and biases integral
        # quantities such as a gravity source density). Default True; set
        # False to force the legacy pointwise interpolation on every path.
        self._conservative = bool(conservative)
        self._padding = padding
        # Dispatch + precompute per-pair helpers.
        self._mode = self._select_mode(source, target)
        # Auto-pick when no method was given: 'linear' (bilinear/trilinear) on
        # the structured fast-paths, 'overlap' (box-overlap volume weights) on
        # OctreeMesh↔OctreeMesh, 'idw' (inverse-distance) on the general k-NN
        # path. 'barycentric' is NEVER auto-picked (opt-in; the general default
        # stays 'idw' so existing UnstructuredMesh pairs are untouched).
        if method is None:
            if self._mode == "om_to_om":
                method = "overlap"
            elif self._mode == "general":
                method = "idw"
            else:                     # tm fast paths, tm_linear, cyl_to_cyl
                method = "linear"
        self.method = method
        # Each method names ONE specific interpolant, so a method that does not
        # describe the resolved mode's math must fail loudly instead of being
        # silently conflated ('linear' is true bi/trilinear, 'idw' is k-NN
        # inverse distance, 'overlap' is box-overlap volume weighting,
        # 'barycentric' is centre-triangulation linear interpolation). An
        # auto-picked method never reaches these guards; only an explicit
        # mismatch does.
        if self._mode == "general":
            if method == "linear":
                raise GeoBrainError(
                    "method='linear' is bilinear/trilinear and only valid for "
                    "structured TensorMesh pairs; use method='idw' for "
                    "unstructured/non-uniform/general projection",
                    object_name="MeshProjection", field="method",
                    expected="'idw', 'nearest', or 'barycentric' on the "
                             "general path",
                    actual="'linear'",
                )
            if method == "overlap":
                raise GeoBrainError(
                    "method='overlap' is the OctreeMesh↔OctreeMesh box-overlap "
                    "kernel; the general (unstructured/non-uniform) path uses "
                    "'idw', 'nearest', or 'barycentric'",
                    object_name="MeshProjection", field="method",
                    expected="'idw', 'nearest', or 'barycentric' on the "
                             "general path",
                    actual="'overlap'",
                )
            if method == "barycentric" and not isinstance(
                source, UnstructuredMesh
            ):
                raise GeoBrainError(
                    "method='barycentric' Delaunay-triangulates the SOURCE "
                    "cell centres and needs an UnstructuredMesh source; this "
                    f"pair's source is {type(source).__name__}. Use "
                    "method='idw' or 'nearest' instead",
                    object_name="MeshProjection", field="method",
                    expected="UnstructuredMesh source for 'barycentric'",
                    actual=type(source).__name__,
                )
        elif self._mode == "cyl_to_cyl":
            if method not in ("linear", "nearest"):
                raise GeoBrainError(
                    f"method='{method}' is not valid for a "
                    "CylindricalMesh pair: both sides are rectilinear (z, r) "
                    "grids, projected with exact per-axis linear "
                    "interpolation (method='linear', the default, "
                    "conservative coarsening is ring-exact via (z, r²) "
                    "overlap) or a nearest gather (method='nearest')",
                    object_name="MeshProjection", field="method",
                    expected="'linear' or 'nearest' on a cyl↔cyl pair",
                    actual=f"'{method}'",
                )
        elif self._mode == "om_to_om":
            if method not in ("overlap", "nearest"):
                raise GeoBrainError(
                    f"method='{method}' is not valid for an "
                    "OctreeMesh↔OctreeMesh pair: both sides are axis-aligned "
                    "boxes, projected with exact box-overlap volume weights "
                    "(method='overlap', the default) or a nearest-centre "
                    "gather (method='nearest')",
                    object_name="MeshProjection", field="method",
                    expected="'overlap' or 'nearest' on an OM↔OM pair",
                    actual=f"'{method}'",
                )
        else:
            if method == "idw":
                raise GeoBrainError(
                    "method='idw' is inverse-distance weighting and only valid "
                    "for the general (unstructured/non-uniform) path; structured "
                    "TensorMesh pairs use method='linear' (bilinear/trilinear) "
                    "or 'nearest'",
                    object_name="MeshProjection", field="method",
                    expected="'linear' or 'nearest' on a structured path",
                    actual="'idw'",
                )
            if method == "volume":
                raise GeoBrainError(
                    "method='volume' is the general-path conservative "
                    "restriction; structured TensorMesh pairs already have "
                    "EXACT overlap-volume conservative coarsening built into "
                    "method='linear' (conservative=True, the default)",
                    object_name="MeshProjection", field="method",
                    expected="'linear' or 'nearest' on a structured path",
                    actual="'volume'",
                )
            if method in ("overlap", "barycentric"):
                raise GeoBrainError(
                    f"method='{method}' is not valid for a structured pair: "
                    "'overlap' is the OctreeMesh↔OctreeMesh box-overlap kernel "
                    "and 'barycentric' the UnstructuredMesh-source centre-"
                    "triangulation kernel; structured TensorMesh pairs use "
                    "method='linear' (bilinear/trilinear) or 'nearest'",
                    object_name="MeshProjection", field="method",
                    expected="'linear' or 'nearest' on a structured path",
                    actual=f"'{method}'",
                )
        self._k_neighbors = _validate_k_neighbors(k_neighbors)
        self._idw_power = _validate_idw_power(idw_power)
        # The per-mode kernel strategy: precomputed geometry lives in
        # the kernel's OWN (non-persistent) buffers: ``.to(device)`` moves it
        # through the submodule, ``state_dict`` stays empty, and each mode's
        # precompute / forward / materialization sit in one class
        # (see projection/_kernels.py). The operator keeps the public
        # surface: validation above, padding policy below, identity fast
        # path, and the legacy read-only buffer aliases (properties).
        self._kernel = projection_kernel(
            self._mode, source, target, method=self.method,
            conservative=self._conservative, padding=self._padding,
            k_neighbors=self._k_neighbors,
            idw_power=self._idw_power,
        )
        uncovered = self._kernel.uncovered
        # Boolean (n_target,) mask of target cells OUTSIDE the source domain,
        # registered (non-persistent) for the ``padding='zeros'`` masking.
        self.register_buffer("_uncovered", None, persistent=False)
        # Explicit out-of-coverage policy. ``padding='raise'`` (the default)
        # fails loudly when any target cell falls outside the source domain
        # instead of silently border-clamping / extrapolating. ``'border'`` and
        # ``'zeros'`` keep the mask for the forward pass ('zeros' zeroes those
        # cells; 'border' leaves each kernel's own clamp/extrapolate in place).
        if uncovered is not None and bool(uncovered.any()):
            frac = float(uncovered.to(torch.float64).mean())
            if self._padding == "raise":
                raise GeoBrainError(
                    "target mesh extends beyond the source domain: "
                    f"{frac:.1%} of target cells are uncovered; pass "
                    "padding='border' to clamp/extrapolate at the boundary or "
                    "padding='zeros' to fill uncovered cells with 0",
                    object_name="MeshProjection", field="padding",
                    expected="target within source coverage "
                             "(or padding='border' / 'zeros')",
                    actual=f"{frac:.1%} of target cells uncovered",
                )
            # Downgraded padding must stay LOUD: 'zeros' plants hard zeros
            # with an exactly-zero gradient (invisible to an optimizer) and
            # 'border' extrapolates: both are silent-degradation traps in
            # an inversion if forgotten. One warning at construction; the
            # mask stays public via ``uncovered`` / ``coverage_report()``.
            warnings.warn(
                f"MeshProjection: {frac:.1%} of target cells are outside "
                f"the source domain under padding='{self._padding}' "
                + ("(filled with 0, ZERO gradient; the optimizer cannot "
                   "see these cells)" if self._padding == "zeros"
                   else "(border clamp/extrapolation)")
                + "; inspect .uncovered / .coverage_report()",
                RuntimeWarning, stacklevel=3,
            )
            self._uncovered = uncovered
        # Per-instance differentiability spec: every
        # projected field is a trainable input and an output key. Assigned via
        # ``object.__setattr__``: matching the module's other bypasses of
        # ``nn.Module.__setattr__`` (e.g. ``_w_cache`` below), though a plain
        # ``self.differentiability = ...`` would behave identically here since
        # ``differentiability`` is no longer a property.
        object.__setattr__(
            self,
            "differentiability",
            DifferentiabilitySpec(
                level=DifferentiabilityLevel.FULL_AUTOGRAD,
                trainable_inputs=self._field_names,
                output_keys=self._field_names,
            ),
        )

    @staticmethod
    def _select_mode(source: Mesh, target: Mesh) -> str:
        """Pick the projection kernel from the source/target mesh capabilities.

        The structured fast-paths (bilinear ``interpolate`` / ``grid_sample``)
        require a *uniform* grid on each structured side, keyed on the
        :class:`UniformMesh` capability, which a uniform ``TensorMesh`` declares
        per-instance and a graded one does not. The OctreeMesh fast-paths exploit
        octree-specific leaf/grid structure, so they stay type-keyed.

        A TensorMesh↔TensorMesh pair that is NOT the uniform fast path (i.e. at
        least one side is a GRADED TensorMesh) is still fully rectilinear, so it
        routes to the exact ``tm_linear`` kernel (per-axis searchsorted + linear
        weights, tensor-product) rather than to the approximate k-NN ``general``
        path; bilinear/trilinear interpolation is EXACT on a linear field where
        IDW is not. An OctreeMesh↔OctreeMesh pair is two sets of axis-aligned
        boxes, so it routes to the exact ``om_to_om`` box-overlap kernel (was
        the approximate general IDW). Everything else (unstructured,
        TM↔unstructured, ...) routes to the general ``cell_centers`` kernel.
        """
        from ..cylindrical import CylindricalMesh

        s_cyl = isinstance(source, CylindricalMesh)
        t_cyl = isinstance(target, CylindricalMesh)
        if s_cyl != t_cyl:
            # A cylinder's (z, r) coordinates are NOT a Cartesian frame: r is
            # a radius and every cell a ring, so pairing with a Cartesian
            # mesh through the coordinate-blind general kernel would silently
            # interpolate r as x and ignore ring volumes. An axisymmetric
            # expansion kernel (r = sqrt(x² + y²)) is the documented
            # follow-up; until then the pair fails loudly.
            raise GeoBrainError(
                "projection between a CylindricalMesh and a Cartesian mesh "
                "is not defined (r is a radius, cells are rings); project "
                "cyl↔cyl, or use an explicit axisymmetric expansion",
                object_name="MeshProjection", field="mesh pair",
                expected="both cylindrical or both Cartesian",
                actual=f"{type(source).__name__} → {type(target).__name__}",
            )
        if s_cyl and t_cyl:
            return "cyl_to_cyl"
        su = source.declares(UniformMesh)
        tu = target.declares(UniformMesh)
        if su and tu:
            return "tm_to_tm"
        if su and isinstance(target, OctreeMesh):
            return "tm_to_om"
        if isinstance(source, OctreeMesh) and tu:
            return "om_to_tm"
        # Both octrees: exact box-overlap volume weights (leaves are
        # axis-aligned centre±half boxes on BOTH sides).
        if isinstance(source, OctreeMesh) and isinstance(target, OctreeMesh):
            return "om_to_om"
        # Both structured TensorMesh but not the uniform fast path (>=1 graded):
        # exact rectilinear linear interpolation via a tensor-product sparse W.
        if isinstance(source, TensorMesh) and isinstance(target, TensorMesh):
            return "tm_linear"
        return "general"      # any other GeometryMesh pair (US, TM↔US, ...)

    _w_cache: "torch.Tensor | None" = None  # as_sparse_matrix() lazy cache

    # ------------------------------------------------------------------
    # Read-only aliases for the legacy shared buffer names. The geometry
    # now lives in the per-mode kernel's own buffers; these properties keep
    # the long-standing introspection surface (pinned by the projection test
    # battery) working unchanged. A name a kernel does not define reads None,
    # exactly like the former always-registered-None buffers.
    # ------------------------------------------------------------------

    @property
    def _index_map(self):
        return getattr(self._kernel, "_index_map", None)

    @property
    def _grid(self):
        return getattr(self._kernel, "_grid", None)

    @property
    def _weights(self):
        return getattr(self._kernel, "_weights", None)

    @property
    def _tm_grid(self):
        return getattr(self._kernel, "_tm_grid", None)

    @property
    def _om_avg(self):
        return getattr(self._kernel, "_om_avg", None)

    @property
    def _om_coarse(self):
        return getattr(self._kernel, "_om_coarse", None)

    @property
    def _tm_ov(self):
        return getattr(self._kernel, "_tm_ov", None)

    @property
    def _tm_ov_rows(self):
        return getattr(self._kernel, "_tm_ov_rows", None)

    @property
    def _om2tm_avg(self):
        return getattr(self._kernel, "_om2tm_avg", None)

    @property
    def _om2tm_coarse(self):
        return getattr(self._kernel, "_om2tm_coarse", None)

    @property
    def mode(self) -> str:
        """The resolved kernel mode (read-only, documented contract).

        One of ``'tm_to_tm'`` (structured pair), ``'tm_to_om'`` /
        ``'om_to_tm'`` (structured/octree), ``'om_to_om'`` (octree box
        overlap), ``'tm_linear'`` (graded structured pair) or ``'general'``
        (k-NN kernel on cell centres). The public form of what the examples
        previously read off the private ``_mode``.
        """
        return self._mode

    def project(self, field: torch.Tensor) -> torch.Tensor:
        """Project ONE tensor from the source mesh onto the target mesh.

        The standalone convenience for scripts and diagnostics, no
        ``ModelState`` / ``ForwardContext`` wrapping. Autograd flows through
        exactly as in the operator route (identical kernel); identity
        projections return ``field`` unchanged. For pipeline use, keep the
        operator call convention.
        """
        if self._is_identity():
            return field
        return self._project_field("field", field)

    def as_sparse_matrix(self) -> torch.Tensor:
        """The projection as a cached sparse ``(n_target, n_source)`` matrix.

        ``W @ field.reshape(-1) == self.project(field).reshape(-1)`` for every
        kernel (including conservative blends and ``padding='zeros'`` row
        masking), useful for building regularization / adjoint operators.
        Built once on first call (float64, on the kernel buffers' device at
        that moment) and cached; the fast forward kernels never touch it.
        """
        if self._w_cache is None:
            object.__setattr__(self, "_w_cache", self._materialize_weights())
        return self._w_cache

    def _is_identity(self) -> bool:
        """True when source and target are the same TensorMesh (no-op projection)."""
        if self._mode != "tm_to_tm":
            return False
        return self.source == self.target

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        """
        Project every named field from the source mesh onto the target mesh.

        Args:
            state: Must contain each projected field shaped like the source
                mesh.
            ctx: Unused; present for the operator contract.

        Returns:
            A new :class:`ModelState` with each projected field reshaped to
            the target mesh (other entries pass through untouched). Identity
            projections return ``state`` unchanged.
        """
        fields = state.fetch(*self._field_names)
        if self._is_identity():
            return state
        return state.with_tensors(**{
            name: self._project_field(name, field)
            for name, field in zip(self._field_names, fields)
        })

    def _project_field(self, name: str, field: torch.Tensor) -> torch.Tensor:
        """Validate + project ONE field (or a LEADING-dim batch of fields).

        Extra leading dimensions: an SVGD particle ensemble
        ``(n_particles, *src_shape)``, a channel stack, …; are projected in
        one shot through the kernel's batched path (single sparse.mm on the
        stored-``W`` kernels; per-slice loop fallback elsewhere) and returned
        with the same leading dims.
        """
        if field.is_complex():
            raise GeoBrainError(
                "MeshProjection supports real fields only: every kernel "
                "computes in float64, and casting a complex field would "
                "silently discard its imaginary part. Project the real and "
                "imaginary parts as two separate fields.",
                object_name="MeshProjection", field=name,
                expected="a real-valued field", actual=str(field.dtype),
            )
        batch_shape: tuple[int, ...] | None
        if self._kernel.takes_grid_input:
            # Grid-source kernels (tm_to_tm / tm_to_om / tm_linear /
            # cyl_to_cyl): the source carries a structured (nz, nx[, ny]) /
            # (nz, nr) layout to validate against.
            from ..cylindrical import CylindricalMesh
            assert isinstance(self.source, (TensorMesh, CylindricalMesh))
            src_shape = tuple(self.source.shape)
            nd = len(src_shape)
            if tuple(field.shape) == src_shape:
                batch_shape = None
            elif field.dim() > nd and tuple(field.shape[-nd:]) == src_shape:
                batch_shape = tuple(field.shape[:-nd])
            else:
                raise GeoBrainError(
                    "field shape mismatches source mesh",
                    object_name="MeshProjection",
                    field=name,
                    expected=f"{src_shape} or (batch..., *{src_shape})",
                    actual=tuple(field.shape),
                )
            single_in: tuple[int, ...] = src_shape
        else:
            # A flat-source kernel (octree / unstructured source): a flat
            # vector, a structured source's own grid layout, or either with
            # extra leading batch dims.
            n = int(self.source.n_cells)
            grid = (tuple(self.source.shape)
                    if isinstance(self.source, TensorMesh) else None)
            if int(field.numel()) == n:
                batch_shape = None
            elif (grid is not None and field.dim() > len(grid)
                  and tuple(field.shape[-len(grid):]) == grid):
                batch_shape = tuple(field.shape[:-len(grid)])
            elif field.dim() >= 2 and int(field.shape[-1]) == n:
                batch_shape = tuple(field.shape[:-1])
            else:
                raise GeoBrainError(
                    "field shape mismatches source mesh n_cells",
                    object_name="MeshProjection", field=name,
                    expected=(n,), actual=tuple(field.shape),
                )
            single_in = (n,)

        if batch_shape is None:
            projected = self._kernel.apply(field)
        else:
            b = 1
            for s in batch_shape:
                b *= int(s)
            out = self._kernel.apply_batched(field.reshape((b,) + single_in))
            projected = out.reshape(batch_shape + tuple(out.shape[1:]))
        # padding='zeros': overwrite uncovered target cells with 0. A 0/1 mask
        # multiply is differentiable (zero gradient through the masked cells)
        # and device-correct (``self._uncovered`` is a registered buffer);
        # for a batch the mask broadcasts over the leading dims.
        if self._padding == "zeros" and self._uncovered is not None:
            single_out = (projected.shape if batch_shape is None
                          else projected.shape[len(batch_shape):])
            keep = (~self._uncovered).to(dtype=projected.dtype)
            projected = projected * keep.reshape(single_out)
        return projected

    @property
    def uncovered(self) -> "torch.Tensor | None":
        """Public read-only ``(n_target,)`` bool mask of uncovered target
        cells (``None`` when the pair is fully covered). Inversion drivers
        should freeze or regularize these cells deliberately rather than let
        ``padding='zeros'``'s zero-gradient hide them."""
        if self._uncovered is None:
            return None
        return self._uncovered.clone()

    def coverage_report(self) -> dict:
        """``{n_target, n_uncovered, uncovered_fraction, padding}`` summary."""
        n_t = int(self.target.n_cells)
        n_unc = 0 if self._uncovered is None else int(self._uncovered.sum())
        return {
            "n_target": n_t,
            "n_uncovered": n_unc,
            "uncovered_fraction": n_unc / n_t if n_t else 0.0,
            "padding": self._padding,
        }

    def transpose(self) -> "_MeshProjectionAdjoint":
        """Return the exact adjoint (transpose ``Wᵀ``) of this projection.

        Available for EVERY kernel. Each projection is a linear operator; the
        sparse-weight kernels (``general`` / ``tm_linear`` / ``om_to_om``)
        transpose their stored ``(n_target, n_source)`` matrix ``W`` directly, and the
        grid_sample / index-map kernels (``tm_to_tm`` / ``tm_to_om`` /
        ``om_to_tm``) materialize an equivalent sparse ``W`` on demand from the
        same precomputed geometry (see :meth:`_materialize_weights`; the
        forward paths themselves are untouched). The returned operator maps a
        TARGET-shaped field to a SOURCE-shaped one via ``Wᵀ``, the forward's
        own autograd backward, and the true mathematical adjoint. This is NOT
        the same as ``self.target.projection_to(self.source, ...)``, which
        would build the (different) ``B → A`` interpolant rather than ``Wᵀ``.
        """
        return _MeshProjectionAdjoint(self)

    def _materialize_weights(self) -> torch.Tensor:
        """Sparse ``(n_target, n_source)`` float64 ``W`` equivalent to
        :meth:`_forward`: ``W @ field.flatten() == forward(field).flatten()``
        for every kernel (including the conservative blends and the
        ``padding='zeros'`` row masking). Built on demand, the forward paths
        keep their fast kernels and never touch this matrix.
        """
        n_s = int(self.source.n_cells)
        if self._is_identity():
            idx = torch.arange(n_s)
            return torch.sparse_coo_tensor(
                torch.stack([idx, idx]),
                torch.ones(n_s, dtype=torch.float64),
                size=(n_s, n_s), dtype=torch.float64,
            ).coalesce()
        W = self._kernel.materialize_W()
        if self._padding == "zeros" and self._uncovered is not None:
            W = _drop_rows(W, self._uncovered)
        return W.coalesce()


class _MeshProjectionAdjoint(PropertyTransform):
    """The exact transpose ``Wᵀ`` of a :class:`MeshProjection`.

    Returned by :meth:`MeshProjection.transpose` for EVERY kernel: the
    ``general`` / ``tm_linear`` / ``om_to_om`` modes transpose their stored sparse ``W``; the
    grid_sample / index-map kernels first materialize an equivalent sparse ``W``
    (``MeshProjection._materialize_weights``). Applying it to a TARGET-shaped
    field yields ``Wᵀ @ field`` shaped like the SOURCE mesh: the mathematical
    adjoint of the forward interpolant (identical to the forward's own autograd
    backward). Autograd flows through the constant sparse ``Wᵀ`` unchanged.
    """

    def __init__(self, forward: "MeshProjection") -> None:
        super().__init__()
        self.field_name = forward.field_name
        self._field_names = forward._field_names
        # The adjoint maps a target-shaped field back to a source-shaped one, so
        # its source/target are the forward's swapped.
        self.source = forward.target
        self.target = forward.source
        # Store Wᵀ as a (non-persistent) buffer so ``.to(device)`` moves it with
        # the module, mirroring the forward's registered kernels.
        self.register_buffer(
            "_weights_t",
            forward._materialize_weights().t().coalesce(),
            persistent=False,
        )
        object.__setattr__(
            self, "differentiability",
            DifferentiabilitySpec(
                level=DifferentiabilityLevel.FULL_AUTOGRAD,
                trainable_inputs=self._field_names,
                output_keys=self._field_names,
            ),
        )
        # Plain (unregistered) back-reference so ``transpose()`` can return
        # the forward without re-materializing anything. ``object.__setattr__``
        # bypasses nn.Module's auto-submodule-registration for a Module-typed
        # value (``forward`` is itself an nn.Module): this back-reference must
        # NOT become a registered submodule (would double-count parameters /
        # appear in state_dict).
        object.__setattr__(self, "_forward_projection", forward)

    def transpose(self) -> "MeshProjection":
        """The adjoint's adjoint: the original forward projection."""
        return self._forward_projection

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        fields = state.fetch(*self._field_names)
        updates: dict[str, torch.Tensor] = {}
        for name, field in zip(self._field_names, fields):
            if field.is_complex():
                raise GeoBrainError(
                    "MeshProjection supports real fields only: the float64 "
                    "compute would silently discard the imaginary part.",
                    object_name="MeshProjection.transpose", field=name,
                    expected="a real-valued field", actual=str(field.dtype),
                )
            n = int(self.source.n_cells)
            # Leading batch dims are supported exactly like the forward
            # projection: a flat (n,) field, or (batch..., n): one shared
            # sparse.mm over the flattened batch.
            if int(field.numel()) == n:
                batch_shape: tuple[int, ...] | None = None
            elif field.dim() >= 2 and int(field.shape[-1]) == n:
                batch_shape = tuple(field.shape[:-1])
            else:
                raise GeoBrainError(
                    "field shape mismatches the adjoint source (forward target) mesh",
                    object_name="MeshProjection.transpose", field=name,
                    expected=f"({n},) or (batch..., {n})",
                    actual=tuple(field.shape),
                )
            if batch_shape is None:
                src_flat = field.reshape(-1).to(torch.float64)
                out = torch.sparse.mm(self._weights_t, src_flat.unsqueeze(1)).reshape(-1)
                out = out.to(field.dtype)
                if isinstance(self.target, TensorMesh):
                    out = out.reshape(self.target.shape)
            else:
                flat = field.reshape(-1, n).to(torch.float64)      # (B, n)
                out = torch.sparse.mm(self._weights_t, flat.T).T    # (B, m)
                out = out.to(field.dtype)
                tgt_shape = (tuple(self.target.shape)
                             if isinstance(self.target, TensorMesh)
                             else (out.shape[-1],))
                out = out.reshape(batch_shape + tgt_shape)
            updates[name] = out
        return state.with_tensors(**updates)


# ----------------------------------------------------------------------
# Cross-mesh-type helpers
# ----------------------------------------------------------------------
