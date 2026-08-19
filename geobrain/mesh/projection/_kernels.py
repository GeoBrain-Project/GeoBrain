"""Per-mode projection kernels: the strategy layer behind :class:`MeshProjection`.

Each kernel is a small :class:`torch.nn.Module` owning exactly the buffers its
mode needs (registered non-persistent, so ``.to(device)`` moves them and
``state_dict`` stays empty, the same contract the operator's former shared
buffer namespace had). A kernel:

- precomputes its geometry in ``__init__`` (and leaves the pair's
  out-of-coverage mask in ``self.uncovered`` for the operator's padding
  policy);
- projects one field via :meth:`apply` (autograd flows through: all
  arithmetic is plain torch ops on constant geometry);
- materializes the equivalent sparse ``(n_target, n_source)`` matrix via
  :meth:`materialize_W` (pre-padding; the operator applies the
  ``padding='zeros'`` row masking and the identity fast path).

Adding a mesh-pair kernel = adding one class here + one ``_KERNELS`` entry;
the operator's validation/padding/dispatch surface does not change. The
bodies preserve the original per-mode methods verbatim (byte-identical
math; the projection test battery pins it).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import warnings

import torch

from ..cylindrical import CylindricalMesh
from ..tensor import TensorMesh
from ._builders import (
    _build_barycentric_weights,
    _build_general_weights,
    _build_om_to_om_overlap,
    _build_om_to_tm_conservative,
    _build_om_to_tm_index_map,
    _build_rect_source_overlap,
    _build_tm_linear_weights,
    _build_tm_to_om_conservative,
    _build_tm_to_om_grid,
    _build_tm_to_tm_grid,
    _build_volume_weights,
    _cyl_axis_edges_sq,
    _cyl_cell_boxes_sq,
    _general_uncovered,
    _source_domain_bounds,
    _grid_sample_weight_matrix,
    _grid_uncovered,
    _interpolate_weight_matrix,
    _nearest_center_rows,
    _rect_target_coarser,
    _replace_rows,
    _tm_axis_edges,
    _tm_cell_boxes,
    _tm_extents_match,
    _tm_origins_match,
    _uniform_target_coarser,
)

__all__ = ["projection_kernel"]


class _KernelBase(torch.nn.Module):
    """Shared kernel plumbing: mesh refs, method/conservative flags."""

    #: does :meth:`apply` take the source's grid layout (``mesh.shape``)
    #: rather than a flat ``(n_cells,)`` vector?
    takes_grid_input: bool = False

    def __init__(self, source, target, *, method: str, conservative: bool,
                 padding: str, k_neighbors: int, idw_power: float = 1.0) -> None:
        super().__init__()
        self.source = source
        self.target = target
        self.method = method
        self._conservative = bool(conservative)
        self._padding = padding
        self._k_neighbors = int(k_neighbors)
        self._idw_power = float(idw_power)
        #: (n_target,) bool mask of uncovered target cells, or ``None``,
        #: consumed once by the operator's padding policy.
        self.uncovered: torch.Tensor | None = None

    def apply(self, field: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def apply_batched(self, fields: torch.Tensor) -> torch.Tensor:
        """Project a ``(B, *single_layout)`` stack; default = per-slice loop.

        ``single_layout`` is the source grid shape for grid-input kernels and
        ``(n_cells,)`` otherwise. Sparse-W kernels override with one
        sparse.mm; the loop fallback keeps every kernel batch-correct.
        """
        return torch.stack([self.apply(f) for f in fields])

    def materialize_W(self) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class _TmToTmKernel(_KernelBase):
    """Uniform TensorMesh pair: index-space ``F.interpolate`` fast path on an
    equal-extent/equal-origin pair, physically-anchored ``grid_sample``
    otherwise, with exact overlap-volume weights for conservative anchored
    coarsening."""

    takes_grid_input = True

    def __init__(self, source, target, **kw) -> None:
        super().__init__(source, target, **kw)
        self.register_buffer("_tm_grid", None, persistent=False)
        self.register_buffer("_tm_ov", None, persistent=False)
        self.register_buffer("_tm_ov_rows", None, persistent=False)
        # The index-space fast path is only physically correct when the two
        # meshes span the SAME domain: equal extents AND equal origins.
        if not (_tm_extents_match(source, target)
                and _tm_origins_match(source, target)):
            self._tm_grid = _build_tm_to_tm_grid(source, target)
            self.uncovered = _grid_uncovered(self._tm_grid)
            # Conservative anchored coarsening: a COARSER uniform target must
            # volume-average the source cells each target cell overlaps.
            if (self._conservative and self.method == "linear"
                    and _uniform_target_coarser(source, target)):
                tgt_lo, tgt_hi = _tm_cell_boxes(target)
                ov, ov_rows = _build_rect_source_overlap(
                    _tm_axis_edges(source), tgt_lo, tgt_hi,
                    full_volume_norm=(self._padding == "zeros"),
                )
                self._tm_ov = ov
                if not bool(ov_rows.all()):
                    self._tm_ov_rows = ov_rows
        else:
            # Equal-extent, equal-origin pair: fully covered by construction.
            self.uncovered = torch.zeros(target.n_cells, dtype=torch.bool)

    def _grid_point_sample(self, field: torch.Tensor) -> torch.Tensor:
        # Unequal physical extents: sample the source field at the target
        # cell centres' PHYSICAL positions. grid_sample requires input and
        # grid to share a dtype; the grid is float64 geometry, so match it.
        inp = field.to(dtype=self._tm_grid.dtype).unsqueeze(0).unsqueeze(0)
        mode = "bilinear" if self.method == "linear" else "nearest"
        y = torch.nn.functional.grid_sample(
            inp, self._tm_grid, mode=mode,
            padding_mode="border", align_corners=False,
        )
        return y.squeeze(0).squeeze(0).to(dtype=field.dtype)

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        if self._tm_grid is not None:
            if self._tm_ov is not None:
                # Conservative anchored coarsening: exact overlap-volume
                # weighted restriction (constant sparse matrix: plain linear
                # op, autograd flows to every covered source cell). Target
                # cells with NO source overlap keep the grid_sample border
                # value.
                src_flat = field.reshape(-1).to(dtype=self._tm_ov.dtype)
                y = torch.sparse.mm(
                    self._tm_ov, src_flat.unsqueeze(1)
                ).reshape(-1)
                if self._tm_ov_rows is not None:
                    y_gs = self._grid_point_sample(field)
                    y = torch.where(
                        self._tm_ov_rows, y, y_gs.reshape(-1).to(dtype=y.dtype)
                    )
                return y.reshape(self.target.shape).to(dtype=field.dtype)
            return self._grid_point_sample(field)
        src_shape = tuple(self.source.shape)
        tgt_shape = tuple(self.target.shape)
        # Conservative coarsening: pure downsampling with the linear method
        # uses 'area' (adaptive average pooling) so the cell mean, and hence
        # the integral: is preserved instead of aliased by a point sample.
        is_downsampling = (
            all(t <= s for t, s in zip(tgt_shape, src_shape))
            and tgt_shape != src_shape
        )
        # Uniform dtype policy: compute float64, cast back (no-op for the
        # pinned float64 majority: bit-for-bit unchanged).
        x = field.to(dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        if self._conservative and self.method == "linear" and is_downsampling:
            y = torch.nn.functional.interpolate(x, size=tgt_shape, mode="area")
            return y.squeeze(0).squeeze(0).to(dtype=field.dtype)
        if self.source.n_dim == 2:
            mode = "bilinear" if self.method == "linear" else "nearest"
        else:
            mode = "trilinear" if self.method == "linear" else "nearest"
        interp_kwargs = {"size": tgt_shape, "mode": mode}
        if mode in ("bilinear", "trilinear"):
            interp_kwargs["align_corners"] = False
        y = torch.nn.functional.interpolate(x, **interp_kwargs)
        return y.squeeze(0).squeeze(0).to(dtype=field.dtype)

    def materialize_W(self) -> torch.Tensor:
        if self._tm_grid is not None:                      # anchored pair
            if self._tm_ov is not None and self._tm_ov_rows is None:
                return self._tm_ov                         # fully overlapped
            W = _grid_sample_weight_matrix(
                self._tm_grid, tuple(self.source.shape), method=self.method,
            )
            if self._tm_ov is not None:
                assert self._tm_ov_rows is not None
                W = _replace_rows(W, self._tm_ov, self._tm_ov_rows)
            return W
        return _interpolate_weight_matrix(                 # equal-extent pair
            tuple(self.source.shape), tuple(self.target.shape),
            method=self.method, conservative=self._conservative,
        )


class _TmToOmKernel(_KernelBase):
    """Uniform TensorMesh → OctreeMesh: ``grid_sample`` at the leaf centres,
    with overlap-volume averaging for leaves coarser than the source."""

    takes_grid_input = True

    def __init__(self, source, target, **kw) -> None:
        super().__init__(source, target, **kw)
        self.register_buffer("_grid", None, persistent=False)
        self.register_buffer("_om_avg", None, persistent=False)
        self.register_buffer("_om_coarse", None, persistent=False)
        self._grid = _build_tm_to_om_grid(source, target)
        self.uncovered = _grid_uncovered(self._grid)
        if self._conservative:
            # Coarse leaves are volume-averaged instead of point-sampled;
            # the builder returns (None, None) when there are none, keeping
            # the pure point-sample fast path byte-identical.
            avg, coarse = _build_tm_to_om_conservative(
                source, target, padding=self._padding,
            )
            self._om_avg = avg
            self._om_coarse = coarse

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        # The two unsqueezes give the (N=1, C=1, ...) input grid_sample
        # wants; the grid was built with the matching rank (4-D for 2-D,
        # 5-D for 3-D), so one call dispatches both dimensionalities.
        assert self._grid is not None
        inp = field.to(dtype=self._grid.dtype).unsqueeze(0).unsqueeze(0)
        mode = "bilinear" if self.method == "linear" else "nearest"
        y = torch.nn.functional.grid_sample(
            inp, self._grid, mode=mode,
            padding_mode="border", align_corners=False,
        )
        y = y.reshape(-1)                                  # (n_leaves,) f64
        # Conservative coarsening: overwrite COARSE leaves' point-samples
        # with the volume-weighted mean of the source cells they cover.
        if self._om_coarse is not None and self._om_avg is not None:
            src_flat = field.reshape(-1).to(dtype=self._om_avg.dtype)
            y_avg = torch.sparse.mm(
                self._om_avg, src_flat.unsqueeze(1)
            ).reshape(-1)
            y = torch.where(self._om_coarse, y_avg, y)
        return y.to(dtype=field.dtype)

    def materialize_W(self) -> torch.Tensor:
        assert self._grid is not None
        W = _grid_sample_weight_matrix(
            self._grid, tuple(self.source.shape), method=self.method,
        )
        if self._om_coarse is not None and self._om_avg is not None:
            W = _replace_rows(W, self._om_avg, self._om_coarse)
        return W


class _OmToTmKernel(_KernelBase):
    """OctreeMesh → uniform TensorMesh: precomputed containing-leaf gather,
    with overlap-volume means for cells straddling ≥ 2 leaves."""

    takes_grid_input = False

    def __init__(self, source, target, **kw) -> None:
        super().__init__(source, target, **kw)
        self.register_buffer("_index_map", None, persistent=False)
        self.register_buffer("_om2tm_avg", None, persistent=False)
        self.register_buffer("_om2tm_coarse", None, persistent=False)
        self._index_map, self.uncovered = _build_om_to_tm_index_map(
            source, target)
        if self._conservative:
            # A TM cell covering ≥ 2 leaves takes the overlap-volume mean;
            # single-leaf cells keep the byte-identical gather. The builder
            # returns (None, None) when no cell straddles.
            avg, coarse = _build_om_to_tm_conservative(
                source, target, padding=self._padding,
            )
            self._om2tm_avg = avg
            self._om2tm_coarse = coarse

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        assert self._index_map is not None
        # An octree field is an intrinsic flat (n_cells,) leaf vector;
        # flatten defensively so the gather stays correct for any layout.
        flat = field.reshape(-1)
        gathered = flat[self._index_map]
        if self._om2tm_coarse is not None and self._om2tm_avg is not None:
            src = flat.to(dtype=self._om2tm_avg.dtype)
            avg = torch.sparse.mm(
                self._om2tm_avg, src.unsqueeze(1)
            ).reshape(-1)
            gathered = torch.where(
                self._om2tm_coarse, avg.to(dtype=gathered.dtype), gathered
            )
        return gathered.reshape(self.target.shape)

    def materialize_W(self) -> torch.Tensor:
        assert self._index_map is not None
        n_t = int(self.target.n_cells)
        n_s = int(self.source.n_cells)
        device = self._index_map.device
        rows = torch.arange(n_t, device=device)
        W = torch.sparse_coo_tensor(
            torch.stack([rows, self._index_map]),
            torch.ones(n_t, dtype=torch.float64, device=device),
            size=(n_t, n_s), dtype=torch.float64,
        ).coalesce()
        if self._om2tm_coarse is not None and self._om2tm_avg is not None:
            W = _replace_rows(W, self._om2tm_avg, self._om2tm_coarse)
        return W


class _SparseWKernel(_KernelBase):
    """The stored-``W`` family: graded structured pairs (``tm_linear``),
    cylindrical pairs (``cyl_to_cyl``, same rectilinear machinery, with
    ring-exact conservative coarsening in the ``(z, r²)`` measure frame),
    octree pairs (``om_to_om`` box overlap / nearest gather) and the general
    cell-centre kernel (``idw`` / ``nearest`` / opt-in ``barycentric`` /
    conservative ``volume`` restriction)."""

    def __init__(self, source, target, *, mode: str, **kw) -> None:
        super().__init__(source, target, **kw)
        self._sparse_mode = mode
        self.takes_grid_input = mode in ("tm_linear", "cyl_to_cyl")
        self.register_buffer("_weights", None, persistent=False)
        method = self.method
        if mode == "cyl_to_cyl":
            # Exact per-axis rectilinear interpolation on (z, r): the cyl
            # mesh carries the full structured surface (shape / cell_widths /
            # origin), so the tm_linear builder applies verbatim.
            self._weights, self.uncovered = _build_tm_linear_weights(
                source, target, method=method, padding=self._padding,
            )
            if (self._conservative and method == "linear"
                    and _rect_target_coarser(source, target)):
                # Ring-exact conservative coarsening: rectangular overlap
                # in the (z, r²) frame, whose box volume IS the ring volume
                # π·Δz·Δ(r²) (π cancels in the normalisation).
                tgt_lo, tgt_hi = _cyl_cell_boxes_sq(target)
                ov, ov_rows = _build_rect_source_overlap(
                    _cyl_axis_edges_sq(source), tgt_lo, tgt_hi,
                    full_volume_norm=(self._padding == "zeros"),
                )
                self._weights = _replace_rows(self._weights, ov, ov_rows)
        elif mode == "tm_linear":
            self._weights, self.uncovered = _build_tm_linear_weights(
                source, target, method=method, padding=self._padding,
            )
            # Conservative coarsening on the graded rectilinear path: replace
            # the point-interpolating rows of a genuinely-coarser target with
            # exact overlap-volume weights; no-overlap cells keep their
            # border-extrapolating linear rows.
            if (self._conservative and method == "linear"
                    and _rect_target_coarser(source, target)):
                tgt_lo, tgt_hi = _tm_cell_boxes(target)
                ov, ov_rows = _build_rect_source_overlap(
                    _tm_axis_edges(source), tgt_lo, tgt_hi,
                    full_volume_norm=(self._padding == "zeros"),
                )
                self._weights = _replace_rows(self._weights, ov, ov_rows)
        elif mode == "om_to_om":
            if method == "overlap":
                # Exact box-overlap volume weights with the established
                # padding×normalization semantics.
                W, covered = _build_om_to_om_overlap(
                    source, target,
                    full_volume_norm=(self._padding == "zeros"),
                )
                self.uncovered = ~covered
                if self._padding == "border" and bool(self.uncovered.any()):
                    # 'border': zero-overlap leaves clamp to the NEAREST
                    # source leaf's value.
                    W = _replace_rows(
                        W,
                        _nearest_center_rows(
                            source.cell_centers(), target.cell_centers(),
                            self.uncovered,
                        ),
                        self.uncovered,
                    )
                self._weights = W
            else:
                # method='nearest': the legacy nearest-centre gather.
                self._weights = _build_general_weights(
                    source.cell_centers(), target.cell_centers(),
                    method="nearest", k_neighbors=1,
                )
                self.uncovered = _general_uncovered(
                    source.cell_centers(), target.cell_centers(),
                    bounds=_source_domain_bounds(source),
                )
        else:                                              # general
            if method == "volume":
                # Conservative volume-binned restriction (reverse-nearest
                # partition of the source; see _build_volume_weights).
                self._weights = _build_volume_weights(
                    source.cell_centers(), target.cell_centers(),
                    source.cell_volumes(),
                )
            elif method == "barycentric":
                self._weights = _build_barycentric_weights(
                    source.cell_centers(), target.cell_centers(),
                    k_neighbors=self._k_neighbors,
                )
            else:
                self._weights = _build_general_weights(
                    source.cell_centers(), target.cell_centers(),
                    method=method, k_neighbors=self._k_neighbors,
                    idw_power=self._idw_power,
                )
                if self._conservative:
                    # conservative=True has never had an effect on the k-NN
                    # general path (point interpolation only). When the
                    # target genuinely coarsens, say so instead of silently
                    # aliasing integral quantities.
                    src_v = source.cell_volumes().reshape(-1)
                    tgt_v = target.cell_volumes().reshape(-1)
                    if float(tgt_v.mean()) > 2.0 * float(src_v.mean()):
                        warnings.warn(
                            "MeshProjection: conservative=True has no effect "
                            "on the general k-NN path; this looks like a "
                            "coarsening (mean target cell volume > 2x "
                            "source); pass method='volume' for a "
                            "volume-binned conservative restriction",
                            RuntimeWarning, stacklevel=4,
                        )
            # Coverage: true cell-hull domain for box-carrying sources,
            # centre-bbox fallback for centroid clouds.
            self.uncovered = _general_uncovered(
                source.cell_centers(), target.cell_centers(),
                bounds=_source_domain_bounds(source),
            )

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        assert self._weights is not None
        src_flat = field.reshape(-1).to(torch.float64)
        out = torch.sparse.mm(self._weights, src_flat.unsqueeze(1)).reshape(-1)
        out = out.to(field.dtype)
        if isinstance(self.target, (TensorMesh, CylindricalMesh)):
            return out.reshape(self.target.shape)
        return out

    def apply_batched(self, fields: torch.Tensor) -> torch.Tensor:
        assert self._weights is not None
        b = int(fields.shape[0])
        flat = fields.reshape(b, -1).to(torch.float64)     # (B, n_src)
        out = torch.sparse.mm(self._weights, flat.T).T     # (B, n_tgt)
        out = out.to(fields.dtype)
        if isinstance(self.target, (TensorMesh, CylindricalMesh)):
            return out.reshape((b,) + tuple(self.target.shape))
        return out

    def materialize_W(self) -> torch.Tensor:
        assert self._weights is not None
        return self._weights


def projection_kernel(mode: str, source, target, *, method: str,
                      conservative: bool, padding: str,
                      k_neighbors: int, idw_power: float = 1.0) -> _KernelBase:
    """Build the kernel strategy for a resolved ``mode`` (closed world;
    ``MeshProjection._select_mode`` is the single naming authority)."""
    kw = dict(method=method, conservative=conservative, padding=padding,
              k_neighbors=k_neighbors, idw_power=idw_power)
    if mode == "tm_to_tm":
        return _TmToTmKernel(source, target, **kw)
    if mode == "tm_to_om":
        return _TmToOmKernel(source, target, **kw)
    if mode == "om_to_tm":
        return _OmToTmKernel(source, target, **kw)
    return _SparseWKernel(source, target, mode=mode, **kw)
