"""
Image Quilting MPS simulation (Mariethoz & Caers 2014, adapted from Efros 2001).

Patch-based simulator: tiles cut from the training image are stitched into the
target grid with overlapping seams. The overlap region of each new tile is
matched to the already-pasted neighbourhood by squared-error, then blended
along a minimum-error seam.

The overlap seam uses a 2-D **graph-cut** based on
``scipy.sparse.csgraph.maximum_flow``. For each z-slice of the overlap
region we build a 4-connected grid graph; node weight = squared
difference between the already-pasted ``sim_region`` and the candidate
``new_tile``; source = pixels on the "old" boundary; sink = pixels on
the "new" boundary; a min-cut partitions the overlap into pixels kept
from old vs new along a smoother 2-D seam than the 1-D DP. Falls back
to the 1-D DP if the graph would degenerate (no candidate source/sink).

References:
- Efros, A. & Freeman, W. (2001). Image quilting for texture synthesis and
  transfer. *SIGGRAPH 2001*.
- Kwatra, V., Schödl, A., Essa, I., Turk, G. & Bobick, A. (2003).
  Graphcut textures: Image and video synthesis using graph cuts.
  *SIGGRAPH 2003*.
- Mariethoz, G. & Caers, J. (2014). *Multiple-Point Geostatistics: Stochastic
  Modelling with Training Images*. Wiley.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from functools import partial
from typing import cast

import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from ....core import GeoBrainError
from ...frames._arrays import FloatArray, as_float_array
from ._grid_utils import fill_hard_data
from .._domain import (
    derive_realization_seeds as _derive_realization_seeds,
)
from ...frames import GeoFrame, GeoGrid, PropertyMetadata, gslib_grid_layout
from ._parallel import run_realizations
from .execution import SimulationExecutionConfig
from .agent_contract import SimulationAgentContract
from .results import SimulationEnsemble
from .training_image_index import TrainingImageIndex, TrainingImageSpec, assemble_mps_ensemble
from .sequential import make_simulation_frame

__all__ = ["ImageQuilting"]


def _image_quilting_realization(
    seed: int,
    *,
    simulator: "ImageQuilting",
    target_shape: tuple[int, int, int],
    pre_filled: np.ndarray,
) -> np.ndarray:
    field = simulator._one_realisation(
        target_shape,
        np.random.default_rng(int(seed)),
    )
    mask = np.isfinite(pre_filled)
    field[mask] = pre_filled[mask]
    return as_float_array(field.transpose(2, 1, 0).reshape(-1))


class ImageQuilting(SimulationAgentContract):
    """
    Image-Quilting MPS simulator.

    Args:
        training_image: 2-D ``(nx, ny)`` or 3-D ``(nx, ny, nz)`` numpy array.
        tile_size: ``(tx, ty)`` or ``(tx, ty, tz)`` tile dimensions in cells.
        overlap: overlap fraction between adjacent tiles (in ``(0, 0.5]``).
            Default ``1/6``.
        tolerance: pattern-matching tolerance, tiles within
            ``(1+tolerance) * best_dist`` of the best are eligible. Default
            ``0.1``.
        n_candidates: number of random TI tile candidates to score per
            placement. Default ``512``.
        path: ``"raster"`` (default) or ``"random"`` placement order.
        nsim: number of realisations.
        seed: master seed (per-realisation seeds derived from it).
        n_jobs: realisation workers. ``1`` (default) serial; ``>1`` / ``-1`` /
            ``"auto"`` run realisations across processes. Results are
            identical to the serial run for any ``n_jobs`` (per-realisation
            seeds are fixed and order is preserved).
    """

    def __init__(
        self,
        training_image: np.ndarray,
        tile_size: tuple[int, ...],
        *,
        property: PropertyMetadata,
        execution: SimulationExecutionConfig,
        overlap: float = 1.0 / 6.0,
        tolerance: float = 0.1,
        n_candidates: int = 512,
        path: str = "raster",
    ) -> None:
        ti = as_float_array(training_image)
        if ti.ndim not in (2, 3):
            raise GeoBrainError(
                "training_image must be 2-D or 3-D",
                object_name="ImageQuilting",
                field="training_image",
                expected="ndim 2 or 3",
                actual=ti.ndim,
            )
        if ti.ndim == 2:
            ti = as_float_array(ti[..., None])
        if any(s <= 0 for s in ti.shape):
            raise GeoBrainError(
                "training_image must have positive dimensions",
                object_name="ImageQuilting",
                field="training_image",
                expected="all dims > 0",
                actual=ti.shape,
            )

        ts = tuple(int(v) for v in tile_size)
        if len(ts) not in (2, 3):
            raise GeoBrainError(
                "tile_size must be 2-tuple or 3-tuple",
                object_name="ImageQuilting",
                field="tile_size",
                expected="len 2 or 3",
                actual=len(ts),
            )
        if any(t < 2 for t in ts):
            raise GeoBrainError(
                "tile dimensions must be >= 2",
                object_name="ImageQuilting",
                field="tile_size",
                expected=">= 2",
                actual=ts,
            )
        if len(ts) == 2:
            ts = (ts[0], ts[1], 1)
        if not (0.0 < overlap <= 0.5):
            raise GeoBrainError(
                "overlap must be in (0, 0.5]",
                object_name="ImageQuilting",
                field="overlap",
                expected="(0, 0.5]",
                actual=overlap,
            )
        if tolerance < 0.0:
            raise GeoBrainError(
                "tolerance must be >= 0",
                object_name="ImageQuilting",
                field="tolerance",
                expected=">= 0",
                actual=tolerance,
            )
        if n_candidates < 1:
            raise GeoBrainError(
                "n_candidates must be >= 1",
                object_name="ImageQuilting",
                field="n_candidates",
                expected=">= 1",
                actual=n_candidates,
            )
        if path not in ("raster", "random"):
            raise GeoBrainError(
                "path must be 'raster' or 'random'",
                object_name="ImageQuilting",
                field="path",
                expected="'raster'|'random'",
                actual=path,
            )
        if not isinstance(property, PropertyMetadata):
            raise GeoBrainError(
                "property must be PropertyMetadata",
                object_name="ImageQuilting",
                field="property",
                expected="PropertyMetadata",
                actual=type(property).__name__,
            )
        if not isinstance(execution, SimulationExecutionConfig):
            raise GeoBrainError(
                "execution must be SimulationExecutionConfig", object_name="ImageQuilting",
                field="execution", expected="SimulationExecutionConfig", actual=type(execution).__name__,
            )

        # Validate that the TI is large enough to host at least one tile.
        for axis, (t, n) in enumerate(zip(ts, ti.shape)):
            if t > n:
                raise GeoBrainError(
                    "tile_size larger than training_image",
                    object_name="ImageQuilting",
                    field="tile_size",
                    expected=f"axis {axis} tile <= TI ({n})",
                    actual=t,
                )

        ti_finite = ti[np.isfinite(ti)]
        if ti_finite.size == 0:
            raise GeoBrainError(
                "training_image has no finite values",
                object_name="ImageQuilting",
                field="training_image",
                expected="at least one finite value",
                actual=0,
            )
        finite = ti_finite
        is_int = bool(np.allclose(finite, np.round(finite)))
        n_unique = int(np.unique(finite).size)

        self.training_image: FloatArray = ti
        self.tile_size: tuple[int, int, int] = ts
        self.overlap = float(overlap)
        self.tolerance = float(tolerance)
        self.n_candidates = int(n_candidates)
        self.path = str(path)
        self.property = property
        self.execution = execution
        self.nsim = execution.n_realizations
        self.seed = execution.seed
        self.n_jobs = execution.workers if execution.worker_backend != "serial" else 1
        self._is_categorical = is_int and n_unique <= 20
        raw_ti = ti[..., 0] if ti.shape[2] == 1 else ti
        axes = ("x", "y") if raw_ti.ndim == 2 else ("x", "y", "z")
        self.training_image_spec = TrainingImageSpec(raw_ti, property, axes, np.isnan(raw_ti))
        self.training_image_index = TrainingImageIndex.build(
            self.training_image_spec, np.zeros((1, raw_ti.ndim), dtype=np.int64), execution.budget_bytes
        )

    # ------------------------------------------------------------------

    def __call__(
        self,
        data: GeoFrame | None,
        domain: GeoGrid | GeoFrame,
    ) -> SimulationEnsemble:
        if isinstance(domain, GeoFrame):
            geom = domain.geometry
        else:
            geom = domain
        if not isinstance(geom, GeoGrid):
            raise GeoBrainError(
                "ImageQuilting requires a GeoGrid domain",
                object_name="ImageQuilting",
                field="domain",
                expected="GeoGrid (or GeoFrame wrapping one)",
                actual=type(geom).__name__,
            )

        grid = geom
        target_shape = gslib_grid_layout(grid).shape
        ti_shape = self.training_image.shape

        if (target_shape[2] > 1) != (ti_shape[2] > 1):
            raise GeoBrainError(
                "target grid and training image must have matching dimensionality",
                object_name="ImageQuilting",
                field="training_image",
                expected=f"matching ndim ({3 if target_shape[2] > 1 else 2})",
                actual=f"target {target_shape}, TI {ti_shape}",
            )

        # Hard-data pre-fill (used after quilting to overwrite hard values).
        pre_filled = np.full(target_shape, np.nan, dtype=np.float64)
        if data is not None:
            if not isinstance(data, GeoFrame):
                raise GeoBrainError(
                    "data must be a GeoFrame or None",
                    object_name="ImageQuilting",
                    field="data",
                    expected="GeoFrame | None",
                    actual=type(data).__name__,
                )
            if not data.columns:
                raise GeoBrainError(
                    "data has no columns",
                    object_name="ImageQuilting",
                    field="data.columns",
                    expected="non-empty",
                    actual=[],
                )
            col = data.columns[0]
            fill_hard_data(grid, data, col, pre_filled)

        seeds = _derive_realization_seeds(self.seed, self.nsim)

        worker = partial(
            _image_quilting_realization,
            simulator=self,
            target_shape=target_shape,
            pre_filled=pre_filled,
        )
        sims = run_realizations(worker, seeds, self.n_jobs)

        frames = tuple(make_simulation_frame(grid, sim, self.property) for sim in sims)
        return assemble_mps_ensemble(
            "ImageQuilting", self.property, self.execution, seeds, frames,
            cast(str, self.training_image_spec.fingerprint),
        )

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _one_realisation(
        self,
        sim_shape: tuple[int, int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        nx, ny, nz = sim_shape
        tx, ty, tz = self.tile_size
        ox = max(1, int(np.ceil(self.overlap * tx)))
        oy = max(1, int(np.ceil(self.overlap * ty)))
        oz = max(1, int(np.ceil(self.overlap * tz))) if tz > 1 else 0

        sx = max(tx - ox, 1)
        sy = max(ty - oy, 1)
        sz = max(tz - oz, 1) if tz > 1 else 1

        ntx = max(1, int(np.ceil(nx / sx)))
        nty = max(1, int(np.ceil(ny / sy)))
        ntz = max(1, int(np.ceil(nz / sz))) if tz > 1 else 1

        pad_nx = max(ntx * sx + ox, nx)
        pad_ny = max(nty * sy + oy, ny)
        pad_nz = max(ntz * sz + (oz if tz > 1 else 0), nz)

        simgrid = np.zeros((pad_nx, pad_ny, pad_nz), dtype=np.float64)
        pasted = np.zeros((pad_nx, pad_ny, pad_nz), dtype=bool)

        ti = self.training_image
        ti_nx, ti_ny, ti_nz = ti.shape
        max_ix = ti_nx - tx + 1
        max_iy = ti_ny - ty + 1
        max_iz = max(1, ti_nz - tz + 1)

        positions: list[tuple[int, int, int]] = []
        for kk in range(ntz):
            for jj in range(nty):
                for ii in range(ntx):
                    positions.append((ii, jj, kk))
        if self.path == "random":
            order = rng.permutation(len(positions))
            positions = [positions[i] for i in order]

        ti_mean = float(np.mean(ti[np.isfinite(ti)]))
        first_placed = False

        for ii, jj, kk in positions:
            x0 = ii * sx
            y0 = jj * sy
            z0 = kk * sz if tz > 1 else 0
            x1 = min(x0 + tx, pad_nx)
            y1 = min(y0 + ty, pad_ny)
            z1 = min(z0 + tz, pad_nz)
            actual_tx = x1 - x0
            actual_ty = y1 - y0
            actual_tz = z1 - z0

            overlap_mask = pasted[x0:x1, y0:y1, z0:z1]

            if np.any(overlap_mask):
                sim_region = simgrid[x0:x1, y0:y1, z0:z1]
                best_tile = self._find_best_tile(
                    ti,
                    sim_region,
                    overlap_mask,
                    actual_tx,
                    actual_ty,
                    actual_tz,
                    max_ix,
                    max_iy,
                    max_iz,
                    rng,
                )
                blended = self._boundary_cut(
                    sim_region,
                    best_tile,
                    overlap_mask,
                )
                simgrid[x0:x1, y0:y1, z0:z1] = blended
            else:
                if not first_placed:
                    n_first = min(max_ix * max_iy * max_iz, 50)
                    best_tile = None
                    best_diff = np.inf
                    for _ in range(n_first):
                        ix = int(rng.integers(0, max_ix))
                        iy = int(rng.integers(0, max_iy))
                        iz = int(rng.integers(0, max_iz))
                        cand = ti[ix : ix + actual_tx, iy : iy + actual_ty, iz : iz + actual_tz]
                        diff = abs(float(np.mean(cand)) - ti_mean)
                        if diff < best_diff:
                            best_diff = diff
                            best_tile = cand.copy()
                    simgrid[x0:x1, y0:y1, z0:z1] = best_tile
                    first_placed = True
                else:
                    ix = int(rng.integers(0, max_ix))
                    iy = int(rng.integers(0, max_iy))
                    iz = int(rng.integers(0, max_iz))
                    simgrid[x0:x1, y0:y1, z0:z1] = ti[
                        ix : ix + actual_tx,
                        iy : iy + actual_ty,
                        iz : iz + actual_tz,
                    ]

            pasted[x0:x1, y0:y1, z0:z1] = True

        return simgrid[:nx, :ny, :nz]

    # ------------------------------------------------------------------

    def _find_best_tile(
        self,
        ti: np.ndarray,
        sim_region: np.ndarray,
        overlap_mask: np.ndarray,
        tx: int,
        ty: int,
        tz: int,
        max_ix: int,
        max_iy: int,
        max_iz: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if max_ix < 1 or max_iy < 1 or max_iz < 1:
            return ti[:tx, :ty, :tz].copy()

        n_total = max_ix * max_iy * max_iz
        n_sample = min(n_total, self.n_candidates)
        effective_tol = self.tolerance
        if self._is_categorical:
            effective_tol = max(effective_tol, 0.3)

        cands: list[tuple[float, int, int, int]] = []
        best_dist = np.inf
        best_tile: np.ndarray | None = None

        if n_total <= n_sample:
            for iz in range(max_iz):
                for iy in range(max_iy):
                    for ix in range(max_ix):
                        tile = ti[ix : ix + tx, iy : iy + ty, iz : iz + tz]
                        diff = (tile - sim_region) * overlap_mask
                        dist = float(np.sum(diff * diff))
                        cands.append((dist, ix, iy, iz))
                        if dist < best_dist:
                            best_dist = dist
                            best_tile = tile.copy()
        else:
            for _ in range(n_sample):
                ix = int(rng.integers(0, max_ix))
                iy = int(rng.integers(0, max_iy))
                iz = int(rng.integers(0, max_iz))
                tile = ti[ix : ix + tx, iy : iy + ty, iz : iz + tz]
                diff = (tile - sim_region) * overlap_mask
                dist = float(np.sum(diff * diff))
                cands.append((dist, ix, iy, iz))
                if dist < best_dist:
                    best_dist = dist
                    best_tile = tile.copy()

        if best_dist == 0.0:
            threshold = 1.0 if self._is_categorical else 0.0
        else:
            threshold = (1.0 + effective_tol) * best_dist

        good = [c for c in cands if c[0] <= threshold]
        if not good:
            assert best_tile is not None
            return best_tile
        idx = int(rng.integers(0, len(good)))
        _, ix, iy, iz = good[idx]
        return ti[ix : ix + tx, iy : iy + ty, iz : iz + tz].copy()

    # ------------------------------------------------------------------

    @classmethod
    def _boundary_cut(
        cls,
        sim_region: np.ndarray,
        new_tile: np.ndarray,
        overlap_mask: np.ndarray,
    ) -> np.ndarray:
        """
        2-D graph-cut seam along each z-slice.

        For each z-slice of the patch we build a 4-connected grid graph
        whose pixels are the overlap-region cells. Edge weights are the
        squared-error matching cost between ``sim_region`` and
        ``new_tile`` (averaged across the two endpoints of the edge).
        Source = pixels on the "old" side boundary (cells whose
        only-pasted neighbours come from the already-pasted side);
        Sink = pixels on the "new" side boundary. Min-cut partitions
        the overlap into a smoother 2-D seam than a 1-D DP.

        Non-overlap z-slices and degenerate cases fall back to the
        simple per-axis selection that the 1-D DP would have produced.
        """
        result = new_tile.copy()
        if not np.any(overlap_mask):
            return result

        nx, ny, nz = sim_region.shape
        for kz in range(nz):
            mask = overlap_mask[:, :, kz]
            if not np.any(mask):
                continue
            sim_slc = sim_region[:, :, kz]
            tile_slc = new_tile[:, :, kz]

            keep_old = cls._graphcut_slice(sim_slc, tile_slc, mask)
            if keep_old is None:
                # Fall back to 1-D cumulative-DP along x for this slice.
                keep_old = cls._dp_slice(sim_slc, tile_slc, mask)

            result[:, :, kz] = np.where(keep_old, sim_slc, tile_slc)
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _graphcut_slice(
        sim_slc: np.ndarray,
        tile_slc: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray | None:
        """
        Compute the min-cut partition of one 2-D overlap slice.

        Returns a bool ``(nx, ny)`` array (``True`` = keep ``sim_slc``,
        ``False`` = take ``tile_slc``), or ``None`` if the graph
        degenerates (no source/sink available, or scipy maximum_flow
        returns a degenerate cut).

        Source/sink follow Kwatra et al. (2003): the min-cut seam must
        separate the already-pasted region from the tile's fresh
        interior. Sink side: overlap cells with an in-slice ``~mask``
        neighbour, inside the tile window every non-overlap cell is
        the tile's fresh interior, so these cells must come from the
        new tile. Source side: overlap cells adjacent to the
        already-pasted region, which (the fresh interior being the
        only in-slice non-overlap area) lies past the slice border,
        i.e. overlap cells on the slice border whose outside neighbour
        continues into the pasted region. Cells adjacent to both sides
        (overlap corners) are assigned to the sink so source and sink
        never coincide.
        """
        nx, ny = sim_slc.shape
        if nx == 0 or ny == 0:
            return None

        err = (sim_slc - tile_slc) ** 2
        ij = np.argwhere(mask)
        if ij.size == 0:
            return None

        src_mask = np.zeros((nx, ny), dtype=bool)
        snk_mask = np.zeros((nx, ny), dtype=bool)
        for ix, iy in ij:
            adj_fresh = False  # borders the tile's fresh interior
            adj_pasted = False  # borders the pasted region (out of slice)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                jx, jy = ix + dx, iy + dy
                if not (0 <= jx < nx and 0 <= jy < ny):
                    adj_pasted = True
                elif not mask[jx, jy]:
                    adj_fresh = True
            if adj_fresh:
                snk_mask[ix, iy] = True
            elif adj_pasted:
                src_mask[ix, iy] = True

        # Degenerate overlap (no old side or no fresh interior visible):
        # fall back to the 1-D DP.
        if not src_mask.any() or not snk_mask.any():
            return None

        # Build the directed graph: source = node S, sink = node T,
        # plus one node per overlap pixel.
        # Index 0 = S, 1 = T, 2..(2+n_pix-1) = overlap pixels.
        pix_id = -np.ones((nx, ny), dtype=np.int64)
        ids = []
        for ix, iy in ij:
            pix_id[ix, iy] = len(ids) + 2
            ids.append((int(ix), int(iy)))
        n_pix = len(ids)
        N = n_pix + 2
        S, T = 0, 1

        rows: list[int] = []
        cols: list[int] = []
        caps: list[int] = []

        # scipy.sparse.csgraph.maximum_flow requires integer capacities.
        # Scale squared errors to int32, clipped to a sane range.
        # Use scale * max(1, err.max()) so we span the int range usefully.
        max_err = float(err[mask].max()) if mask.any() else 0.0
        scale = 10_000.0 / max(max_err, 1e-9)
        INF_CAP = 10**9

        # Edges between adjacent overlap pixels (both directions, as
        # scipy max-flow uses directed capacities).
        for ix, iy in ij:
            for dx, dy in ((1, 0), (0, 1)):
                jx, jy = ix + dx, iy + dy
                if not (0 <= jx < nx and 0 <= jy < ny):
                    continue
                if not mask[jx, jy]:
                    continue
                w = (err[ix, iy] + err[jx, jy]) * 0.5 * scale
                cap = max(1, int(round(w)))
                u = int(pix_id[ix, iy])
                v = int(pix_id[jx, jy])
                rows.append(u)
                cols.append(v)
                caps.append(cap)
                rows.append(v)
                cols.append(u)
                caps.append(cap)

        # Source/sink edges (infinite capacity to make boundary
        # assignments hard constraints).
        for ix, iy in ij:
            if src_mask[ix, iy]:
                rows.append(S)
                cols.append(int(pix_id[ix, iy]))
                caps.append(INF_CAP)
            if snk_mask[ix, iy]:
                rows.append(int(pix_id[ix, iy]))
                cols.append(T)
                caps.append(INF_CAP)

        if not rows:
            return None
        graph = csr_matrix(
            (
                np.asarray(caps, dtype=np.int32),
                (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
            ),
            shape=(N, N),
        )
        try:
            flow = maximum_flow(graph, S, T)
        except Exception as exc:
            # Graceful degradation, but never silent: the caller falls back
            # to the straight-seam cut, which lowers seam quality. Python's
            # default warning filter deduplicates by location, so this fires
            # once per process, not once per tile.
            warnings.warn(
                f"image quilting min-cut failed ({type(exc).__name__}: {exc});"
                " falling back to straight seams; seam quality is reduced",
                stacklevel=2,
            )
            return None

        # Recover S-side reachability in the residual graph =>
        # those pixels keep sim_slc; the rest take tile_slc. The
        # residual capacity along edge u->v is cap(u,v) - flow(u,v);
        # scipy's maximum_flow returns a *signed* flow matrix
        # (positive = forward flow on the original edge, negative on
        # the reverse). Residual capacity = cap - flow.
        cap_mat = graph.toarray().astype(np.int64)
        flow_mat = flow.flow.toarray().astype(np.int64)
        residual = cap_mat - flow_mat
        reachable = np.zeros(N, dtype=bool)
        stack = [S]
        reachable[S] = True
        while stack:
            u = stack.pop()
            for v in range(N):
                if not reachable[v] and residual[u, v] > 0:
                    reachable[v] = True
                    stack.append(v)

        keep_old = np.zeros((nx, ny), dtype=bool)
        for k, (ix, iy) in enumerate(ids):
            keep_old[ix, iy] = bool(reachable[k + 2])
        # Cells outside the overlap mask are not touched by the seam;
        # caller's np.where will overwrite with new_tile by default, so
        # we leave keep_old False there (matches the old DP convention).
        return keep_old

    # ------------------------------------------------------------------

    @staticmethod
    def _dp_slice(
        sim_slc: np.ndarray,
        tile_slc: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        1-D cumulative-DP fallback along the x-axis.

        Used when the graph-cut degenerates (e.g., single-pixel overlap
        with no valid source/sink split).
        """
        nx, ny = sim_slc.shape
        error = (sim_slc - tile_slc) ** 2
        cum_error = np.zeros_like(error)
        cum_error[0, :] = error[0, :]
        for i in range(1, nx):
            cum_error[i, :] = error[i, :] + cum_error[i - 1, :]
        total = cum_error[-1, :]
        total = np.where(total > 0, total, 1.0)
        half = 0.5 * total

        keep_old = np.zeros((nx, ny), dtype=bool)
        for i in range(nx):
            keep_old[i, :] = mask[i, :] & (cum_error[i, :] <= half)
        return keep_old

    def __repr__(self) -> str:
        return (
            f"ImageQuilting(ti_shape={self.training_image.shape}, "
            f"tile_size={self.tile_size}, overlap={self.overlap}, "
            f"tolerance={self.tolerance}, nsim={self.nsim})"
        )
