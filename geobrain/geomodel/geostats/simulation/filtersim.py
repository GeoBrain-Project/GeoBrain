"""FILTERSIM: Filter-based MPS simulation (Zhang et al. 2006).

Patch-based pattern simulator that, unlike SNESIM's offset-tuple matching,
projects each TI patch into a low-dimensional **filter-score** space and
clusters patches into **prototypes**. At simulation time, each local data
event is projected into the same filter-score space, matched to the
nearest prototype, and a representative patch from the cluster is pasted.

Patch matching uses an **explicit 7-filter bank**:

- Average (uniform mean)
- 4 directional Sobel responses (East, South, North-East, South-East)
- Laplacian (4-connected discrete Laplacian)
- Local variance

Each TI patch is reduced to a 7-vector of filter outputs, and k-means
clusters the filter-output space. This matches the SGeMS-style
filter-bank reduction (cf. the ``core.filters.FilterBank``,
which uses 9-filters of directional averages + gradients; we use a
mathematically distinct but spec-compliant bank emphasising edge
responses).

Single-pass raster pasting (a full implementation would add multi-pass refinement). Supports
binary/categorical AND continuous training images.

References:
- Zhang, T., Switzer, P. & Journel, A. (2006). Filter-based classification
  of training image patterns for spatial simulation. *Mathematical Geology*
  38(1).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from functools import partial
from typing import cast

import numpy as np

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

__all__ = ["FILTERSIM", "build_default_filter_bank"]


# ---------------------------------------------------------------------------
# Filter-bank construction
# ---------------------------------------------------------------------------


def build_default_filter_bank(
    pnx: int,
    pny: int,
    pnz: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the default 7-filter bank for FILTERSIM.

    Returns ``(weights, names)`` where ``weights`` is a
    ``(n_filters, patch_len)`` matrix; applying a filter is just
    ``weights @ patch.ravel() / norm_per_filter`` for linear filters.
    For the *variance* filter (non-linear) we mark it specially via the
    name and handle it in :func:`_compute_scores`.

    Filters:

    - ``avg``: uniform average (sum / N).
    - ``sobel_e``: directional Sobel response, eastward gradient.
    - ``sobel_s``: directional Sobel response, southward gradient.
    - ``sobel_ne``: directional Sobel response, north-east diagonal.
    - ``sobel_se``: directional Sobel response, south-east diagonal.
    - ``laplacian``: 4-connected (2-D) / 6-connected (3-D) Laplacian
      summed over the patch.
    - ``var``: marker for the variance filter (handled by name).
    """
    shape = (pnx, pny, pnz)
    patch_len = pnx * pny * pnz
    weights: list[np.ndarray] = []
    names: list[str] = []

    # ---- average ------------------------------------------------------
    w_avg = np.full(shape, 1.0 / patch_len, dtype=np.float64)
    weights.append(w_avg.ravel())
    names.append("avg")

    # Standard 3x3 Sobel kernels (replicated along z for 3-D).
    # E = positive x-direction gradient.
    sobel_e_2d = np.array(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # S = positive y-direction gradient.
    sobel_s_2d = np.array(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=np.float64,
    )
    # NE diagonal gradient (high in NE, low in SW).
    sobel_ne_2d = np.array(
        [[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    # SE diagonal gradient (high in SE, low in NW).
    sobel_se_2d = np.array(
        [[-2.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]],
        dtype=np.float64,
    )

    # Helper: expand a 3x3 2-D kernel to the full patch by centering and
    # zero-padding, then replicating across z for 3-D patches.
    def _expand_2d_kernel(k2d: np.ndarray) -> np.ndarray:
        ker = np.zeros(shape, dtype=np.float64)
        cx, cy = pnx // 2, pny // 2
        kxh, kyh = k2d.shape[0] // 2, k2d.shape[1] // 2
        for ix in range(k2d.shape[0]):
            for iy in range(k2d.shape[1]):
                gx, gy = cx + (ix - kxh), cy + (iy - kyh)
                if 0 <= gx < pnx and 0 <= gy < pny:
                    if pnz == 1:
                        ker[gx, gy, 0] = k2d[ix, iy]
                    else:
                        # Replicate across z, weight 1/pnz so total
                        # response stays comparable across patch depths.
                        ker[gx, gy, :] = k2d[ix, iy] / pnz
        return ker

    for name, k2d in (
        ("sobel_e", sobel_e_2d),
        ("sobel_s", sobel_s_2d),
        ("sobel_ne", sobel_ne_2d),
        ("sobel_se", sobel_se_2d),
    ):
        weights.append(_expand_2d_kernel(k2d).ravel())
        names.append(name)

    # ---- Laplacian (sum of discrete 4-/6-connected Laplacian over patch)
    # The Laplacian at each pixel is (sum of neighbours) - n_neighbours*centre.
    # Summed over the whole patch, only the boundary terms survive
    # (interior cancels). We just build the weight matrix explicitly:
    # weight[p] = (count of neighbours of cell p inside the patch) -
    # n_max_neighbours * indicator(p is a centre of any cell, i.e., 1).
    # Concretely, for each cell, contribute +1 to each of its in-patch
    # neighbours and -n_neighbours_total to itself, then sum.
    lap = np.zeros(shape, dtype=np.float64)
    neigh_offsets_2d = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
    neigh_offsets_3d = neigh_offsets_2d + [(0, 0, -1), (0, 0, 1)]
    neigh = neigh_offsets_2d if pnz == 1 else neigh_offsets_3d
    n_neigh = len(neigh)
    for ix in range(pnx):
        for iy in range(pny):
            for iz in range(pnz):
                # The Laplacian at (ix,iy,iz) is sum_neighbours - n*centre.
                # When we sum across the whole patch:
                #   centre contribution: -n_neigh to each cell
                #   neighbour contribution: +1 to each in-patch neighbour
                lap[ix, iy, iz] -= n_neigh  # itself, when it is the centre
                for dx, dy, dz in neigh:
                    jx, jy, jz = ix + dx, iy + dy, iz + dz
                    if 0 <= jx < pnx and 0 <= jy < pny and 0 <= jz < pnz:
                        lap[jx, jy, jz] += 1.0
    weights.append(lap.ravel())
    names.append("laplacian")

    # ---- variance (non-linear) marker --------------------------------
    weights.append(np.zeros(patch_len, dtype=np.float64))
    names.append("var")

    W = np.stack(weights, axis=0)
    return W, names


def _filtersim_realization(
    seed: int,
    *,
    simulator: "FILTERSIM",
    pre_filled: np.ndarray,
    patches: np.ndarray,
    score_mean: np.ndarray,
    score_std: np.ndarray,
    proto_centroids: np.ndarray,
    cluster_index: list[list[int]],
) -> np.ndarray:
    field = simulator._one_realisation(
        pre_filled.copy(),
        patches,
        score_mean,
        score_std,
        proto_centroids,
        cluster_index,
        np.random.default_rng(int(seed)),
    )
    return as_float_array(field.transpose(2, 1, 0).reshape(-1))


# ---------------------------------------------------------------------------


class FILTERSIM(SimulationAgentContract):
    """
    FILTERSIM filter-bank-prototype MPS simulator.

    Args:
        training_image: 2-D ``(nx, ny)`` or 3-D ``(nx, ny, nz)`` numpy array.
        categories: optional list of category values; if ``None`` the TI is
            treated as continuous.
        patch_half: half-size of the patch template ``(hx, hy, hz)``.
            Default ``(3, 3, 0)`` (7x7 patch in 2-D).
        n_prototypes: number of prototype clusters (k-means). Default ``20``.
        n_kmeans_iter: max k-means iterations. Default ``20``.
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
        categories: list[int] | None = None,
        *,
        property: PropertyMetadata,
        execution: SimulationExecutionConfig,
        patch_half: tuple[int, int, int] = (3, 3, 0),
        n_prototypes: int = 20,
        n_kmeans_iter: int = 20,
    ) -> None:
        ti = as_float_array(training_image)
        if ti.ndim not in (2, 3):
            raise GeoBrainError(
                "training_image must be 2-D or 3-D",
                object_name="FILTERSIM",
                field="training_image",
                expected="ndim 2 or 3",
                actual=ti.ndim,
            )
        if ti.ndim == 2:
            ti = as_float_array(ti[..., None])
        if any(s <= 0 for s in ti.shape):
            raise GeoBrainError(
                "training_image must have positive dimensions",
                object_name="FILTERSIM",
                field="training_image",
                expected="all dims > 0",
                actual=ti.shape,
            )

        ph = tuple(int(v) for v in patch_half)
        if len(ph) != 3 or any(v < 0 for v in ph):
            raise GeoBrainError(
                "patch_half must be 3-tuple of non-negative ints",
                object_name="FILTERSIM",
                field="patch_half",
                expected="(hx, hy, hz) with hi >= 0",
                actual=ph,
            )
        if all(v == 0 for v in ph):
            raise GeoBrainError(
                "patch_half must have at least one positive component",
                object_name="FILTERSIM",
                field="patch_half",
                expected="not all zero",
                actual=ph,
            )
        if n_prototypes < 1:
            raise GeoBrainError(
                "n_prototypes must be >= 1",
                object_name="FILTERSIM",
                field="n_prototypes",
                expected=">= 1",
                actual=n_prototypes,
            )
        if n_kmeans_iter < 1:
            raise GeoBrainError(
                "n_kmeans_iter must be >= 1",
                object_name="FILTERSIM",
                field="n_kmeans_iter",
                expected=">= 1",
                actual=n_kmeans_iter,
            )
        if not isinstance(property, PropertyMetadata):
            raise GeoBrainError(
                "property must be PropertyMetadata",
                object_name="FILTERSIM",
                field="property",
                expected="PropertyMetadata",
                actual=type(property).__name__,
            )
        if not isinstance(execution, SimulationExecutionConfig):
            raise GeoBrainError(
                "execution must be SimulationExecutionConfig", object_name="FILTERSIM",
                field="execution", expected="SimulationExecutionConfig", actual=type(execution).__name__,
            )

        ti_finite = ti[np.isfinite(ti)]
        if ti_finite.size == 0:
            raise GeoBrainError(
                "training_image has no finite values",
                object_name="FILTERSIM",
                field="training_image",
                expected="at least one finite value",
                actual=0,
            )

        self.training_image: FloatArray = ti
        self.patch_half = (ph[0], ph[1], ph[2] if ti.shape[2] > 1 else 0)
        self.n_prototypes = int(n_prototypes)
        self.n_kmeans_iter = int(n_kmeans_iter)
        self.property = property
        self.execution = execution
        self.nsim = execution.n_realizations
        self.seed = execution.seed
        self.n_jobs = execution.workers if execution.worker_backend != "serial" else 1
        self.categories: list[int] | None = (
            [int(c) for c in categories] if categories is not None else None
        )
        self._is_categorical = self.categories is not None
        raw_ti = ti[..., 0] if ti.shape[2] == 1 else ti
        axes = ("x", "y") if raw_ti.ndim == 2 else ("x", "y", "z")
        self.training_image_spec = TrainingImageSpec(raw_ti, property, axes, np.isnan(raw_ti))
        self.training_image_index = TrainingImageIndex.build(
            self.training_image_spec, np.zeros((1, raw_ti.ndim), dtype=np.int64), execution.budget_bytes
        )

        # Patch dimensions
        self._pnx = 2 * self.patch_half[0] + 1
        self._pny = 2 * self.patch_half[1] + 1
        self._pnz = 2 * self.patch_half[2] + 1
        self._patch_len = self._pnx * self._pny * self._pnz

        # Build the explicit filter bank once.
        self._filter_weights, self._filter_names = build_default_filter_bank(
            self._pnx,
            self._pny,
            self._pnz,
        )

        # TI-wide mode (fill value for out-of-bounds when extracting events)
        self._ti_mode = float(np.mean(ti_finite))
        if self._is_categorical:
            unique, counts = np.unique(ti_finite, return_counts=True)
            self._ti_mode = float(unique[int(np.argmax(counts))])

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
                "FILTERSIM requires a GeoGrid domain",
                object_name="FILTERSIM",
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
                object_name="FILTERSIM",
                field="training_image",
                expected=f"matching ndim ({3 if target_shape[2] > 1 else 2})",
                actual=f"target {target_shape}, TI {ti_shape}",
            )

        pre_filled = np.full(target_shape, np.nan, dtype=np.float64)
        if data is not None:
            if not isinstance(data, GeoFrame):
                raise GeoBrainError(
                    "data must be a GeoFrame or None",
                    object_name="FILTERSIM",
                    field="data",
                    expected="GeoFrame | None",
                    actual=type(data).__name__,
                )
            if not data.columns:
                raise GeoBrainError(
                    "data has no columns",
                    object_name="FILTERSIM",
                    field="data.columns",
                    expected="non-empty",
                    actual=[],
                )
            col = data.columns[0]
            fill_hard_data(grid, data, col, pre_filled)

        # ---- Build patch catalogue + filter scores + prototypes ----
        patches = self._extract_ti_patches()
        scores = self._compute_scores(patches)
        # Standardise scores per-filter so k-means isn't dominated by
        # the highest-magnitude response (Sobel responses dwarf the
        # Laplacian otherwise).
        score_mean = scores.mean(axis=0)
        score_std = scores.std(axis=0)
        score_std = np.where(score_std > 0, score_std, 1.0)
        scores_std = (scores - score_mean) / score_std

        proto_assignments, proto_centroids = self._kmeans(scores_std)
        # Cluster -> indices of TI patches belonging to it
        cluster_index: list[list[int]] = [[] for _ in range(self.n_prototypes)]
        for pi, c in enumerate(proto_assignments):
            cluster_index[int(c)].append(pi)

        seeds = _derive_realization_seeds(self.seed, self.nsim)

        worker = partial(
            _filtersim_realization,
            simulator=self,
            pre_filled=pre_filled,
            patches=patches,
            score_mean=score_mean,
            score_std=score_std,
            proto_centroids=proto_centroids,
            cluster_index=cluster_index,
        )
        sims = run_realizations(worker, seeds, self.n_jobs)

        frames = tuple(make_simulation_frame(grid, sim, self.property) for sim in sims)
        return assemble_mps_ensemble(
            "FILTERSIM", self.property, self.execution, seeds, frames,
            cast(str, self.training_image_spec.fingerprint),
        )

    # ------------------------------------------------------------------
    # Patch catalogue & filter scores
    # ------------------------------------------------------------------

    def _extract_ti_patches(self) -> np.ndarray:
        """
        Return ``(n_patches, patch_len)`` array of TI patches.

        Each patch uses periodic wrap of the TI at borders.
        """
        ti = self.training_image
        tnx, tny, tnz = ti.shape
        hx, hy, hz = self.patch_half
        dxs = np.arange(-hx, hx + 1)
        dys = np.arange(-hy, hy + 1)
        dzs = np.arange(-hz, hz + 1)

        n_patches = tnx * tny * tnz
        patches = np.empty((n_patches, self._patch_len), dtype=np.float64)
        DX, DY, DZ = np.meshgrid(dxs, dys, dzs, indexing="ij")
        flat_dx = DX.ravel()
        flat_dy = DY.ravel()
        flat_dz = DZ.ravel()

        p = 0
        for ck in range(tnz):
            for cj in range(tny):
                for ci in range(tnx):
                    ii = (ci + flat_dx) % tnx
                    jj = (cj + flat_dy) % tny
                    kk = (ck + flat_dz) % tnz
                    vals = ti[ii, jj, kk]
                    vals = np.where(np.isfinite(vals), vals, self._ti_mode)
                    patches[p, :] = vals
                    p += 1
        return patches

    def _compute_scores(self, patches: np.ndarray) -> np.ndarray:
        """
        Apply the 7-filter bank to a ``(n_patches, patch_len)`` matrix.

        Returns ``(n_patches, n_filters)`` array of filter outputs.
        """
        W = self._filter_weights  # (n_filters, patch_len)
        names = self._filter_names
        n_patches = patches.shape[0]
        n_filters = W.shape[0]
        scores = np.empty((n_patches, n_filters), dtype=np.float64)
        # Linear filters via matrix multiply.
        # Variance is computed separately because it's non-linear.
        var_idx = names.index("var") if "var" in names else -1
        # For the linear rows, do W @ patches.T -> (n_filters, n_patches).
        linear = W @ patches.T  # (n_filters, n_patches)
        for i in range(n_filters):
            if i == var_idx:
                scores[:, i] = patches.var(axis=1)
            else:
                scores[:, i] = linear[i]
        return scores

    # ------------------------------------------------------------------
    # k-means
    # ------------------------------------------------------------------

    def _kmeans(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Lloyd k-means with k-means++ seeding.

        Returns ``(assignments, centroids)``.
        """
        n, d = scores.shape
        k = min(self.n_prototypes, n)
        rng = np.random.default_rng(self.seed)

        # k-means++ init
        centroids = np.empty((k, d), dtype=np.float64)
        first = int(rng.integers(0, n))
        centroids[0] = scores[first]
        closest_sq = np.full(n, np.inf, dtype=np.float64)
        for c in range(1, k):
            diff = scores - centroids[c - 1]
            d_sq = np.sum(diff * diff, axis=1)
            closest_sq = np.minimum(closest_sq, d_sq)
            total = float(closest_sq.sum())
            if total <= 0.0:
                idx = int(rng.integers(0, n))
            else:
                r = float(rng.random()) * total
                cum = 0.0
                idx = n - 1
                for i, val in enumerate(closest_sq):
                    cum += float(val)
                    if cum >= r:
                        idx = i
                        break
            centroids[c] = scores[idx]

        assignments = np.zeros(n, dtype=np.int64)
        for _ in range(self.n_kmeans_iter):
            diffs = scores[:, None, :] - centroids[None, :, :]
            d_sq = np.sum(diffs * diffs, axis=2)
            new_assign = np.argmin(d_sq, axis=1).astype(np.int64)
            if np.array_equal(new_assign, assignments):
                assignments = new_assign
                break
            assignments = new_assign
            for c in range(k):
                mask = assignments == c
                if np.any(mask):
                    centroids[c] = scores[mask].mean(axis=0)
                else:
                    centroids[c] = scores[int(rng.integers(0, n))]
        if k < self.n_prototypes:
            padded = np.zeros((self.n_prototypes, d), dtype=np.float64)
            padded[:k] = centroids
            centroids = padded
        return assignments, centroids

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _one_realisation(
        self,
        sim: np.ndarray,
        patches: np.ndarray,
        score_mean: np.ndarray,
        score_std: np.ndarray,
        proto_centroids: np.ndarray,
        cluster_index: list[list[int]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        nx, ny, nz = sim.shape
        hx, hy, hz = self.patch_half
        pnx, pny, pnz = self._pnx, self._pny, self._pnz

        # Snapshot of pre-filled (hard data) mask before quilting starts.
        cond_mask = np.isfinite(sim).copy()

        # Initialise NaN cells with marginal samples drawn from TI values.
        ti_pool = self.training_image[np.isfinite(self.training_image)].ravel()
        nan_mask = np.isnan(sim)
        if nan_mask.any():
            draws = rng.choice(ti_pool, size=int(nan_mask.sum()))
            sim[nan_mask] = draws

        # Patch-pasting raster path with step = patch_size - overlap (1 cell).
        step_x = max(pnx - 1, 1)
        step_y = max(pny - 1, 1)
        step_z = max(pnz - 1, 1) if pnz > 1 else 1

        W = self._filter_weights
        var_idx = self._filter_names.index("var") if "var" in self._filter_names else -1
        n_filters = W.shape[0]

        for kz in range(0, nz, step_z):
            for jy in range(0, ny, step_y):
                for ix in range(0, nx, step_x):
                    # Extract local data event (with edge clamping to fill).
                    # NOTE: the flat ordering MUST match the TI patch
                    # catalogue built in ``_extract_ti_patches`` (meshgrid
                    # ``indexing="ij"`` over (dxs, dys, dzs) + C-order
                    # ravel): dx outermost, dz fastest.
                    patch = np.empty(self._patch_len, dtype=np.float64)
                    p = 0
                    for dx in range(-hx, hx + 1):
                        for dy in range(-hy, hy + 1):
                            for dz in range(-hz, hz + 1):
                                jx, jjy, jjz = ix + dx, jy + dy, kz + dz
                                if 0 <= jx < nx and 0 <= jjy < ny and 0 <= jjz < nz:
                                    patch[p] = sim[jx, jjy, jjz]
                                else:
                                    patch[p] = self._ti_mode
                                p += 1

                    # Compute filter scores for this patch.
                    scores = np.empty(n_filters, dtype=np.float64)
                    linear = W @ patch
                    for i in range(n_filters):
                        if i == var_idx:
                            scores[i] = patch.var()
                        else:
                            scores[i] = linear[i]
                    # Standardise to match clustering space.
                    scores_std = (scores - score_mean) / score_std

                    # Nearest prototype
                    diff = proto_centroids - scores_std
                    d_sq = np.sum(diff * diff, axis=1)
                    best = int(np.argmin(d_sq))
                    members = cluster_index[best]
                    if not members:
                        # Empty cluster: fall back to a random patch.
                        chosen = int(rng.integers(0, patches.shape[0]))
                    else:
                        chosen = int(members[int(rng.integers(0, len(members)))])
                    chosen_patch = patches[chosen].reshape((pnx, pny, pnz))

                    # Paste, but never overwrite hard data.
                    p = 0
                    for dz in range(-hz, hz + 1):
                        for dy in range(-hy, hy + 1):
                            for dx in range(-hx, hx + 1):
                                jx, jjy, jjz = ix + dx, jy + dy, kz + dz
                                if 0 <= jx < nx and 0 <= jjy < ny and 0 <= jjz < nz:
                                    if not cond_mask[jx, jjy, jjz]:
                                        sim[jx, jjy, jjz] = chosen_patch[dx + hx, dy + hy, dz + hz]

        # Categorical TI: snap to nearest category.
        if self._is_categorical and self.categories is not None:
            cats = np.array(self.categories, dtype=np.float64)
            flat = sim.ravel()
            for i in range(flat.size):
                v = flat[i]
                if np.isnan(v):
                    flat[i] = cats[int(rng.integers(0, cats.size))]
                else:
                    idx = int(np.argmin(np.abs(cats - v)))
                    flat[i] = cats[idx]
            sim = flat.reshape(sim.shape)
        return sim

    def __repr__(self) -> str:
        return (
            f"FILTERSIM(ti_shape={self.training_image.shape}, "
            f"patch_half={self.patch_half}, n_prototypes={self.n_prototypes}, "
            f"n_filters={len(self._filter_names)}, nsim={self.nsim})"
        )
