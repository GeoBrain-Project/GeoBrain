"""SNESIM: Single Normal Equation SIMulation (Strebelle 2002).

Pattern-based multi-point simulator. For each unsimulated node, builds a
**data event** from a fixed search template applied to the already-simulated
neighbourhood, then queries a **search tree** built from the training
image to obtain a conditional category distribution.

Key simplifications relative to the full SGeMS algorithm:

- Servo-system correction and simulated-annealing post-pass are dropped;
  see the SGeMS reference for the complete algorithm.

Pattern storage uses a hierarchical :class:`SearchTree` keyed
by the position-ordered sequence of neighbour categories. The tree gives
``O(depth)`` lookup per simulated node and matches the data structure
one-to-one. The pattern counts collected by the tree are exactly the
same multiset as the flat catalogue's per-prefix counts, so all existing
simulation results are bit-identical.

Currently supports **binary or low-cardinality categorical** TIs only.

References:
- Strebelle, S. (2002). Conditional simulation of complex geological
  structures using multiple-point statistics. *Mathematical Geology* 34(1).

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

__all__ = ["SNESIM", "SearchTree", "SearchTreeNode"]


def _build_template_offsets(
    template_half: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    """
    Ellipsoidal template, sorted by squared distance from the centre.

    The centre itself is excluded.
    """
    hx, hy, hz = template_half
    items: list[tuple[float, tuple[int, int, int]]] = []
    for dz in range(-hz, hz + 1):
        for dy in range(-hy, hy + 1):
            for dx in range(-hx, hx + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                norm = 0.0
                if hx > 0:
                    norm += (dx / hx) ** 2
                if hy > 0:
                    norm += (dy / hy) ** 2
                if hz > 0:
                    norm += (dz / hz) ** 2
                if norm <= 1.0:
                    items.append((dx * dx + dy * dy + dz * dz, (dx, dy, dz)))
    items.sort()
    return [off for _, off in items]


def _snesim_realization(
    seed: int,
    *,
    simulator: "SNESIM",
    pre_filled: np.ndarray,
    levels: list[tuple[list[tuple[int, int, int]], "SearchTree"]],
) -> np.ndarray:
    field = simulator._one_realisation(
        pre_filled.copy(),
        levels,
        np.random.default_rng(int(seed)),
    )
    return as_float_array(field.transpose(2, 1, 0).reshape(-1))


# ---------------------------------------------------------------------------
# SearchTree (hierarchical pattern dictionary)
# ---------------------------------------------------------------------------


class SearchTreeNode:
    """
    One node in the SNESIM search tree.

    Each node stores:

    - ``counts``: ``(n_categories,)`` int64 array: number of times the
      centre cell took each category when the data-event prefix reaching
      this node matched the TI.
    - ``children``: dict mapping ``(offset_index, category_value)`` →
      child node. The ``offset_index`` is the index into the per-level
      template's ``offsets`` list. We key on the full ``(k, v)`` pair
      (rather than just ``v``) so the tree honours offset-index
      identity, the same key structure as a flat
      catalogue, just nested.
    """

    __slots__ = ("counts", "children")

    def __init__(self, n_categories: int) -> None:
        self.counts = np.zeros(n_categories, dtype=np.int64)
        self.children: dict[tuple[int, float], SearchTreeNode] = {}


class SearchTree:
    """
    Hierarchical pattern dictionary for SNESIM.

    A hierarchical form of the flat ``(offset_idx, value)``-tuple → counts dict.
    The tree is built by scanning the training image: for every fully
    finite TI cell, walk the template starting from the nearest offset
    and, at each step, increment ``counts`` along the path of nodes
    indexed by the encountered category values.

    Query semantics: we walk the tree following the
    data event nearest-first, return the deepest node whose
    ``counts.sum() >= min_replicates``; if no such node exists, fall
    back to the prior. This iterative truncation is achieved naturally
    by tree traversal, the deepest valid node corresponds to the
    longest data-event prefix that still has enough replicates.
    """

    __slots__ = ("n_categories", "_cat_index", "categories", "root", "n_patterns")

    def __init__(
        self,
        n_categories: int,
        categories: list[int],
    ) -> None:
        self.n_categories = int(n_categories)
        self.categories = list(categories)
        self._cat_index = {c: i for i, c in enumerate(self.categories)}
        self.root = SearchTreeNode(self.n_categories)
        self.n_patterns = 0

    # ------------------------------------------------------------------

    def insert(
        self,
        data_event: list[tuple[int, float]],
        center_value: int,
    ) -> None:
        """
        Insert one TI pattern: ``data_event`` is nearest-first list of
        ``(offset_index, value)`` along the template; ``center_value`` is
        the centre cell's category.
        """
        idx = self._cat_index.get(center_value)
        if idx is None:
            return
        # Increment the root (empty prefix → marginal).
        self.root.counts[idx] += 1
        node = self.root
        for key in data_event:
            child = node.children.get(key)
            if child is None:
                child = SearchTreeNode(self.n_categories)
                node.children[key] = child
            child.counts[idx] += 1
            node = child
        self.n_patterns += 1

    # ------------------------------------------------------------------

    def query(
        self,
        data_event: list[tuple[int, float]],
        min_replicates: int,
    ) -> tuple[np.ndarray, int]:
        """
        Walk the tree along ``data_event``. Return the counts at the
        deepest reached node whose ``counts.sum() >= min_replicates``,
        together with the conditioning depth used.

        If no internal node along the path meets ``min_replicates``, the
        root counts are returned (which is the global marginal).
        """
        node = self.root
        best_counts = node.counts
        best_depth = 0
        depth = 0
        for key in data_event:
            child = node.children.get(key)
            if child is None:
                break
            depth += 1
            node = child
            if int(node.counts.sum()) >= min_replicates:
                best_counts = node.counts
                best_depth = depth
        return best_counts, best_depth

    # ------------------------------------------------------------------

    @property
    def tree_size(self) -> int:
        """Total number of nodes in the tree."""
        return self._count_nodes(self.root)

    def _count_nodes(self, node: SearchTreeNode) -> int:
        n = 1
        for child in node.children.values():
            n += self._count_nodes(child)
        return n


class SNESIM(SimulationAgentContract):
    """
    SNESIM SearchTree-based MPS simulator (binary / categorical).

    Args:
        training_image: 2-D ``(nx, ny)`` or 3-D ``(nx, ny, nz)`` numpy array.
            Must be categorical (integer-valued).
        categories: sequence of distinct category values present in the TI.
            If omitted, taken from ``np.unique(training_image)``.
        template_half: half-size of the ellipsoidal search template
            ``(hx, hy, hz)``. Default ``(4, 4, 0)`` (2-D).
        n_multigrid: number of multigrid levels (>=1). Default ``2``.
        min_replicates: minimum number of TI replicates needed to honour a
            data event; if fewer, the data event is truncated by dropping
            the furthest informed neighbours. Default ``4``.
        max_cond: maximum conditioning data-event size retained. Default
            ``30``.
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
        template_half: tuple[int, int, int] = (4, 4, 0),
        n_multigrid: int = 2,
        min_replicates: int = 4,
        max_cond: int = 30,
    ) -> None:
        ti = as_float_array(training_image)
        if ti.ndim not in (2, 3):
            raise GeoBrainError(
                "training_image must be 2-D or 3-D",
                object_name="SNESIM",
                field="training_image",
                expected="ndim 2 or 3",
                actual=ti.ndim,
            )
        if ti.ndim == 2:
            ti = as_float_array(ti[..., None])
        if any(s <= 0 for s in ti.shape):
            raise GeoBrainError(
                "training_image must have positive dimensions",
                object_name="SNESIM",
                field="training_image",
                expected="all dims > 0",
                actual=ti.shape,
            )

        ti_finite = ti[np.isfinite(ti)]
        if ti_finite.size == 0:
            raise GeoBrainError(
                "training_image has no finite values",
                object_name="SNESIM",
                field="training_image",
                expected="at least one finite value",
                actual=0,
            )
        if not np.allclose(ti_finite, np.round(ti_finite)):
            raise GeoBrainError(
                "SNESIM requires a categorical training_image (integer values)",
                object_name="SNESIM",
                field="training_image",
                expected="integer-valued",
                actual="non-integer values present",
            )

        if categories is None:
            cats = sorted({int(v) for v in np.unique(ti_finite).tolist()})
        else:
            cats = [int(c) for c in categories]
            if not cats:
                raise GeoBrainError(
                    "categories must be non-empty",
                    object_name="SNESIM",
                    field="categories",
                    expected="non-empty",
                    actual=[],
                )

        th = tuple(int(v) for v in template_half)
        if len(th) != 3 or any(v < 0 for v in th):
            raise GeoBrainError(
                "template_half must be 3-tuple of non-negative ints",
                object_name="SNESIM",
                field="template_half",
                expected="(hx, hy, hz) with hi >= 0",
                actual=th,
            )
        if all(v == 0 for v in th):
            raise GeoBrainError(
                "template_half must have at least one positive component",
                object_name="SNESIM",
                field="template_half",
                expected="not all zero",
                actual=th,
            )
        if n_multigrid < 1:
            raise GeoBrainError(
                "n_multigrid must be >= 1",
                object_name="SNESIM",
                field="n_multigrid",
                expected=">= 1",
                actual=n_multigrid,
            )
        if min_replicates < 1:
            raise GeoBrainError(
                "min_replicates must be >= 1",
                object_name="SNESIM",
                field="min_replicates",
                expected=">= 1",
                actual=min_replicates,
            )
        if max_cond < 1:
            raise GeoBrainError(
                "max_cond must be >= 1",
                object_name="SNESIM",
                field="max_cond",
                expected=">= 1",
                actual=max_cond,
            )
        if not isinstance(property, PropertyMetadata) or property.kind != "categorical":
            raise GeoBrainError(
                "SNESIM property must be categorical",
                object_name="SNESIM",
                field="property",
                expected="categorical PropertyMetadata",
                actual=type(property).__name__,
            )
        if not isinstance(execution, SimulationExecutionConfig):
            raise GeoBrainError(
                "execution must be SimulationExecutionConfig", object_name="SNESIM",
                field="execution", expected="SimulationExecutionConfig", actual=type(execution).__name__,
            )

        self.training_image: FloatArray = ti
        self.categories: list[int] = cats
        self.template_half = (th[0], th[1], th[2] if ti.shape[2] > 1 else 0)
        self.n_multigrid = int(n_multigrid)
        self.min_replicates = int(min_replicates)
        self.max_cond = int(max_cond)
        self.property = property
        self.execution = execution
        self.nsim = execution.n_realizations
        self.seed = execution.seed
        self.n_jobs = execution.workers if execution.worker_backend != "serial" else 1
        raw_ti = ti[..., 0] if ti.shape[2] == 1 else ti
        axes = ("x", "y") if raw_ti.ndim == 2 else ("x", "y", "z")
        self.training_image_spec = TrainingImageSpec(raw_ti, property, axes, np.isnan(raw_ti))
        self.training_image_index = TrainingImageIndex.build(
            self.training_image_spec, np.zeros((1, raw_ti.ndim), dtype=np.int64), execution.budget_bytes
        )

        # Marginal (target) proportions, used as prior fallback.
        counts = np.array([int(np.sum(ti_finite == c)) for c in cats], dtype=np.float64)
        total = counts.sum()
        self._prior = counts / total if total > 0 else np.ones(len(cats)) / len(cats)

        self._cat_index = {c: i for i, c in enumerate(self.categories)}
        # Per-level (offsets, search_tree) pairs are built lazily in __call__.

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
                "SNESIM requires a GeoGrid domain",
                object_name="SNESIM",
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
                object_name="SNESIM",
                field="training_image",
                expected=f"matching ndim ({3 if target_shape[2] > 1 else 2})",
                actual=f"target {target_shape}, TI {ti_shape}",
            )

        pre_filled = np.full(target_shape, np.nan, dtype=np.float64)
        if data is not None:
            if not isinstance(data, GeoFrame):
                raise GeoBrainError(
                    "data must be a GeoFrame or None",
                    object_name="SNESIM",
                    field="data",
                    expected="GeoFrame | None",
                    actual=type(data).__name__,
                )
            if not data.columns:
                raise GeoBrainError(
                    "data has no columns",
                    object_name="SNESIM",
                    field="data.columns",
                    expected="non-empty",
                    actual=[],
                )
            col = data.columns[0]
            fill_hard_data(grid, data, col, pre_filled)

        # Build per-level templates and SearchTree pattern stores.
        levels: list[tuple[list[tuple[int, int, int]], SearchTree]] = []
        base_offsets = _build_template_offsets(self.template_half)
        # Cap offsets at max_cond to keep the tree shallow.
        base_offsets = base_offsets[: self.max_cond]
        for lv in range(self.n_multigrid):
            scale = 2**lv
            scaled = [(dx * scale, dy * scale, dz * scale) for dx, dy, dz in base_offsets]
            tree = self._build_search_tree(scaled)
            levels.append((scaled, tree))

        seeds = _derive_realization_seeds(self.seed, self.nsim)

        worker = partial(
            _snesim_realization,
            simulator=self,
            pre_filled=pre_filled,
            levels=levels,
        )
        sims = run_realizations(worker, seeds, self.n_jobs)

        frames = tuple(make_simulation_frame(grid, sim, self.property) for sim in sims)
        return assemble_mps_ensemble(
            "SNESIM", self.property, self.execution, seeds, frames,
            cast(str, self.training_image_spec.fingerprint),
        )

    # ------------------------------------------------------------------

    def _build_search_tree(
        self,
        offsets: list[tuple[int, int, int]],
    ) -> SearchTree:
        """
        Scan TI; insert every (offset-pattern → centre-category) replicate
        into a :class:`SearchTree`.

        The tree key sequence is the *prefix* of the data event sorted by
        offset distance. Inserting a pattern of length *L* updates one
        node at each depth ``0..L``; the depth-``L`` node records the
        centre category. At query time, we walk as deep as the data
        event allows and pick the deepest valid (``>= min_replicates``)
        node, equivalent to tuple-truncation-from-the-right
        behaviour.
        """
        ti = self.training_image
        tnx, tny, tnz = ti.shape
        idx_map = self._cat_index
        tree = SearchTree(
            n_categories=len(self.categories),
            categories=list(self.categories),
        )

        for ck in range(tnz):
            for cj in range(tny):
                for ci in range(tnx):
                    cv = ti[ci, cj, ck]
                    if not np.isfinite(cv):
                        continue
                    if int(cv) not in idx_map:
                        continue
                    # Walk the template, skipping offsets that run off the
                    # TI or hit a non-finite value: informed nodes beyond
                    # a gap still contribute. The tree keys on
                    # ``(offset_index, value)`` so gapped paths stay
                    # consistent with simulation-time queries.
                    event: list[tuple[int, float]] = []
                    for k, (dx, dy, dz) in enumerate(offsets):
                        ti_i, ti_j, ti_k = ci + dx, cj + dy, ck + dz
                        if not (0 <= ti_i < tnx and 0 <= ti_j < tny and 0 <= ti_k < tnz):
                            continue
                        nv = ti[ti_i, ti_j, ti_k]
                        if not np.isfinite(nv):
                            continue
                        event.append((k, float(nv)))
                    tree.insert(event, int(cv))
        return tree

    # ------------------------------------------------------------------

    def _query(
        self,
        tree: SearchTree,
        data_event: list[tuple[int, float]],
    ) -> np.ndarray:
        """
        Return a ``(categories,)`` probability vector for the given
        event by querying the search tree.

        Falls back to the global marginal if the tree has no node whose
        replicate count meets ``min_replicates``.
        """
        counts, _ = tree.query(data_event, self.min_replicates)
        total = int(counts.sum())
        if total >= self.min_replicates:
            return counts.astype(np.float64) / total
        return self._prior.copy()

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _one_realisation(
        self,
        sim: np.ndarray,
        levels: list[tuple[list[tuple[int, int, int]], SearchTree]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        nx, ny, nz = sim.shape

        for lv in range(self.n_multigrid - 1, -1, -1):
            offsets, tree = levels[lv]
            scale = 2**lv
            step = max(1, scale)

            # Node partition by multigrid level.
            if lv == self.n_multigrid - 1:

                def node_filter(i, j, k, step=step):
                    return i % step == 0 and j % step == 0 and k % step == 0
            else:
                coarser = 2 ** (lv + 1)

                def node_filter(i, j, k, coarser=coarser, step=step):
                    on_level = i % step == 0 and j % step == 0 and k % step == 0
                    on_coarser = i % coarser == 0 and j % coarser == 0 and k % coarser == 0
                    return on_level and not on_coarser

            nodes: list[tuple[int, int, int]] = []
            for kk in range(nz):
                for jj in range(ny):
                    for ii in range(nx):
                        if not node_filter(ii, jj, kk):
                            continue
                        if np.isnan(sim[ii, jj, kk]):
                            nodes.append((ii, jj, kk))
            rng.shuffle(nodes)

            for i, j, k in nodes:
                # Build data event (offset-index, value), nearest-first.
                # Uninformed/out-of-bounds neighbours are SKIPPED (not
                # truncated at): informed neighbours beyond a gap still
                # condition the draw. The tree keys children on
                # ``(offset_index, value)`` so the walk advances past
                # gaps, matching ``_build_search_tree``.
                event: list[tuple[int, float]] = []
                for kdx, (dx, dy, dz) in enumerate(offsets):
                    ii, jj, kk_ = i + dx, j + dy, k + dz
                    if not (0 <= ii < nx and 0 <= jj < ny and 0 <= kk_ < nz):
                        continue
                    v = sim[ii, jj, kk_]
                    if np.isnan(v):
                        continue
                    event.append((kdx, float(v)))
                cpdf = self._query(tree, event)

                # Draw from cpdf
                r = float(rng.random())
                cum = 0.0
                drawn = self.categories[-1]
                for ci, cat in enumerate(self.categories):
                    cum += cpdf[ci]
                    if r <= cum:
                        drawn = cat
                        break
                sim[i, j, k] = float(drawn)

        # Fill any leftover NaN (shouldn't normally occur, but defensive).
        if np.any(np.isnan(sim)):
            mask = np.isnan(sim)
            draws = rng.choice(self.categories, size=int(mask.sum()))
            sim[mask] = draws.astype(np.float64)
        return sim

    def __repr__(self) -> str:
        return (
            f"SNESIM(ti_shape={self.training_image.shape}, "
            f"categories={self.categories}, "
            f"template_half={self.template_half}, "
            f"n_multigrid={self.n_multigrid}, nsim={self.nsim})"
        )
