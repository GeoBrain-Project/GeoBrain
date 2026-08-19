"""Direct Sampling: multi-point statistics simulation (Mariethoz 2010).

Pattern-based simulator that draws from a **training image** rather
than a parametric variogram. At each unsimulated node of the target
grid the algorithm:

1. Builds a *data event* from the nearest ``n_neighbors`` informed
   cells (hard data + already-simulated nodes), represented as a
   list of integer ``(dx, dy, dz)`` offsets together with their
   values.
2. Scans **random** locations in the training image. At each
   candidate it extracts the same offset pattern and computes a
   distance to the target event:

   - **categorical**: fraction of mismatches (Hamming).
   - **continuous**:  RMSE normalised by the TI's value range.

3. If the distance is below ``threshold`` the candidate's centre
   value is accepted. After scanning ``max_search_fraction`` of the
   TI without acceptance, the algorithm uses the best candidate
   seen so far.

References:
- Mariethoz, G., Renard, P. & Straubhaar, J. (2010). The Direct
  Sampling method to perform multiple-point geostatistical
  simulations. *Water Resources Research* 46, W11536.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from functools import partial
from typing import cast

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import FloatArray, as_float_array, as_int_array
from ._grid_utils import fill_hard_data, flatten_gslib
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

__all__ = ["DirectSampling"]


def _to_5d_index(idx: int, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert flat index to ``(i, j, k)`` using GSLIB x-fastest ordering."""
    nx, ny, _nz = shape
    k = idx // (nx * ny)
    rem = idx % (nx * ny)
    j = rem // nx
    i = rem % nx
    return (i, j, k)


def _flat_index(ijk: tuple[int, int, int], shape: tuple[int, int, int]) -> int:
    nx, ny, _nz = shape
    i, j, k = ijk
    return int(i + j * nx + k * nx * ny)


def _direct_sampling_realization(
    seed: int,
    *,
    simulator: "DirectSampling",
    pre_filled: np.ndarray,
) -> np.ndarray:
    field = simulator._one_realisation(
        pre_filled.copy(),
        np.random.default_rng(int(seed)),
    )
    return flatten_gslib(field)


class DirectSampling(SimulationAgentContract):
    """
    Direct-Sampling MPS simulator.

    Args:
        training_image: 2-D ``(nx, ny)`` or 3-D ``(nx, ny, nz)`` numpy
            array representing the training image. NaNs are treated
            as "no information" and skipped during pattern extraction.
        n_neighbors: maximum number of informed neighbours used to
            build each data event (default 30).
        threshold: acceptance distance (default 0.05). For
            categorical TIs this is a mismatch fraction; for
            continuous TIs it is RMSE normalised by ``ptp(TI)``.
        max_search_fraction: stop scanning the TI after this fraction
            of its cells have been tried (default 0.25). The best
            candidate seen so far is accepted.
        categorical: if ``True``, use Hamming distance; otherwise
            normalised RMSE.
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
        *,
        property: PropertyMetadata,
        execution: SimulationExecutionConfig,
        n_neighbors: int = 30,
        threshold: float = 0.05,
        max_search_fraction: float = 0.25,
    ) -> None:
        ti = as_float_array(training_image)
        if ti.ndim not in (2, 3):
            raise GeoBrainError(
                "training_image must be 2-D or 3-D",
                object_name="DirectSampling",
                field="training_image",
                expected="ndim 2 or 3",
                actual=ti.ndim,
            )
        if ti.ndim == 2:
            ti = as_float_array(ti[..., None])  # promote to 3-D
        if any(s <= 0 for s in ti.shape):
            raise GeoBrainError(
                "training_image must have positive dimensions",
                object_name="DirectSampling",
                field="training_image",
                expected="all dims > 0",
                actual=ti.shape,
            )
        if n_neighbors < 1:
            raise GeoBrainError(
                "n_neighbors must be >= 1",
                object_name="DirectSampling",
                field="n_neighbors",
                expected=">= 1",
                actual=n_neighbors,
            )
        if not (0.0 < threshold):
            raise GeoBrainError(
                "threshold must be positive",
                object_name="DirectSampling",
                field="threshold",
                expected="> 0",
                actual=threshold,
            )
        if not (0.0 < max_search_fraction <= 1.0):
            raise GeoBrainError(
                "max_search_fraction must be in (0, 1]",
                object_name="DirectSampling",
                field="max_search_fraction",
                expected="(0, 1]",
                actual=max_search_fraction,
            )
        if not isinstance(property, PropertyMetadata):
            raise GeoBrainError(
                "property must be PropertyMetadata",
                object_name="DirectSampling",
                field="property",
                expected="PropertyMetadata",
                actual=type(property).__name__,
            )
        if not isinstance(execution, SimulationExecutionConfig):
            raise GeoBrainError(
                "execution must be SimulationExecutionConfig",
                object_name="DirectSampling", field="execution",
                expected="SimulationExecutionConfig", actual=type(execution).__name__,
            )

        self.training_image: FloatArray = ti
        self.n_neighbors = int(n_neighbors)
        self.threshold = float(threshold)
        self.max_search_fraction = float(max_search_fraction)
        self.property = property
        self.execution = execution
        self.categorical = property.kind == "categorical"
        self.nsim = execution.n_realizations
        self.seed = execution.seed
        self.n_jobs = execution.workers if execution.worker_backend != "serial" else 1
        raw_ti = ti[..., 0] if ti.shape[2] == 1 else ti
        axes = ("x", "y") if raw_ti.ndim == 2 else ("x", "y", "z")
        self.training_image_spec = TrainingImageSpec(raw_ti, property, axes, np.isnan(raw_ti))
        template = np.zeros((1, raw_ti.ndim), dtype=np.int64)
        self.training_image_index = TrainingImageIndex.build(
            self.training_image_spec, template, execution.budget_bytes
        )

        ti_finite = ti[np.isfinite(ti)]
        if ti_finite.size == 0:
            raise GeoBrainError(
                "training_image has no finite values",
                object_name="DirectSampling",
                field="training_image",
                expected="at least one finite value",
                actual=0,
            )
        # Range used to normalise the continuous-distance computation.
        self._ti_range = float(np.ptp(ti_finite)) or 1.0

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
                "DirectSampling requires a GeoGrid domain",
                object_name="DirectSampling",
                field="domain",
                expected="GeoGrid (or GeoFrame wrapping one)",
                actual=type(geom).__name__,
            )

        grid = geom
        target_shape = gslib_grid_layout(grid).shape
        ti_shape = self.training_image.shape  # always 3-D after promotion

        if (target_shape[2] > 1) != (ti_shape[2] > 1):
            raise GeoBrainError(
                "target grid and training image must have matching dimensionality",
                object_name="DirectSampling",
                field="training_image",
                expected=f"matching ndim ({3 if target_shape[2] > 1 else 2})",
                actual=f"target {target_shape}, TI {ti_shape}",
            )

        # Pre-fill grid with hard data
        pre_filled = np.full(target_shape, np.nan, dtype=np.float64)
        if data is not None:
            if not isinstance(data, GeoFrame):
                raise GeoBrainError(
                    "data must be a GeoFrame or None",
                    object_name="DirectSampling",
                    field="data",
                    expected="GeoFrame | None",
                    actual=type(data).__name__,
                )
            if not data.columns:
                raise GeoBrainError(
                    "data has no columns",
                    object_name="DirectSampling",
                    field="data.columns",
                    expected="non-empty",
                    actual=[],
                )
            col = data.columns[0]
            fill_hard_data(grid, data, col, pre_filled)

        seeds = _derive_realization_seeds(self.seed, self.nsim)

        worker = partial(
            _direct_sampling_realization,
            simulator=self,
            pre_filled=pre_filled,
        )
        sims = run_realizations(worker, seeds, self.n_jobs)

        frames = tuple(make_simulation_frame(grid, sim, self.property) for sim in sims)
        return assemble_mps_ensemble(
            "DirectSampling", self.property, self.execution, seeds, frames,
            cast(str, self.training_image_spec.fingerprint),
        )

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _one_realisation(
        self,
        sim: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        nx, ny, nz = sim.shape
        n_total = nx * ny * nz
        path = rng.permutation(n_total)

        ti = self.training_image
        tnx, tny, tnz = ti.shape
        max_candidates = max(1, int(self.max_search_fraction * tnx * tny * tnz))
        ti_finite = ti[np.isfinite(ti)]
        unconditional_pool = ti_finite  # used when data event is empty

        for flat in path:
            i, j, k = _to_5d_index(int(flat), (nx, ny, nz))
            if not np.isnan(sim[i, j, k]):
                continue

            offsets, neigh_values = self._collect_neighbours(sim, i, j, k)

            if offsets.shape[0] == 0:
                # No informed neighbours yet: pick any value from the TI.
                sim[i, j, k] = float(rng.choice(unconditional_pool))
                continue

            best_value = float("nan")
            best_dist = np.inf

            for _ in range(max_candidates):
                ci = int(rng.integers(0, tnx))
                cj = int(rng.integers(0, tny))
                ck = int(rng.integers(0, tnz))
                value = float(ti[ci, cj, ck])
                if not np.isfinite(value):
                    continue
                dist = self._pattern_distance(ti, ci, cj, ck, offsets, neigh_values)
                if dist < best_dist:
                    best_dist = dist
                    best_value = value
                if dist <= self.threshold:
                    break

            sim[i, j, k] = (
                best_value if np.isfinite(best_value) else float(rng.choice(unconditional_pool))
            )

        return sim

    # ------------------------------------------------------------------

    def _collect_neighbours(
        self,
        sim: np.ndarray,
        i: int,
        j: int,
        k: int,
    ) -> tuple[np.ndarray, FloatArray]:
        """
        Return ``(offsets, values)`` for the ``n_neighbors`` nearest
        informed cells around ``(i, j, k)``."""
        # Iterate informed cells; cheaper than KD-tree for small grids.
        informed = np.argwhere(np.isfinite(sim))
        if informed.size == 0:
            return np.zeros((0, 3), dtype=np.int64), as_float_array([])
        delta = informed - np.array([i, j, k])
        dist_sq = np.sum(delta * delta, axis=1)
        order = np.argsort(dist_sq, kind="mergesort")
        keep = order[: self.n_neighbors]
        sel = informed[keep]
        offsets = as_int_array(sel - np.array([i, j, k]))
        values = as_float_array(sim[sel[:, 0], sel[:, 1], sel[:, 2]])
        return offsets, values

    def _pattern_distance(
        self,
        ti: np.ndarray,
        ci: int,
        cj: int,
        ck: int,
        offsets: np.ndarray,
        target_values: FloatArray,
    ) -> float:
        """Compute the chosen distance between target event and TI event."""
        tnx, tny, tnz = ti.shape
        diffs: list[float] = []
        mismatches = 0
        compared = 0
        for n, (dx, dy, dz) in enumerate(offsets):
            ti_i, ti_j, ti_k = ci + int(dx), cj + int(dy), ck + int(dz)
            if not (0 <= ti_i < tnx and 0 <= ti_j < tny and 0 <= ti_k < tnz):
                # Out of TI: treat as a hard mismatch.
                if self.categorical:
                    mismatches += 1
                    compared += 1
                else:
                    return float("inf")
                continue
            ti_val = float(ti[ti_i, ti_j, ti_k])
            if not np.isfinite(ti_val):
                if self.categorical:
                    mismatches += 1
                    compared += 1
                continue
            tgt = float(target_values[n])
            if self.categorical:
                compared += 1
                if ti_val != tgt:
                    mismatches += 1
            else:
                diffs.append(tgt - ti_val)
                compared += 1
        if compared == 0:
            return float("inf")
        if self.categorical:
            return mismatches / compared
        diffs_arr = np.asarray(diffs, dtype=np.float64)
        rmse = float(np.sqrt(np.mean(diffs_arr**2)))
        return rmse / self._ti_range

    def __repr__(self) -> str:
        return (
            f"DirectSampling(ti_shape={self.training_image.shape}, "
            f"n_neighbors={self.n_neighbors}, threshold={self.threshold}, "
            f"categorical={self.categorical}, nsim={self.nsim})"
        )
