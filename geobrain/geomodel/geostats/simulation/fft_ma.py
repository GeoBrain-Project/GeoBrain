"""FFT-MA: FFT-based Moving Average Gaussian simulation.

FFT-MA generates spatially-correlated Gaussian random fields by
filtering white noise in the spectral domain::

    Z = IFFT( sqrt(|FFT(C)|) · FFT(W) )

where ``C(h)`` is the covariance function of the requested
:class:`CovarianceModel`, ``W`` is standard-normal white noise, and the
FFT runs on a (possibly padded) cell-centred grid.

Compared to LUSIM the cost is ``O(N log N)`` instead of ``O(N³)``, so
million-cell grids are routine. The trade-off is that the spectrum is
sampled on a periodic grid, so an exponential **taper** is applied to
the covariance kernel before transforming. The taper matches the
``order=3, decay=0.25`` GSLIB convention.

Conditioning:
FFT-MA itself is *unconditional*. We add conditioning the standard way
(Journel & Huijbregts 1978, §VII): every unconditional draw
``Z_unc(x)`` is post-processed with the **residual kriging** correction

::

    Z_cond(x) = Z_unc(x) + Σ_i λ_i(x) [ d_i − Z_unc(s_i) ]

where ``λ_i(x)`` are simple-kriging weights computed once (the
covariance of the field is stationary and independent of the
realisation), and ``Z_unc(s_i)`` is the unconditional draw sampled at
the data locations via trilinear interpolation. The result honours the
hard data exactly when the data sit on grid nodes, and approximately
otherwise.

Anisotropy is honoured through the first nested structure's rotation
matrix; the nugget is treated as an extra white-noise contribution to
the spectrum so it does not require a separate FFT.

Memory:
The internal padded shape is ``2·shape`` (one cell of zero padding on
either side) capped by the next power of two for fast FFTs. Set
``padding=False`` to skip padding (slightly faster, but boundary
correlations leak through periodicity).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import numpy as np

from ...conditioning import ConditioningSet, normalize_conditioning
from ...errors import GeomodelContractError, GeomodelNumericsError
from ...frames._arrays import FloatArray, as_float_array
from ._grid_utils import flatten_gslib
from .._domain import (
    derive_realization_seeds as _derive_realization_seeds,
)
from ...frames import GeoFrame, GeoGrid, PropertyMetadata, gslib_grid_layout
from ..estimation.covariance_matrix import covariance_matrix
from ..models.covariance import CovarianceModel
from ..models.rotation import setup_rotation_matrix
from ._parallel import run_realizations
from ._parallel import RealizationRun
from .execution import SimulationExecutionConfig
from .agent_contract import SimulationAgentContract
from .results import SimulationEnsemble
from .sequential import assemble_ensemble, hard_conditioning, make_simulation_frame

__all__ = ["FFTEmbeddingPolicy", "FFTMA"]


@dataclass(frozen=True, slots=True)
class FFTEmbeddingPolicy:
    """Explicit covariance-embedding growth and roundoff tolerances.

    Attributes:
        max_growth_steps: embedding-size doubling attempts.
        negative_eigenvalue_relative: tolerated spectral negativity.
        imaginary_leakage_relative: tolerated imaginary leakage.
    """

    max_growth_steps: int = 4
    negative_eigenvalue_relative: float = -1.0e-12
    imaginary_leakage_relative: float = 1.0e-12

    def __post_init__(self) -> None:
        if isinstance(self.max_growth_steps, bool) or self.max_growth_steps < 0:
            raise GeomodelContractError(
                "FFT growth steps must be a non-negative integer",
                object_name=type(self).__name__, field="max_growth_steps",
                expected=">= 0", actual=self.max_growth_steps,
            )
        negative = float(self.negative_eigenvalue_relative)
        leakage = float(self.imaginary_leakage_relative)
        if not math.isfinite(negative) or negative > 0.0:
            raise GeomodelContractError(
                "FFT negative-eigenvalue tolerance must be finite and non-positive",
                object_name=type(self).__name__, field="negative_eigenvalue_relative",
                expected="finite value <= 0", actual=negative,
            )
        if not math.isfinite(leakage) or leakage < 0.0:
            raise GeomodelContractError(
                "FFT imaginary-leakage tolerance must be finite and non-negative",
                object_name=type(self).__name__, field="imaginary_leakage_relative",
                expected="finite value >= 0", actual=leakage,
            )
        object.__setattr__(self, "max_growth_steps", int(self.max_growth_steps))
        object.__setattr__(self, "negative_eigenvalue_relative", negative)
        object.__setattr__(self, "imaginary_leakage_relative", leakage)

    def to_dict(self) -> dict[str, object]:
        return {
            "max_growth_steps": self.max_growth_steps,
            "negative_eigenvalue_relative": self.negative_eigenvalue_relative,
            "imaginary_leakage_relative": self.imaginary_leakage_relative,
        }


def _next_pow2(n: int) -> int:
    """Smallest power of two greater than or equal to ``n``."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _fftma_realization(
    index: int,
    seed: int,
    *,
    simulator: "FFTMA",
    grid: GeoGrid,
    sqrt_spec: FloatArray,
    padded_shape: tuple[int, int, int],
    kriging_op: tuple[FloatArray, FloatArray] | None,
    cond_coords: FloatArray | None,
    cond_values: FloatArray | None,
) -> tuple[int, int, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(int(seed))
    field = simulator._draw_unconditional(
        rng,
        sqrt_spec,
        padded_shape,
        gslib_grid_layout(grid).shape,
    )

    if kriging_op is not None and cond_coords is not None and cond_values is not None:
        weights, cond_grid_idx = kriging_op
        field = simulator._apply_conditioning(
            field, grid, cond_coords, cond_values, weights, cond_grid_idx
        )

    field = field + simulator.mean
    return index, seed, flatten_gslib(field), {
        "embedding_shape": list(padded_shape),
        "conditioning_count": 0 if cond_values is None else int(cond_values.size),
    }


class FFTMA(SimulationAgentContract):
    """
    FFT-based Moving Average Gaussian simulator.

    Args:
        variogram: nested :class:`CovarianceModel`. Anisotropy and
            rotation are read from the *first* nested structure.
            Multiple structures are summed in lag
            space then transformed once.
        column: which column of ``data`` to condition on (default: the
            first column).
        nsim: number of realisations.
        seed: master RNG seed (per-realisation seeds derived from it).
        mean: stationary mean ``μ`` of the field.
        padding: if ``True`` (default) pad the FFT grid to the next
            power of two of ``2·shape``. Saves boundary leakage at the
            cost of ~8× the memory.
        taper_decay: exponential taper decay (default ``0.25``).
        taper_order: exponential taper order (default ``3``).
        n_jobs: realisation workers. ``1`` (default) serial; ``>1`` / ``-1`` /
            ``"auto"`` run realisations across processes. Results are
            identical to the serial run for any ``n_jobs`` (per-realisation
            seeds are fixed and order is preserved).
    """

    def __init__(
        self,
        model: CovarianceModel,
        *,
        property: PropertyMetadata,
        execution: SimulationExecutionConfig,
        embedding: FFTEmbeddingPolicy = FFTEmbeddingPolicy(),
        mean: float = 0.0,
        padding: bool = True,
        taper_decay: float = 0.25,
        taper_order: int = 3,
    ) -> None:
        if not isinstance(model, CovarianceModel):
            raise GeomodelContractError(
                "model must be a CovarianceModel",
                object_name="FFTMA",
                field="model",
                expected="CovarianceModel",
                actual=type(model).__name__,
            )
        if not model.structures:
            raise GeomodelContractError(
                "model must have at least one nested structure",
                object_name="FFTMA",
                field="model.structures",
                expected="non-empty",
                actual=[],
            )
        if not isinstance(property, PropertyMetadata) or property.kind != "continuous":
            raise GeomodelContractError(
                "FFTMA property must be continuous",
                object_name="FFTMA", field="property",
                expected="continuous PropertyMetadata", actual=type(property).__name__,
            )
        if not isinstance(execution, SimulationExecutionConfig):
            raise GeomodelContractError(
                "FFTMA requires SimulationExecutionConfig",
                object_name="FFTMA", field="execution",
                expected="SimulationExecutionConfig", actual=type(execution).__name__,
            )
        if not isinstance(embedding, FFTEmbeddingPolicy):
            raise GeomodelContractError(
                "FFTMA requires FFTEmbeddingPolicy",
                object_name="FFTMA", field="embedding",
                expected="FFTEmbeddingPolicy", actual=type(embedding).__name__,
            )
        if taper_decay <= 0.0:
            raise GeomodelContractError(
                "taper_decay must be positive",
                object_name="FFTMA",
                field="taper_decay",
                expected="> 0",
                actual=taper_decay,
            )
        if taper_order < 1:
            raise GeomodelContractError(
                "taper_order must be >= 1",
                object_name="FFTMA",
                field="taper_order",
                expected=">= 1",
                actual=taper_order,
            )
        model.require_stationary_covariance(object_name=type(self).__name__)
        self.variogram = model
        self.property = property
        self.execution = execution
        self.embedding = embedding
        self.mean = float(mean)
        self.padding = bool(padding)
        self.taper_decay = float(taper_decay)
        self.taper_order = int(taper_order)

    # ------------------------------------------------------------------

    def __call__(
        self,
        data: GeoFrame | ConditioningSet | None,
        domain: Any,
    ) -> SimulationEnsemble:
        if not isinstance(domain, GeoGrid):
            if isinstance(domain, GeoFrame) and isinstance(domain.geometry, GeoGrid):
                grid = domain.geometry
            else:
                raise GeomodelContractError(
                    "FFTMA requires a GeoGrid domain (or a GeoFrame wrapping one)",
                    object_name="FFTMA",
                    field="domain",
                    expected="GeoGrid",
                    actual=type(domain).__name__,
                )
        else:
            grid = domain

        conditioning = normalize_conditioning(data, grid, self.property)
        cond_coords, cond_values = hard_conditioning(conditioning)

        # Pre-compute the (cached, per-call) spectrum and conditioning
        # kriging operator (the latter depends only on data positions).
        sqrt_spec, padded_shape = self._build_sqrt_spectrum(grid)

        kriging_op: tuple[FloatArray, FloatArray] | None = None
        if cond_coords.size:
            kriging_op = self._build_kriging_op(grid, cond_coords)

        seeds = _derive_realization_seeds(self.execution.seed, self.execution.n_realizations)

        worker = partial(
            _fftma_realization,
            simulator=self,
            grid=grid,
            sqrt_spec=sqrt_spec,
            padded_shape=padded_shape,
            kriging_op=kriging_op,
            cond_coords=cond_coords if cond_coords.size else None,
            cond_values=cond_values if cond_values.size else None,
        )
        run = cast(RealizationRun, run_realizations(worker, seeds, self.execution))
        frames = tuple(
            make_simulation_frame(grid, item.result, self.property)
            for item in run.results
        )
        return assemble_ensemble(
            self.property,
            self.execution,
            run,
            frames,
            diagnostics={
                "algorithm": "FFTMA",
                "property": self.property.to_dict(),
                "embedding": self.embedding.to_dict(),
                "embedding_shape": list(padded_shape),
                "conditioning": conditioning.diagnostics,
            },
        )

    # ------------------------------------------------------------------
    # Spectrum construction (cached per __call__)
    # ------------------------------------------------------------------

    def _padded_shape(self, shape: tuple[int, int, int]) -> tuple[int, int, int]:
        nx, ny, nz = shape
        if not self.padding:
            return (nx, ny, nz)
        return (
            _next_pow2(2 * nx),
            _next_pow2(2 * ny),
            _next_pow2(2 * nz) if nz > 1 else 1,
        )

    def _build_sqrt_spectrum(self, grid: GeoGrid) -> tuple[FloatArray, tuple[int, int, int]]:
        """Build sqrt(|FFT(C·taper)|), growing the embedding when inadmissible.

        The initial padded shape may yield a spectrum with materially negative
        eigenvalues (a long-range covariance on a small domain is not
        circulant-embeddable there). ``embedding.max_growth_steps`` bounds how
        many times the embedding may double before the failure is raised with
        every attempted shape. ``padding=False`` pins the embedding to the
        domain and therefore never grows.
        """
        layout = gslib_grid_layout(grid)
        shape = layout.shape
        padded = self._padded_shape(shape)
        growth_steps = self.embedding.max_growth_steps if self.padding else 0
        attempted: list[list[int]] = []
        minimum = threshold = 0.0
        for _ in range(growth_steps + 1):
            attempted.append(list(padded))
            sqrt_spec, minimum, threshold = self._spectrum_attempt(layout, padded)
            if sqrt_spec is not None:
                return sqrt_spec, padded
            padded = (
                _next_pow2(2 * padded[0]),
                _next_pow2(2 * padded[1]),
                _next_pow2(2 * padded[2]) if padded[2] > 1 else 1,
            )
        raise GeomodelNumericsError(
            "FFT covariance embedding is not positive semidefinite",
            object_name=type(self).__name__, field="spectrum",
            expected={"minimum_eigenvalue": f">= {threshold}"},
            actual={"minimum_eigenvalue": minimum, "attempted_shapes": attempted},
        )

    def _spectrum_attempt(
        self,
        layout: Any,
        padded: tuple[int, int, int],
    ) -> tuple[FloatArray | None, float, float]:
        """Try one embedding shape; return (sqrt_spec | None, minimum, threshold)."""
        nx, ny, nz = padded

        # Centred index grids (lag in cells). Always build a 3-D field so
        # downstream FFT / cropping see a consistent ``(nx, ny, nz)`` shape
        # even for the 2-D case (``nz == 1``).
        ix = np.arange(nx, dtype=np.float64) - nx / 2.0
        iy = np.arange(ny, dtype=np.float64) - ny / 2.0
        iz = np.arange(nz, dtype=np.float64) - nz / 2.0
        Ix, Iy, Iz = np.meshgrid(ix, iy, iz, indexing="ij")

        # Lag in physical units.
        lx = Ix * layout.spacing_m[0]
        ly = Iy * layout.spacing_m[1]
        lz = Iz * layout.spacing_m[2]

        # Build covariance C(h) = sill − γ(h) summed over nested structures
        # using each structure's own rotation/anisotropy. We replicate
        # covariance_matrix's logic for a *single* lag-vector field.
        cov = np.zeros_like(lx)
        deltas = np.stack([lx, ly, lz], axis=-1)  # (..., 3)
        for struct in self.variogram.structures:
            R = setup_rotation_matrix(
                struct.angles[0],
                struct.angles[1],
                struct.angles[2],
                struct.anis1,
                struct.anis2,
            )
            rotated = deltas @ R.T
            h = np.sqrt(np.sum(rotated * rotated, axis=-1))
            gamma = struct.evaluate(h)
            cov = cov + struct.contribution - gamma

        # Nugget at lag=0 (single cell).
        if self.variogram.nugget > 0.0:
            iso_sq = lx * lx + ly * ly + lz * lz
            cov = cov + np.where(iso_sq < 1e-20, self.variogram.nugget, 0.0)

        # Exponential taper to enforce a smooth periodic boundary.
        order = self.taper_order
        decay = self.taper_decay
        taper = np.exp(
            -(
                (np.abs(Ix) / max(decay * nx, 1.0)) ** order
                + (np.abs(Iy) / max(decay * ny, 1.0)) ** order
                + (np.abs(Iz) / max(decay * nz, 1.0)) ** order
            )
        )
        cov = cov * taper

        # ifftshift so that the lag-zero sits at index 0 (FFT convention).
        cov = np.fft.ifftshift(cov)
        spectrum = np.fft.fftn(cov)
        spectral_scale = max(float(np.max(np.absolute(spectrum.real))), 1.0)
        imaginary_leakage = float(np.max(np.absolute(spectrum.imag)))
        if imaginary_leakage > self.embedding.imaginary_leakage_relative * spectral_scale:
            raise GeomodelNumericsError(
                "FFT covariance embedding has material imaginary leakage",
                object_name=type(self).__name__, field="spectrum",
                expected={"imaginary_leakage_relative": self.embedding.imaginary_leakage_relative},
                actual={"imaginary_leakage": imaginary_leakage, "spectral_scale": spectral_scale},
            )
        real_spec = spectrum.real
        minimum = float(np.min(real_spec))
        threshold = self.embedding.negative_eigenvalue_relative * spectral_scale
        if minimum < threshold:
            return None, minimum, threshold
        admissible_spec = np.where(real_spec < 0.0, 0.0, real_spec)
        # Deterministic normalisation: taking |·| (and any taper-induced
        # spectral leakage) perturbs the implied lag-0 covariance
        # C(0) = mean(|spectrum|) away from the model C(0) (= sill; the
        # taper is 1 at lag zero). Rescale the spectrum ONCE so every
        # draw's ensemble variance honours the sill: individual
        # realizations keep their natural ergodic fluctuation.
        c0_model = float(self.variogram.sill)
        c0_implied = float(admissible_spec.mean())
        if c0_model > 0.0 and c0_implied > 0.0:
            admissible_spec = admissible_spec * (c0_model / c0_implied)
        sqrt_spec = as_float_array(np.sqrt(admissible_spec))
        return sqrt_spec, minimum, threshold

    def _draw_unconditional(
        self,
        rng: np.random.Generator,
        sqrt_spec: FloatArray,
        padded_shape: tuple[int, int, int],
        target_shape: tuple[int, int, int],
    ) -> FloatArray:
        """Draw one unconditional realisation; crop to the target grid."""
        nx, ny, nz = padded_shape
        # White noise (complex N(0,1) via real+imag halves).
        noise = rng.standard_normal(padded_shape).astype(np.float64)
        noise_fft = np.fft.fftn(noise)
        filtered_fft = sqrt_spec * noise_fft
        field = np.real(np.fft.ifftn(filtered_fft))

        # Crop to the requested grid (drop the padding). No per-draw
        # rescaling: the spectrum is normalised once (deterministically)
        # in ``_build_sqrt_spectrum`` so C(0) equals the sill, and each
        # realization keeps its ergodic variance fluctuation (including
        # the legitimate dispersion variance on small domains).
        tx, ty, tz = target_shape
        return as_float_array(field[:tx, :ty, :tz])

    # ------------------------------------------------------------------
    # Conditioning via simple-kriging residuals
    # ------------------------------------------------------------------

    def _build_kriging_op(
        self, grid: GeoGrid, cond_coords: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """
        Build SK weights for every grid cell against the data set.

        Returns ``(weights, cond_grid_idx)`` where ``weights`` has
        shape ``(ncells, ndata)`` and ``cond_grid_idx`` is the
        ``(ndata, 3)`` nearest-cell ``(i, j, k)`` index for each datum
        (used to look up the unconditional draw at the data locations).
        """
        grid_coords = grid.coords
        n_cond = cond_coords.shape[0]

        cov_dd = covariance_matrix(self.variogram, cond_coords, cond_coords)
        cov_gd = covariance_matrix(self.variogram, grid_coords, cond_coords)

        cov_dd_reg = cov_dd + np.eye(n_cond) * 1e-10
        try:
            # weights = cov_gd · cov_dd⁻¹    shape (ncells, ndata)
            weights = as_float_array(np.linalg.solve(cov_dd_reg, cov_gd.T).T)
        except np.linalg.LinAlgError as exc:
            raise GeomodelNumericsError(
                "FFTMA conditioning covariance is singular",
                object_name=type(self).__name__, field="conditioning",
                expected="nonsingular covariance matrix", actual="singular",
            ) from exc

        # Nearest-cell index for each datum (snap to grid). Cell centres
        # sit at ``(i + 0.5)·xsiz + xmn``; inverting gives
        # ``i = floor((x - xmn)/xsiz)``.
        cond_grid_idx = np.zeros((n_cond, 3), dtype=np.int64)
        layout = gslib_grid_layout(grid)
        for k in range(n_cond):
            i = int(np.floor((cond_coords[k, 0] - grid.xmn) / grid.xsiz))
            j = int(np.floor((cond_coords[k, 1] - grid.ymn) / grid.ysiz))
            kk = int(np.floor((cond_coords[k, 2] - layout.origin_m[2]) / layout.spacing_m[2]))
            cond_grid_idx[k, 0] = min(max(i, 0), grid.nx - 1)
            cond_grid_idx[k, 1] = min(max(j, 0), grid.ny - 1)
            cond_grid_idx[k, 2] = min(max(kk, 0), layout.shape[2] - 1)
        return weights, as_float_array(cond_grid_idx.astype(np.float64))

    def _apply_conditioning(
        self,
        field: FloatArray,
        grid: GeoGrid,
        cond_coords: FloatArray,
        cond_values: FloatArray,
        weights: FloatArray,
        cond_grid_idx_f: FloatArray,
    ) -> FloatArray:
        """
        Add the SK residual correction in place, then snap to data.

        Implements ``Z_cond = Z_unc + W · (d − Z_unc(s_i))`` with weights
        ``W`` flattened over grid cells (GSLIB x-fastest), then reshaped
        back to ``(nx, ny, nz)``. After the smooth correction we snap
        the cell value at each datum's nearest cell to the datum
        exactly; this absorbs the small residual error introduced by
        the tapered spectrum, whose implied covariance deviates
        slightly from the theoretical model used to build ``W``.
        """
        # 1) Look up Z_unc at each datum's nearest cell.
        cond_grid_idx = cond_grid_idx_f.astype(np.int64)
        z_unc_at_data = as_float_array(
            field[cond_grid_idx[:, 0], cond_grid_idx[:, 1], cond_grid_idx[:, 2]]
        )

        # 2) Residual at data (relative to the SK mean = 0 in residual space).
        residual = cond_values - self.mean - z_unc_at_data  # (ndata,)

        # 3) Spread the residual onto every grid cell.
        correction_flat = weights @ residual  # (ncells,)

        # 4) GSLIB flat → (nx, ny, nz). GeoGrid.coords uses
        #    F-order: i fastest, then j, then k.
        nx, ny, nz = gslib_grid_layout(grid).shape
        correction = correction_flat.reshape((nz, ny, nx)).transpose(2, 1, 0)
        corrected = as_float_array(field + correction)

        # 5) Snap to data at the nearest cells (matches LUSIM/SGSIM).
        for k in range(cond_grid_idx.shape[0]):
            ii, jj, kk = cond_grid_idx[k]
            corrected[ii, jj, kk] = float(cond_values[k]) - self.mean
        return corrected

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FFTMA(model={self.variogram!r}, "
            f"mean={self.mean}, padding={self.padding})"
        )
