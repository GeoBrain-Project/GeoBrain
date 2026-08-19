"""Spatial Bayes-action decision accuracy for linear acquisitions.

The PCA and noise-free linear-Gaussian conditioning model follows the existing
spatial estimator. The decision functional is the correct Bayes action value:
for every cell, both binary states and every explicit action are considered
before and after the possible acquisition outcomes.

The supplied ``utilities[action, state]`` matrix is an accuracy payoff. Values
must be finite probabilities in ``[0, 1]``; the identity matrix represents
ordinary correct/incorrect classification. Raw gain is therefore in accuracy
probability. Optional normalization divides by the perfect-information ceiling
for that cell. No arbitrary scale factor or post-hoc clipping is applied.

Reference: Caers, Scheidt, Yin, Wang, Mukerji & House, "Efficacy of Information
in Mineral Exploration Drilling", *Natural Resources Research* 31(3):1157 (2022),
doi:10.1007/s11053-022-10030-1. Independent GeoBrain-native reimplementation.

Design note (accepted platform deviation): the PCA + linear-Gauss conditioning
runs in numpy/scipy (``svd``, ``pinv``, ``ndtr``), not torch; this is a
non-differentiable scoring layer over a frozen realization ensemble, so there is
no autograd graph to keep. Torch inputs are moved to CPU numpy on entry and
results wrapped back to torch (a CUDA ensemble pays a one-time host transfer).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import logging
import math
import time
from importlib import import_module
from typing import Any, Literal, Protocol, Sequence, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from geobrain.core.validation import (
    validate_non_negative_int,
)
from geobrain.core.errors import GeoBrainError
from geobrain.decision.accuracy import (
    DecisionAccuracyResult,
    expected_utility_gain,
)

logger = logging.getLogger(__name__)


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


class _NormalCdf(Protocol):
    def __call__(self, values: ArrayLike) -> Any: ...


ndtr = cast(
    _NormalCdf,
    getattr(import_module("scipy.special"), "ndtr"),
)


class SpatialDecisionAccuracy:
    """Spatial decision-accuracy gain for a linear acquisition.

    Models the prior with a PCA of the realization ensemble and conditions on a
    cell-sampling operator with the exact noise-free linear-Gaussian posterior.
    The binary target indicator is obtained by truncating the continuous PCA
    field at ``threshold``.

    Args:
        realizations: Prior ensemble, shape ``[n_realizations, *grid]``,
            typically binary (0/1) geostatistical indicator realizations.
        utilities: Finite ``[n_actions, 2]`` accuracy payoff matrix. Values
            must be in ``[0, 1]`` so reported raw values remain probabilities.
            Use ``numpy.eye(2)`` for ordinary correct-classification accuracy.
        threshold: Truncation level mapping the continuous field to the binary
            target (``indicator = 1`` where field > threshold). Default 0.5
            (the natural midpoint for 0/1 indicators).
        variance_fraction: Cumulative PCA variance to retain (paper: 0.8).
        normalize: Divide gain by its explicit perfect-information ceiling.
        n_bootstrap: Acquisition-outcome bootstrap replicates for per-cell gain
            standard error. Zero disables uncertainty and returns ``None``.
        bootstrap_seed: Seed for the independent bootstrap RNG.
        device: Torch device for the result tensors.

    Example:
        >>> estimator = SpatialDecisionAccuracy(realizations, np.eye(2))
        >>> result = estimator.compute(borehole_cells)
        >>> print(result.summary()); result.gain_map
    """

    def __init__(
        self,
        realizations: Any,
        utilities: Any,
        threshold: float = 0.5,
        variance_fraction: float = 0.8,
        normalize: bool = True,
        n_bootstrap: int = 0,
        bootstrap_seed: int = 0,
        device: torch.device | None = None,
    ) -> None:
        if isinstance(realizations, torch.Tensor):
            realizations = realizations.detach().cpu().numpy()
        try:
            arr = np.asarray(realizations, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GeoBrainError(
                "realizations must be a numeric ensemble",
                object_name="SpatialDecisionAccuracy",
                field="realizations",
                expected="[n_realizations, *grid] numeric array",
                actual=type(realizations),
            ) from exc
        if (
            arr.ndim < 2
            or arr.size == 0
            or not bool(np.all(np.isfinite(arr)))
        ):
            raise GeoBrainError(
                "realizations must be a non-empty finite ensemble",
                object_name="SpatialDecisionAccuracy",
                field="realizations",
                expected="finite [n_realizations, *grid] array",
                actual=tuple(arr.shape),
            )
        self.grid_shape = arr.shape[1:]
        n = arr.shape[0]
        if n < 2:
            raise GeoBrainError(
                "need at least 2 realizations",
                object_name="SpatialDecisionAccuracy",
                field="realizations", expected=">= 2", actual=n,
            )
        try:
            raw_utility = np.asarray(utilities)
            if np.issubdtype(raw_utility.dtype, np.bool_):
                raise TypeError("bool utilities are not accuracy probabilities")
            utility = np.asarray(utilities, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GeoBrainError(
                "utilities must be a numeric action-by-state matrix",
                object_name="SpatialDecisionAccuracy",
                field="utilities",
                expected="finite [n_actions, 2] values in [0, 1]",
                actual=utilities,
            ) from exc
        if (
            utility.ndim != 2
            or utility.shape[0] < 1
            or utility.shape[1] != 2
            or not bool(np.all(np.isfinite(utility)))
            or bool(np.any(utility < 0.0))
            or bool(np.any(utility > 1.0))
        ):
            raise GeoBrainError(
                "utilities must be finite binary-state accuracy probabilities",
                object_name="SpatialDecisionAccuracy",
                field="utilities",
                expected="finite [n_actions, 2] values in [0, 1]",
                actual=utilities,
            )
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
        ):
            raise GeoBrainError(
                "threshold must be finite",
                object_name="SpatialDecisionAccuracy",
                field="threshold",
                expected="finite number",
                actual=threshold,
            )
        if (
            isinstance(variance_fraction, bool)
            or not isinstance(variance_fraction, (int, float))
            or not math.isfinite(float(variance_fraction))
            or not 0.0 < float(variance_fraction) <= 1.0
        ):
            raise GeoBrainError(
                "variance_fraction must be in (0, 1]",
                object_name="SpatialDecisionAccuracy",
                field="variance_fraction",
                expected="0 < variance_fraction <= 1",
                actual=variance_fraction,
            )
        if not isinstance(normalize, bool):
            raise GeoBrainError(
                "normalize must be a bool",
                object_name="SpatialDecisionAccuracy",
                field="normalize",
                expected=bool,
                actual=type(normalize),
            )
        self.n_bootstrap = validate_non_negative_int(
            n_bootstrap,
            owner="SpatialDecisionAccuracy",
            field="n_bootstrap",
        )
        bootstrap_seed = validate_non_negative_int(
            bootstrap_seed,
            owner="SpatialDecisionAccuracy",
            field="bootstrap_seed",
        )
        self.threshold = float(threshold)
        self.utilities = utility.copy()
        self.normalize = normalize
        self.bootstrap_seed = bootstrap_seed
        self.device = device

        x = arr.reshape(n, -1)                       # (L, n_cells)
        self._mean = x.mean(axis=0)                  # v(x)
        xc = x - self._mean[None, :]
        # Economy SVD: xc = U diag(s) Wt ; PC images = rows of Wt, scores = U*s.
        u, s, wt = np.linalg.svd(xc, full_matrices=False)
        var = s ** 2
        if var.sum() <= 0:
            raise GeoBrainError(
                "realization ensemble has zero variance",
                object_name="SpatialDecisionAccuracy", field="realizations",
                expected="non-degenerate ensemble", actual="zero variance",
            )
        # Keep the full spectrum / basis (for the scree plot, eigen-images and
        # resampling); decision scoring uses only the retained slice.
        self._var_full = var                          # (r,) singular-value²
        self._W_full = wt.T                           # (n_cells, r) eigen-images
        self._scores_full = u * s                     # (L, r) ensemble PC coeffs
        self._n_real = n
        cum = np.cumsum(var) / var.sum()
        k = int(np.searchsorted(cum, float(variance_fraction)) + 1)
        k = max(1, min(k, s.shape[0]))
        self.n_components = k
        self._W = self._W_full[:, :k]                 # (n_cells, k) eigen-images
        self._lam = var[:k] / (n - 1)                 # (k,) prior PC variances
        self._scores = self._scores_full[:, :k]       # (L, k) per-realization coeffs
        # Prior per-cell probability = empirical ensemble proportion (paper's p(x)).
        self._prior_prob = np.clip(self._mean, 0.0, 1.0)
        self._prior_variance = x.var(axis=0)          # empirical ensemble variance

    # ------------------------------------------------------------------
    # PCA-prior introspection (the paper's Bayesian-PCA diagnostics)
    # ------------------------------------------------------------------

    @property
    def eigenvalues(self) -> FloatArray:
        """Covariance eigenvalues (full spectrum), largest first: for the scree plot."""
        return cast(FloatArray, self._var_full / (self._n_real - 1))

    @property
    def explained_variance_ratio(self) -> FloatArray:
        """Cumulative fraction of variance explained by the leading components."""
        return cast(
            FloatArray,
            np.cumsum(self._var_full) / self._var_full.sum(),
        )

    @property
    def prior_variance(self) -> FloatArray:
        """Empirical per-cell ensemble variance, on the model grid."""
        return cast(
            FloatArray,
            self._prior_variance.reshape(self.grid_shape),
        )

    @property
    def prior_mean(self) -> FloatArray:
        """Continuous ensemble mean field (the PCA reconstruction origin)."""
        return cast(FloatArray, self._mean.reshape(self.grid_shape))

    def eigen_images(self, n: int = 4) -> FloatArray:
        """First ``n`` eigenvectors ("eigen-images"), each on the model grid."""
        n = min(int(n), self._W_full.shape[1])
        return cast(
            FloatArray,
            self._W_full[:, :n].T.reshape((n, *self.grid_shape)),
        )

    def scores(self, n_components: int = 4) -> FloatArray:
        """Ensemble PC scores ``[n_realizations, n_components]`` (the SIS scores)."""
        return cast(FloatArray, self._scores_full[:, :int(n_components)])

    def sample(
        self,
        n: int,
        *,
        n_components: int | None = None,
        truncate: bool = False,
        seed: int | None = None,
    ) -> FloatArray:
        """Generate prior realizations from the PCA model.

        Draws ``y ~ N(0, Λ)`` over the leading ``n_components`` PCs (default: the
        80%-variance retained set) and reconstructs ``m = v + W y``. With
        ``truncate=True`` the continuous field is thresholded to the binary
        indicator (paper Fig. 4). Using *all* components reintroduces the small-
        eigenvalue noise (a nugget; paper Fig. 5).
        """
        kk = self._var_full.shape[0] if n_components is None else int(n_components)
        kk = max(1, min(kk, self._var_full.shape[0]))
        lam = self._var_full[:kk] / (self._n_real - 1)
        rng = np.random.default_rng(seed)
        y = rng.standard_normal((int(n), kk)) * np.sqrt(lam)[None, :]
        fields = self._mean[None, :] + y @ self._W_full[:, :kk].T
        if truncate:
            fields = (fields > self.threshold).astype(np.float64)
        return cast(
            FloatArray,
            fields.reshape((int(n), *self.grid_shape)),
        )

    def _truncated_prob(
        self,
        mu: FloatArray,
        sigma: FloatArray,
    ) -> FloatArray:
        """``P(field > threshold)`` for a Gaussian field with mean ``mu``, std ``sigma``."""
        safe = np.maximum(sigma, 1e-12)
        p = ndtr((mu - self.threshold) / safe)
        # Where there is no prior/posterior spread, the indicator is deterministic.
        deterministic = sigma <= 1e-12
        p = np.where(deterministic, (mu > self.threshold).astype(np.float64), p)
        return cast(FloatArray, p)

    def compute(self, sample_cells: Any) -> DecisionAccuracyResult:
        """Map decision-accuracy gain for an exact cell acquisition.

        Args:
            sample_cells: Flat cell indices into the (row-major flattened) grid
                that the acquisition observes (e.g. the cells a borehole passes
                through). Use :meth:`flat_indices` to build these from grid
                coordinates.

        Returns:
            :class:`DecisionAccuracyResult` on the model grid.
        """
        t0 = time.perf_counter()
        if isinstance(sample_cells, torch.Tensor):
            sample_cells = sample_cells.detach().cpu().numpy()
        try:
            raw_cells = np.asarray(sample_cells)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GeoBrainError(
                "sample_cells must contain integer flat indices",
                object_name="SpatialDecisionAccuracy",
                field="sample_cells",
                expected="non-empty integer indices",
                actual=sample_cells,
            ) from exc
        if (
            np.issubdtype(raw_cells.dtype, np.bool_)
            or not np.issubdtype(raw_cells.dtype, np.integer)
        ):
            raise GeoBrainError(
                "sample_cells must contain integer flat indices",
                object_name="SpatialDecisionAccuracy",
                field="sample_cells",
                expected="non-empty integer indices",
                actual=sample_cells,
            )
        cells = np.asarray(raw_cells, dtype=np.int64).ravel()
        if cells.size == 0:
            raise GeoBrainError(
                "sample_cells is empty",
                object_name="SpatialDecisionAccuracy",
                field="sample_cells", expected=">= 1 cell", actual=0,
            )
        n_cells = self._W.shape[0]
        if cells.min() < 0 or cells.max() >= n_cells:
            raise GeoBrainError(
                "sample_cells out of range",
                object_name="SpatialDecisionAccuracy",
                field="sample_cells", expected=f"[0, {n_cells})",
                actual=(int(cells.min()), int(cells.max())),
            )

        lam = self._lam
        a = self._W[cells, :]                          # (n_data, k)  = G W
        ala = a @ (lam[:, None] * a.T)                 # (n_data, n_data) = A Λ Aᵀ
        ala_inv = np.linalg.pinv(ala, rcond=1e-10)
        k_gain = (lam[:, None] * a.T) @ ala_inv         # (k, n_data) = Λ Aᵀ (AΛAᵀ)⁻¹
        # Posterior PC covariance (same for every realization): Λ − K A Λ.
        cy = np.diag(lam) - (k_gain @ a) * lam[None, :]
        sigma_post = np.sqrt(np.maximum(((self._W @ cy) * self._W).sum(1), 0.0))

        # Per-realization posterior mean fields: μ^l = v + W·(K A y^l).
        ey = self._scores @ (k_gain @ a).T              # (L, k)  = E[y | d^l]
        mu = self._mean[None, :] + ey @ self._W.T       # (L, n_cells)
        posterior_probability = self._truncated_prob(
            mu, np.broadcast_to(sigma_post, mu.shape)
        )                                               # (L, n_cells)
        prior_states = np.stack(
            (1.0 - self._prior_prob, self._prior_prob),
            axis=-1,
        )
        posterior_states = np.stack(
            (1.0 - posterior_probability, posterior_probability),
            axis=-1,
        )
        raw_gain = expected_utility_gain(
            prior_states,
            posterior_states,
            self.utilities,
        )
        prior_accuracy = np.einsum(
            "cs,as->ca",
            prior_states,
            self.utilities,
        ).max(axis=-1)
        posterior_optimal = np.einsum(
            "ocs,as->oca",
            posterior_states,
            self.utilities,
        ).max(axis=-1)
        expected_posterior_accuracy = posterior_optimal.mean(axis=0)
        statewise_perfect_accuracy = self.utilities.max(axis=0)
        perfect_information_accuracy = np.einsum(
            "cs,s->c",
            prior_states,
            statewise_perfect_accuracy,
        )
        ceiling = perfect_information_accuracy - prior_accuracy
        if self.normalize:
            gain_flat = np.divide(
                raw_gain,
                ceiling,
                out=np.zeros_like(raw_gain, dtype=np.float64),
                where=ceiling > 0.0,
            )
            units: Literal[
                "normalized_accuracy",
                "accuracy_probability",
            ] = "normalized_accuracy"
        else:
            gain_flat = raw_gain
            units = "accuracy_probability"

        standard_error_flat: FloatArray | None = None
        uncertainty: dict[str, Any]
        if self.n_bootstrap > 0:
            if self.n_bootstrap == 1:
                standard_error_flat = np.zeros_like(gain_flat)
            else:
                bootstrap_rng = np.random.default_rng(self.bootstrap_seed)
                indices = bootstrap_rng.integers(
                    0,
                    posterior_optimal.shape[0],
                    size=(self.n_bootstrap, posterior_optimal.shape[0]),
                )
                replicate_gain = (
                    posterior_optimal[indices].mean(axis=1)
                    - prior_accuracy[None, :]
                )
                if self.normalize:
                    replicate_gain = np.divide(
                        replicate_gain,
                        ceiling[None, :],
                        out=np.zeros_like(replicate_gain),
                        where=ceiling[None, :] > 0.0,
                    )
                standard_error_flat = replicate_gain.std(axis=0, ddof=1)
            uncertainty = {
                "method": "bootstrap_of_acquisition_outcomes",
                "replicates": self.n_bootstrap,
                "seed": self.bootstrap_seed,
            }
        else:
            uncertainty = {
                "method": "not_requested",
                "replicates": 0,
                "seed": self.bootstrap_seed,
            }

        gain_map = gain_flat.reshape(self.grid_shape)
        prior_accuracy_map = prior_accuracy.reshape(self.grid_shape)
        expected_posterior_accuracy_map = (
            expected_posterior_accuracy.reshape(self.grid_shape)
        )
        standard_error_map = (
            None
            if standard_error_flat is None
            else standard_error_flat.reshape(self.grid_shape)
        )
        elapsed = time.perf_counter() - t0

        return DecisionAccuracyResult(
            gain_map=torch.as_tensor(gain_map, device=self.device),
            prior_accuracy=torch.as_tensor(
                prior_accuracy_map,
                device=self.device,
            ),
            expected_posterior_accuracy=torch.as_tensor(
                expected_posterior_accuracy_map,
                device=self.device,
            ),
            mean_gain=float(gain_map.mean()),
            n_realizations=self._scores.shape[0],
            estimator="pca_linear_gaussian_bayes_action",
            units=units,
            gain_standard_error=(
                None
                if standard_error_map is None
                else torch.as_tensor(standard_error_map, device=self.device)
            ),
            metadata={
                "threshold": self.threshold,
                "utilities": self.utilities,
                "n_components": self.n_components,
                "n_sample_cells": int(cells.size),
                "elapsed_seconds": elapsed,
                "uncertainty": uncertainty,
            },
        )

    def flat_indices(self, coords: Sequence[Sequence[int]]) -> IntArray:
        """Map grid-coordinate tuples to row-major flat cell indices."""
        arr = np.asarray(coords, dtype=np.int64)
        return cast(
            IntArray,
            np.asarray(
                np.ravel_multi_index(tuple(arr.T), self.grid_shape),
                dtype=np.int64,
            ),
        )

    def conditional_realizations(
        self,
        sample_cells: Any,
        observed: Any,
        n: int,
        *,
        truncate: bool = False,
        seed: int | None = None,
    ) -> FloatArray:
        """Draw ``n`` realizations conditioned on ``observed`` at ``sample_cells``.

        Stochastic counterpart of :meth:`compute`: samples the exact linear-Gauss
        posterior of the PC coefficients, ``y ~ N(E[y|d], C[y|d])``, and
        reconstructs the field (optionally truncated to the binary indicator).
        Averaging the truncated draws over many ``n`` recovers, per cell, the
        same posterior probability :meth:`compute` uses analytically and can
        independently validate its conditional model.
        """
        cells = np.asarray(sample_cells, dtype=np.int64).ravel()
        obs = np.asarray(observed, dtype=np.float64).ravel()
        lam = self._lam
        a = self._W[cells, :]                              # (n_data, k) = G W
        ala = a @ (lam[:, None] * a.T)
        ala_inv = np.linalg.pinv(ala, rcond=1e-10)
        k_gain = (lam[:, None] * a.T) @ ala_inv             # Λ Aᵀ (AΛAᵀ)⁻¹
        ey = k_gain @ (obs - self._mean[cells])             # E[y | d]   (k,)
        cy = np.diag(lam) - (k_gain @ a) * lam[None, :]     # C[y | d]   (k, k)
        w, vecs = np.linalg.eigh(0.5 * (cy + cy.T))
        sqrt_cov = vecs @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
        rng = np.random.default_rng(seed)
        y = ey[None, :] + rng.standard_normal((int(n), lam.shape[0])) @ sqrt_cov.T
        fields = self._mean[None, :] + y @ self._W.T
        if truncate:
            fields = (fields > self.threshold).astype(np.float64)
        return cast(
            FloatArray,
            fields.reshape((int(n), *self.grid_shape)),
        )


__all__ = ["SpatialDecisionAccuracy"]
