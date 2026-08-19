"""KDE Monte-Carlo estimation of mutual information in nats.

For target samples ``m`` paired with prior-predictive acquisition samples
``d``, the estimator reports

``I(m; d) = E_d[KL(p(m | d) || p(m))]``.

Both densities use product-Gaussian KDEs. The conditional density uses
Nadaraya-Watson kernel weights, and posterior samples are drawn from that
conditional Gaussian mixture within configured property bounds. This
information currency is distinct from the Bayes-action accuracy probability
reported by :mod:`geobrain.decision.accuracy` and
:mod:`geobrain.decision.spatial_eoi`.

The scientific Monte-Carlo path preserves the pre-rename draw order. A
separate RNG performs a deterministic bootstrap of the per-outcome mean to
report mandatory standard-error uncertainty without perturbing those draws.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from geobrain.core.validation import (
    validate_non_negative_int,
    validate_positive_int,
)
from geobrain.core.errors import GeoBrainError
from geobrain.decision._metadata import freeze_metadata, thaw_metadata
from geobrain.decision.protocols import CancellationCheck
from geobrain.decision.status import DecisionRunStatus

logger = logging.getLogger(__name__)


FloatArray: TypeAlias = NDArray[np.float64]


class _LogSumExp(Protocol):
    def __call__(
        self,
        values: ArrayLike,
        *,
        axis: int | None = None,
    ) -> Any: ...


logsumexp = cast(
    _LogSumExp,
    getattr(import_module("scipy.special"), "logsumexp"),
)


# =============================================================================
# Result container
# =============================================================================

@dataclass(frozen=True)
class MutualInformationResult:
    """Owned, validated mutual-information estimate in nats.

    Attributes:
        information_nats: estimated mutual information in nats.
        per_outcome_nats: per-outcome contributions.
        standard_error_nats: Monte-Carlo standard error.
        n_samples / n_posterior: sample budgets used.
        seed: RNG seed of the estimate.
        completed_outcomes: outcomes actually evaluated.
        status: completed / truncated marker.
        elapsed_seconds: wall time.
        metadata: estimator extras.
    """

    information_nats: float
    per_outcome_nats: torch.Tensor
    standard_error_nats: float
    n_samples: int
    n_posterior: int
    seed: int | None
    completed_outcomes: int
    status: DecisionRunStatus
    elapsed_seconds: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        owner = "MutualInformationResult"
        if (
            isinstance(self.information_nats, bool)
            or not isinstance(self.information_nats, (int, float))
            or not math.isfinite(float(self.information_nats))
            or float(self.information_nats) < 0.0
        ):
            raise GeoBrainError(
                "information_nats must be finite and non-negative",
                object_name=owner,
                field="information_nats",
                expected="finite value >= 0",
                actual=self.information_nats,
            )
        if not isinstance(self.per_outcome_nats, torch.Tensor):
            raise GeoBrainError(
                "per_outcome_nats must be a tensor",
                object_name=owner,
                field="per_outcome_nats",
                expected="one-dimensional finite tensor",
                actual=type(self.per_outcome_nats),
            )
        per_outcome = self.per_outcome_nats.detach().clone()
        if (
            per_outcome.ndim != 1
            or not bool(torch.isfinite(per_outcome).all())
            or bool((per_outcome < 0).any())
        ):
            raise GeoBrainError(
                "per_outcome_nats must be a finite non-negative vector",
                object_name=owner,
                field="per_outcome_nats",
                expected="[completed_outcomes] finite values >= 0",
                actual=tuple(per_outcome.shape),
            )
        if (
            isinstance(self.standard_error_nats, bool)
            or not isinstance(self.standard_error_nats, (int, float))
            or not math.isfinite(float(self.standard_error_nats))
            or float(self.standard_error_nats) < 0.0
        ):
            raise GeoBrainError(
                "standard_error_nats must be finite and non-negative",
                object_name=owner,
                field="standard_error_nats",
                expected="finite value >= 0",
                actual=self.standard_error_nats,
            )
        n_samples = validate_positive_int(
            self.n_samples,
            owner=owner,
            field="n_samples",
        )
        n_posterior = validate_positive_int(
            self.n_posterior,
            owner=owner,
            field="n_posterior",
        )
        completed = validate_non_negative_int(
            self.completed_outcomes,
            owner=owner,
            field="completed_outcomes",
        )
        if self.seed is None:
            seed = None
        else:
            seed = validate_non_negative_int(
                self.seed,
                owner=owner,
                field="seed",
            )
        if not isinstance(self.status, DecisionRunStatus):
            raise GeoBrainError(
                "status must be a DecisionRunStatus",
                object_name=owner,
                field="status",
                expected=DecisionRunStatus,
                actual=self.status,
            )
        if completed > n_samples:
            raise GeoBrainError(
                "completed_outcomes cannot exceed n_samples",
                object_name=owner,
                field="completed_outcomes",
                expected=f"0 <= completed_outcomes <= {n_samples}",
                actual=completed,
            )
        if completed != per_outcome.numel():
            raise GeoBrainError(
                "completed_outcomes must match per_outcome_nats",
                object_name=owner,
                field="completed_outcomes",
                expected=per_outcome.numel(),
                actual=completed,
            )
        if (
            self.status is DecisionRunStatus.COMPLETED
            and completed != n_samples
        ):
            raise GeoBrainError(
                "completed results must include every sample outcome",
                object_name=owner,
                field="completed_outcomes",
                expected=n_samples,
                actual=completed,
            )
        if (
            self.status is DecisionRunStatus.CANCELLED
            and completed == n_samples
        ):
            raise GeoBrainError(
                "a fully completed result must use COMPLETED status",
                object_name=owner,
                field="status",
                expected=DecisionRunStatus.COMPLETED,
                actual=DecisionRunStatus.CANCELLED,
            )
        if completed == 0:
            if float(self.information_nats) != 0.0:
                raise GeoBrainError(
                    "information_nats must be zero without completed outcomes",
                    object_name=owner,
                    field="information_nats",
                    expected=0.0,
                    actual=self.information_nats,
                )
            if float(self.standard_error_nats) != 0.0:
                raise GeoBrainError(
                    "standard_error_nats must be zero without completed outcomes",
                    object_name=owner,
                    field="standard_error_nats",
                    expected=0.0,
                    actual=self.standard_error_nats,
                )
        else:
            per_outcome_mean = float(per_outcome.mean())
            if not math.isclose(
                float(self.information_nats),
                per_outcome_mean,
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                raise GeoBrainError(
                    "information_nats must equal the per-outcome mean",
                    object_name=owner,
                    field="information_nats",
                    expected=per_outcome_mean,
                    actual=self.information_nats,
                )
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(float(self.elapsed_seconds))
            or float(self.elapsed_seconds) < 0.0
        ):
            raise GeoBrainError(
                "elapsed_seconds must be finite and non-negative",
                object_name=owner,
                field="elapsed_seconds",
                expected="finite value >= 0",
                actual=self.elapsed_seconds,
            )
        object.__setattr__(self, "information_nats", float(self.information_nats))
        object.__setattr__(self, "per_outcome_nats", per_outcome)
        object.__setattr__(
            self,
            "standard_error_nats",
            float(self.standard_error_nats),
        )
        object.__setattr__(self, "n_samples", n_samples)
        object.__setattr__(self, "n_posterior", n_posterior)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "completed_outcomes", completed)
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        object.__setattr__(
            self,
            "metadata",
            freeze_metadata(self.metadata, object_name=owner),
        )

    def summary(self) -> str:
        """Return a human-readable summary string."""
        per_outcome = self.per_outcome_nats
        if per_outcome.numel() == 0:
            per_outcome_summary = "no completed outcomes"
        else:
            per_outcome_summary = (
                f"min={float(per_outcome.min()):.4f}  "
                f"mean={float(per_outcome.mean()):.4f}  "
                f"max={float(per_outcome.max()):.4f} nats"
            )
        lines = [
            "=== Mutual Information Result ===",
            f"Mutual information       : {self.information_nats:.4f} nats",
            f"Standard error           : {self.standard_error_nats:.4f} nats",
            f"Per-outcome information  : {per_outcome_summary}",
            f"Ensemble size            : {self.n_samples}",
            f"Posterior samples / draw : {self.n_posterior}",
            f"Completed outcomes       : {self.completed_outcomes}",
            f"Status                   : {self.status.value}",
            f"Elapsed time             : {self.elapsed_seconds:.2f} s",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached plain representation with explicit nats keys."""
        return {
            "information_nats": self.information_nats,
            "per_outcome_nats": (
                self.per_outcome_nats.detach().cpu().numpy().copy()
            ),
            "standard_error_nats": self.standard_error_nats,
            "n_samples": self.n_samples,
            "n_posterior": self.n_posterior,
            "seed": self.seed,
            "completed_outcomes": self.completed_outcomes,
            "status": self.status.value,
            "elapsed_seconds": self.elapsed_seconds,
            "metadata": thaw_metadata(self.metadata),
        }


# =============================================================================
# Gaussian KDE primitives (self-contained: product Gaussian kernels)
# =============================================================================

def _as_2d_numpy(x: Any, name: str) -> FloatArray:
    """Coerce array-like / tensor to a 2-D ``(n_samples, dim)`` float64 array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise GeoBrainError(
            f"{name} must be a 2-D (n_samples, dim) array",
            object_name="MutualInformationEstimator", field=name,
            expected="(n_samples, dim)", actual=tuple(arr.shape),
        )
    return arr


def _bandwidth(samples: FloatArray, rule: str) -> FloatArray:
    """Per-dimension rule-of-thumb Gaussian bandwidth (Scott / Silverman)."""
    n, dim = samples.shape
    std = samples.std(axis=0, ddof=1)
    if rule == "scott":
        factor = n ** (-1.0 / (dim + 4))
    elif rule == "silverman":
        factor = (n * (dim + 2) / 4.0) ** (-1.0 / (dim + 4))
    else:
        raise GeoBrainError(
            "bandwidth rule must be 'scott' or 'silverman'",
            object_name="MutualInformationEstimator", field="bandwidth",
            expected="'scott' or 'silverman'", actual=rule,
        )
    return cast(FloatArray, np.maximum(std * factor, 1e-12))


def _kde_logpdf(
    query: FloatArray,
    samples: FloatArray,
    h: FloatArray,
) -> FloatArray:
    """log of a product-Gaussian KDE ``p(x) = mean_i Π_j N(x_j; X_ij, h_j)``."""
    z = (query[:, None, :] - samples[None, :, :]) / h[None, None, :]   # (Q, N, D)
    log_norm = np.log(h).sum() + 0.5 * samples.shape[1] * np.log(2.0 * np.pi)
    log_kernel = -0.5 * (z ** 2).sum(axis=-1) - log_norm                # (Q, N)
    return cast(
        FloatArray,
        logsumexp(log_kernel, axis=1) - np.log(samples.shape[0]),
    )


def _conditional_weights(
    d_query: FloatArray,
    d_samples: FloatArray,
    h_d: FloatArray,
) -> FloatArray:
    """Nadaraya–Watson weights ``w_i ∝ K_{h_d}(d_query − d_i)``, normalised to sum 1."""
    z = (d_query[None, :] - d_samples) / h_d[None, :]                    # (N, Dd)
    log_w = -0.5 * (z ** 2).sum(axis=1)                                  # (N,), norm cancels
    log_w = log_w - logsumexp(log_w)
    return cast(FloatArray, np.exp(log_w))


def _conditional_logpdf_per_query(
    query: FloatArray,
    m_samples: FloatArray,
    h_m: FloatArray,
    weights: FloatArray,
) -> FloatArray:
    """Evaluate rows whose conditional mixture weights differ."""
    z = (
        query[:, None, :] - m_samples[None, :, :]
    ) / h_m[None, None, :]
    log_norm = (
        np.log(h_m).sum()
        + 0.5 * m_samples.shape[1] * np.log(2.0 * np.pi)
    )
    log_kernel = -0.5 * (z**2).sum(axis=-1) - log_norm
    return cast(
        FloatArray,
        logsumexp(log_kernel + np.log(weights), axis=1),
    )


def _sample_conditional(
    m_samples: FloatArray,
    h_m: FloatArray,
    weights: FloatArray,
    n_post: int,
    low: FloatArray,
    up: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw ``n_post`` samples from ``Σ_i w_i N(m_i, h_m)`` truncated to ``[low, up]``.

    Exact Gaussian-mixture sampling (pick a component by its weight, jitter by
    the kernel bandwidth) with rejection of out-of-bounds draws, the same
    truncated conditional the reference targets via uniform rejection, sampled
    directly from its mixture form.
    """
    n, dim = m_samples.shape
    out = np.empty((n_post, dim), dtype=np.float64)
    filled = 0
    max_rounds = 1000
    for _ in range(max_rounds):
        batch = n_post - filled
        comps = rng.choice(n, size=batch, p=weights)
        cand = m_samples[comps] + h_m[None, :] * rng.standard_normal((batch, dim))
        inside = np.all((cand >= low[None, :]) & (cand <= up[None, :]), axis=1)
        kept = cand[inside]
        take = min(kept.shape[0], n_post - filled)
        out[filled:filled + take] = kept[:take]
        filled += take
        if filled >= n_post:
            return cast(FloatArray, out)
    raise GeoBrainError(
        "conditional sampling could not fill the posterior within the bounds; "
        "widen `bounds` or check for degenerate (zero-variance) samples",
        object_name="MutualInformationEstimator", field="bounds",
        expected=f"{n_post} in-bounds samples", actual=filled,
    )


# =============================================================================
# Mutual-information estimator
# =============================================================================

class MutualInformationEstimator:
    """Estimate mutual information for a candidate acquisition.

    Scores a candidate data acquisition by its expected information gain about
    a target quantity ``m`` (held fixed across calls so several candidate
    acquisitions can be compared against the same prior).

    Args:
        m_samples: Prior ensemble of the target quantity, shape
            ``[n_samples, m_dim]`` (1-D input is treated as ``[n_samples, 1]``).
        n_posterior: Posterior Monte-Carlo samples drawn per realization.
        bandwidth: KDE bandwidth rule, ``"scott"`` (default) or ``"silverman"``.
        bounds: ``(low, up)`` per-target-dimension bounds for the truncated
            posterior sampling. Defaults to the min / max of ``m_samples``.
        seed: Seed for the scientific posterior-sampling RNG.
        bootstrap_replicates: Positive number of outcome-mean bootstrap
            replicates used for mandatory standard-error estimation.
        bootstrap_seed: Seed for the bootstrap RNG. This stream is independent
            from ``seed`` and cannot alter scientific posterior draws.
        batch_size: Maximum query and outcome count evaluated per batch.
        device: Torch device for the result tensors (default CPU).

    Example:
        >>> estimator = MutualInformationEstimator(
        ...     m_prior, n_posterior=200, seed=0
        ... )
        >>> r1 = estimator.compute(d1_predicted)
        >>> r2 = estimator.compute(d2_predicted)
        >>> rseq = estimator.compute_sequential(d1_predicted, d2_predicted)
        >>> print(r1.summary())
    """

    def __init__(
        self,
        m_samples: Any,
        n_posterior: int = 200,
        bandwidth: str = "scott",
        bounds: tuple[Any, Any] | None = None,
        seed: int | None = None,
        bootstrap_replicates: int = 200,
        bootstrap_seed: int = 0,
        batch_size: int = 128,
        device: torch.device | None = None,
    ) -> None:
        self.m_samples = _as_2d_numpy(m_samples, "m_samples").copy()
        if self.m_samples.shape[0] < 2:
            raise GeoBrainError(
                "mutual information needs at least 2 prior samples",
                object_name="MutualInformationEstimator", field="m_samples",
                expected=">= 2 samples", actual=self.m_samples.shape[0],
            )
        self.n_posterior = validate_positive_int(
            n_posterior,
            owner="MutualInformationEstimator",
            field="n_posterior",
        )
        self.bootstrap_replicates = validate_positive_int(
            bootstrap_replicates,
            owner="MutualInformationEstimator",
            field="bootstrap_replicates",
        )
        if seed is not None:
            seed = validate_non_negative_int(
                seed,
                owner="MutualInformationEstimator",
                field="seed",
            )
        bootstrap_seed = validate_non_negative_int(
            bootstrap_seed,
            owner="MutualInformationEstimator",
            field="bootstrap_seed",
        )
        self._batch_size: int = validate_positive_int(
            batch_size,
            owner="MutualInformationEstimator",
            field="batch_size",
        )
        self.bandwidth = bandwidth
        self.device = device
        self.seed = seed
        self.bootstrap_seed = bootstrap_seed
        self._h_m = _bandwidth(self.m_samples, bandwidth)
        m_dim = self.m_samples.shape[1]
        if bounds is None:
            self._low = self.m_samples.min(axis=0)
            self._up = self.m_samples.max(axis=0)
        else:
            self._low = np.asarray(bounds[0], dtype=np.float64).reshape(-1)
            self._up = np.asarray(bounds[1], dtype=np.float64).reshape(-1)
            # Fail at construction on a malformed box (wrong dimensionality or
            # a non-positive-width interval) rather than deep inside the
            # rejection sampler where the error is opaque.
            if self._low.shape[0] != m_dim or self._up.shape[0] != m_dim:
                raise GeoBrainError(
                    "bounds dimensionality must match the target dimension",
                    object_name="MutualInformationEstimator", field="bounds",
                    expected=f"(low, up) each of length {m_dim}",
                    actual=(int(self._low.shape[0]), int(self._up.shape[0])),
                )
            if not bool(np.all(self._low < self._up)):
                raise GeoBrainError(
                    "bounds require low < up on every dimension",
                    object_name="MutualInformationEstimator", field="bounds",
                    expected="low < up per dimension",
                    actual=(self._low.tolist(), self._up.tolist()),
                )

    @property
    def batch_size(self) -> int:
        """Maximum number of query points evaluated in one bounded batch."""
        return self._batch_size

    def _prior_logpdf(self, query: FloatArray) -> FloatArray:
        output = np.empty(query.shape[0], dtype=np.float64)
        for start in range(0, query.shape[0], self.batch_size):
            stop = min(start + self.batch_size, query.shape[0])
            output[start:stop] = _kde_logpdf(
                query[start:stop],
                self.m_samples,
                self._h_m,
            )
        return output

    def _conditional_logpdf_batch_weights(
        self,
        query: FloatArray,
        weights: FloatArray,
        *,
        rows_per_condition: int,
    ) -> FloatArray:
        output = np.empty(query.shape[0], dtype=np.float64)
        for start in range(0, query.shape[0], self.batch_size):
            stop = min(start + self.batch_size, query.shape[0])
            condition_indices: NDArray[np.int64] = (
                np.arange(start, stop, dtype=np.int64)
                // rows_per_condition
            )
            output[start:stop] = _conditional_logpdf_per_query(
                query[start:stop],
                self.m_samples,
                self._h_m,
                weights[condition_indices],
            )
        return output

    def _bootstrap_standard_error(self, values: FloatArray) -> float:
        """Return the deterministic bootstrap SE of the outcome mean."""
        if values.size == 1 or self.bootstrap_replicates == 1:
            return 0.0
        rng = np.random.default_rng(self.bootstrap_seed)
        indices = rng.integers(
            0,
            values.size,
            size=(self.bootstrap_replicates, values.size),
        )
        replicate_means = values[indices].mean(axis=1)
        return float(replicate_means.std(ddof=1))

    def _metadata(self, *, mode: str, **extra: Any) -> dict[str, Any]:
        return {
            "estimator": "gaussian_kde_monte_carlo",
            "units": "nats",
            "bandwidth": self.bandwidth,
            "batch_size": self.batch_size,
            "mode": mode,
            "uncertainty": {
                "method": "bootstrap_of_outcome_mean",
                "replicates": self.bootstrap_replicates,
                "seed": self.bootstrap_seed,
            },
            **extra,
        }

    # ------------------------------------------------------------------
    # Internal: one KL(p(m|cond) || p(m)) for a single conditioning vector
    # ------------------------------------------------------------------

    def _kl_given(
        self,
        cond_query: FloatArray,
        cond_samples: FloatArray,
        h_cond: FloatArray,
        rng: np.random.Generator,
    ) -> float:
        """MC estimate of ``KL(p(m | cond=cond_query) || p(m))``, clamped >= 0."""
        return float(
            self._kl_batch_given(
                cond_query[None, :],
                cond_samples,
                h_cond,
                rng,
            )[0],
        )

    def _kl_batch_given(
        self,
        cond_queries: FloatArray,
        cond_samples: FloatArray,
        h_cond: FloatArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Vectorize density scoring while preserving scalar RNG draw order."""
        n_queries = cond_queries.shape[0]
        weights = np.empty(
            (n_queries, self.m_samples.shape[0]),
            dtype=np.float64,
        )
        posterior = np.empty(
            (n_queries, self.n_posterior, self.m_samples.shape[1]),
            dtype=np.float64,
        )
        for index, query in enumerate(cond_queries):
            weights[index] = _conditional_weights(
                query,
                cond_samples,
                h_cond,
            )
            posterior[index] = _sample_conditional(
                self.m_samples,
                self._h_m,
                weights[index],
                self.n_posterior,
                self._low,
                self._up,
                rng,
            )
        flattened = posterior.reshape(-1, self.m_samples.shape[1])
        log_post = self._conditional_logpdf_batch_weights(
            flattened,
            weights,
            rows_per_condition=self.n_posterior,
        ).reshape(n_queries, self.n_posterior)
        log_prior = self._prior_logpdf(flattened).reshape(
            n_queries,
            self.n_posterior,
        )
        return cast(
            FloatArray,
            np.maximum(0.0, (log_post - log_prior).mean(axis=1)),
        )

    @staticmethod
    def _validate_cancellation(
        cancellation: CancellationCheck | None,
    ) -> None:
        if cancellation is not None and not callable(cancellation):
            raise GeoBrainError(
                "cancellation must be callable",
                object_name="MutualInformationEstimator",
                field="cancellation",
                expected="callable () -> bool or None",
                actual=type(cancellation).__name__,
            )

    @staticmethod
    def _cancelled(cancellation: CancellationCheck | None) -> bool:
        if cancellation is None:
            return False
        result = cancellation()
        if not isinstance(result, bool):
            raise GeoBrainError(
                "cancellation callback must return bool",
                object_name="MutualInformationEstimator",
                field="cancellation return",
                expected=bool,
                actual=type(result).__name__,
            )
        return result

    def _result(
        self,
        values: FloatArray,
        *,
        n_samples: int,
        elapsed_seconds: float,
        mode: str,
        **metadata: Any,
    ) -> MutualInformationResult:
        completed = int(values.size)
        status = (
            DecisionRunStatus.COMPLETED
            if completed == n_samples
            else DecisionRunStatus.CANCELLED
        )
        information = float(values.mean()) if completed else 0.0
        standard_error = (
            self._bootstrap_standard_error(values) if completed else 0.0
        )
        return MutualInformationResult(
            information_nats=information,
            per_outcome_nats=torch.as_tensor(values, device=self.device),
            standard_error_nats=standard_error,
            n_samples=n_samples,
            n_posterior=self.n_posterior,
            seed=self.seed,
            completed_outcomes=completed,
            status=status,
            elapsed_seconds=elapsed_seconds,
            metadata=self._metadata(mode=mode, **metadata),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        d_samples: Any,
        verbose: bool = False,
        *,
        cancellation: CancellationCheck | None = None,
    ) -> MutualInformationResult:
        """
        Compute mutual information for a candidate acquisition.

        Args:
            d_samples: Prior-predictive ensemble of the data the candidate
                acquisition would yield, shape ``[n_samples, d_dim]``, paired
                row-for-row with the ``m_samples`` passed to the constructor.
            verbose: Log progress.
            cancellation: Optional cooperative check evaluated before each
                outcome batch. A true result returns the completed prefix.

        Returns:
            :class:`MutualInformationResult`.
        """
        self._validate_cancellation(cancellation)
        t_start = time.perf_counter()
        d = _as_2d_numpy(d_samples, "d_samples")
        if d.shape[0] != self.m_samples.shape[0]:
            raise GeoBrainError(
                "d_samples must be paired row-for-row with m_samples",
                object_name="MutualInformationEstimator", field="d_samples",
                expected=f"{self.m_samples.shape[0]} rows", actual=d.shape[0],
            )
        if verbose:
            logger.info(
                "Computing mutual information: "
                "%d outcomes, %d posterior samples each",
                d.shape[0],
                self.n_posterior,
            )
        rng = np.random.default_rng(self.seed)
        h_d = _bandwidth(d, self.bandwidth)
        batches: list[FloatArray] = []
        for start in range(0, d.shape[0], self.batch_size):
            if self._cancelled(cancellation):
                break
            stop = min(start + self.batch_size, d.shape[0])
            batches.append(self._kl_batch_given(d[start:stop], d, h_d, rng))
        per_outcome = (
            np.concatenate(batches)
            if batches
            else np.empty(0, dtype=np.float64)
        )
        elapsed = time.perf_counter() - t_start
        if verbose:
            logger.info(
                "Mutual information completed %d/%d outcomes",
                per_outcome.size,
                d.shape[0],
            )
        return self._result(
            per_outcome,
            n_samples=d.shape[0],
            elapsed_seconds=elapsed,
            mode="single",
        )

    def compute_sequential(
        self,
        d1_samples: Any,
        d2_samples: Any,
        n_d2: int | None = None,
        verbose: bool = False,
        *,
        cancellation: CancellationCheck | None = None,
    ) -> MutualInformationResult:
        """
        Sequential mutual information for acquiring ``d1`` and then ``d2``.

        ``I(d1 -> d2) = E_{d1}[ max( KL(p(m|d1)||p(m)),
        E_{d2|d1}[ KL(p(m|d1,d2)||p(m)) ] ) ]``.

        Cost note: this is the module's most expensive path, for each of the
        ``n_samples`` d1 realizations it draws ``n_d2`` samples of ``d2|d1`` and
        runs a full ``n_posterior``-sample inner KL per draw, i.e.
        ``O(n_samples · n_d2 · n_posterior)`` conditional evaluations. ``n_d2``
        (the inner ``E_{d2|d1}`` Monte-Carlo count) is a SEPARATE knob from
        ``n_posterior`` (the per-KL posterior count) so the two can be traded
        independently; it defaults to ``n_posterior`` for backward-compatible
        behaviour.

        Args:
            d1_samples / d2_samples: Prior-predictive ensembles for the first
                and second acquisitions, each ``[n_samples, *]`` and paired
                row-for-row with the constructor's ``m_samples``.
            n_d2: Monte-Carlo count for the inner ``E_{d2|d1}`` expectation
                (``>= 1``). ``None`` (default) falls back to ``n_posterior``.
            verbose: Log progress.
            cancellation: Optional cooperative check evaluated before each
                first-stage outcome batch. A true result returns the completed
                prefix without interrupting an in-progress batch.

        Returns:
            :class:`MutualInformationResult` whose ``per_outcome_nats`` holds
            the per-``d1`` ``max(...)`` score.
        """
        self._validate_cancellation(cancellation)
        if n_d2 is None:
            n_d2 = self.n_posterior
        else:
            n_d2 = validate_positive_int(
                n_d2,
                owner="MutualInformationEstimator",
                field="n_d2",
            )
        t_start = time.perf_counter()
        d1 = _as_2d_numpy(d1_samples, "d1_samples")
        d2 = _as_2d_numpy(d2_samples, "d2_samples")
        n = self.m_samples.shape[0]
        if d1.shape[0] != n or d2.shape[0] != n:
            raise GeoBrainError(
                "d1_samples and d2_samples must be paired row-for-row with m_samples",
                object_name="MutualInformationEstimator", field="d1_samples/d2_samples",
                expected=f"{n} rows", actual=(d1.shape[0], d2.shape[0]),
            )
        if verbose:
            logger.info(
                "Computing sequential mutual information: "
                "%d first-stage outcomes, %d posterior samples each",
                n,
                self.n_posterior,
            )
        rng = np.random.default_rng(self.seed)
        h_d1 = _bandwidth(d1, self.bandwidth)
        h_d2 = _bandwidth(d2, self.bandwidth)
        d12 = np.concatenate([d1, d2], axis=1)
        h_d12 = _bandwidth(d12, self.bandwidth)
        d2_low, d2_up = d2.min(axis=0), d2.max(axis=0)

        batches: list[FloatArray] = []
        for start in range(0, n, self.batch_size):
            if self._cancelled(cancellation):
                break
            stop = min(start + self.batch_size, n)
            batch = np.empty(stop - start, dtype=np.float64)
            for offset, i in enumerate(range(start, stop)):
                information_d1 = self._kl_given(d1[i], d1, h_d1, rng)
                w_d2 = _conditional_weights(d1[i], d1, h_d1)
                d2_post = _sample_conditional(
                    d2,
                    h_d2,
                    w_d2,
                    n_d2,
                    d2_low,
                    d2_up,
                    rng,
                )
                kl_d2_sum = 0.0
                for d2_start in range(0, n_d2, self.batch_size):
                    d2_stop = min(d2_start + self.batch_size, n_d2)
                    d2_batch = d2_post[d2_start:d2_stop]
                    d1_batch = np.repeat(
                        d1[i][None, :],
                        d2_stop - d2_start,
                        axis=0,
                    )
                    joint_queries = np.concatenate(
                        [d1_batch, d2_batch],
                        axis=1,
                    )
                    kl_d2_sum += float(
                        self._kl_batch_given(
                            joint_queries,
                            d12,
                            h_d12,
                            rng,
                        ).sum(),
                    )
                batch[offset] = max(information_d1, kl_d2_sum / n_d2)
            batches.append(batch)
        seq = (
            np.concatenate(batches)
            if batches
            else np.empty(0, dtype=np.float64)
        )
        elapsed = time.perf_counter() - t_start
        if verbose:
            logger.info(
                "Sequential mutual information completed %d/%d outcomes",
                seq.size,
                n,
            )
        return self._result(
            seq,
            n_samples=n,
            elapsed_seconds=elapsed,
            mode="sequential",
            n_d2=n_d2,
        )


__all__ = [
    "DecisionRunStatus",
    "MutualInformationEstimator",
    "MutualInformationResult",
]
