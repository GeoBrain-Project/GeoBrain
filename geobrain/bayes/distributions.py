# pyright: reportPrivateImportUsage=false
"""
Tensor-level probability distributions.

Two standalone distributions for use as sampler targets, flipout-layer
reparameterizations, or likelihood components:

- :class:`Gaussian`: multivariate Gaussian with cov / precision support;
  provides analytical score, sample, kl_divergence, entropy.
- :class:`GaussianMixture`: weighted mixture of Gaussians; useful for testing
  samplers on multimodal targets.

These operate on flat ``(batch, dim)`` tensors, the tensor-level analogue of the
**model-space** :class:`Prior` / :class:`GaussianPrior` in
:mod:`geobrain.inverse.prior`, which takes a structured
:class:`~geobrain.core.ModelState` instead. Three homes for probabilistic /
penalty objects:

- :mod:`geobrain.optim`: deterministic regularizers (no ``log_prob``).
- :mod:`geobrain.inverse.prior`: probabilistic priors over a :class:`ModelState`.
- :mod:`geobrain.bayes.distributions` (this module): probabilistic distributions
  over raw tensors.

Use :class:`~geobrain.inverse.GaussianPrior` for inverse-problem priors; use the
distributions below for vector-valued log-densities / sampler targets / NN flipout.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import torch
import torch.autograd as autograd
import torch.nn as nn

from geobrain.core.errors import GeoBrainError

__all__ = [
    "Distribution",
    "Gaussian",
    "GaussianMixture",
]


if TYPE_CHECKING:
    class _ModuleBase:
        def __init__(self, **kwargs: object) -> None: ...

        def register_buffer(self, name: str, tensor: torch.Tensor) -> None: ...

else:
    _ModuleBase = nn.Module


# ---------------------------------------------------------------------------
# Distribution ABC
# ---------------------------------------------------------------------------


class Distribution(ABC):
    """Abstract vector-valued probability distribution.

    Subclasses must implement :meth:`log_prob`. The default
    :meth:`score` uses autograd; subclasses may override with an
    analytical form.

    This ABC is intentionally separate from
    :class:`geobrain.inverse.Prior` (which acts on a structured
    :class:`~geobrain.core.ModelState`); it operates on flat
    ``(batch, dim)`` tensors. See the module docstring for the
    3-layer regularizer / prior / distribution map.
    """

    def __init__(self, dim: int | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._dim = dim

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log-density at ``x``. Returns shape ``(batch,)``."""

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """``grad_x log p(x)`` via autograd. Override for analytical form."""
        x = x.detach().requires_grad_(True)
        log_p = self.log_prob(x)
        score = autograd.grad(
            outputs=log_p.sum(), inputs=x, create_graph=False
        )[0]
        return score

    def sample(
        self, n_samples: int, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Direct sampling. Override in subclasses that support it.

        ``generator`` (optional) makes draws reproducible without touching the
        global RNG, matching the explicit-Generator convention the samplers use.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support direct sampling. "
            "Use a sampler (SVGD, HMC, ...) to generate samples."
        )

    @property
    def dim(self) -> int | None:
        return self._dim

    @dim.setter
    def dim(self, value: int) -> None:
        self._dim = value


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------


class Gaussian(Distribution, _ModuleBase):
    """Multivariate Gaussian.

    Args:
        mean: Mean ``(dim,)`` or ``(1, dim)``.
        cov: Covariance ``(dim, dim)``. Mutually exclusive with
            ``precision``.
        precision: Precision (inverse covariance). Mutually exclusive
            with ``cov``.
    """

    mean: torch.Tensor
    cov: torch.Tensor
    precision: torch.Tensor
    chol: torch.Tensor
    _log_norm: torch.Tensor

    def __init__(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor | None = None,
        precision: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(mean, torch.Tensor):
            raise GeoBrainError(
                f"Gaussian.mean must be a torch.Tensor, got "
                f"{type(mean).__name__}"
            )
        if mean.dim() == 1:
            mean = mean.unsqueeze(0)
        if mean.dim() != 2 or mean.shape[0] != 1:
            raise GeoBrainError(
                f"Gaussian.mean must be (D,) or (1, D); got shape "
                f"{tuple(mean.shape)}."
            )
        D = int(mean.shape[-1])
        super().__init__(dim=D)

        if cov is None and precision is None:
            raise GeoBrainError("Must provide either cov or precision")
        if cov is not None and precision is not None:
            raise GeoBrainError("Cannot provide both cov and precision")

        self._validate_cov_or_precision(
            cov if cov is not None else precision,
            D,
            label="cov" if cov is not None else "precision",
        )

        self.register_buffer("mean", mean)

        if cov is not None:
            self.register_buffer("cov", cov)
            self.register_buffer("precision", torch.linalg.inv(cov))
            self.register_buffer("chol", torch.linalg.cholesky(cov))
        else:
            self.register_buffer("precision", precision)
            self.register_buffer("cov", torch.linalg.inv(precision))
            self.register_buffer("chol", torch.linalg.cholesky(self.cov))

        d = D
        _, log_det = torch.linalg.slogdet(self.cov)
        self.register_buffer(
            "_log_norm", -0.5 * (d * math.log(2 * math.pi) + log_det)
        )

    @staticmethod
    def _validate_cov_or_precision(
        m: torch.Tensor, expected_dim: int, *, label: str
    ) -> None:
        """Square + symmetric + positive-definite + matching dim."""
        if not isinstance(m, torch.Tensor):
            raise GeoBrainError(
                f"Gaussian.{label} must be a torch.Tensor, got "
                f"{type(m).__name__}."
            )
        if m.dim() != 2 or m.shape[0] != m.shape[1]:
            raise GeoBrainError(
                f"Gaussian.{label} must be a square 2-D tensor; got "
                f"shape {tuple(m.shape)}."
            )
        if m.shape[0] != expected_dim:
            raise GeoBrainError(
                f"Gaussian.{label} dim {m.shape[0]} does not match "
                f"mean dim {expected_dim}."
            )
        m_dagger = (
            m.conj().transpose(-1, -2)
            if m.is_complex()
            else m.transpose(-1, -2)
        )
        sym_label = "Hermitian" if m.is_complex() else "symmetric"
        sym_err = (m - m_dagger).abs().max().item()
        sym_tol = max(
            1e-10, 1e-7 * float(m.abs().max().clamp(min=1.0).item())
        )
        if sym_err > sym_tol:
            raise GeoBrainError(
                f"Gaussian.{label} must be {sym_label}; max "
                f"|M - M{'†' if m.is_complex() else 'ᵀ'}| = "
                f"{sym_err:.3e} > tol {sym_tol:.3e}."
            )
        try:
            torch.linalg.cholesky(0.5 * (m + m_dagger))
        except torch._C._LinAlgError as err:  # noqa: SLF001
            raise GeoBrainError(
                f"Gaussian.{label} must be positive-definite "
                f"(Cholesky failed: {err})."
            ) from err

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        diff = x - self.mean
        mahal = torch.einsum("bi,ij,bj->b", diff, self.precision, diff)
        return self._log_norm - 0.5 * mahal

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Analytical score: ``grad log p(x) = -Sigma^{-1} (x - mu)``."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        diff = x - self.mean
        return -torch.matmul(diff, self.precision)

    def sample(
        self, n_samples: int, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Reparameterization-trick sampling: ``mu + L z``."""
        eps = torch.randn(
            n_samples,
            self._dim,
            device=self.mean.device,
            dtype=self.mean.dtype,
            generator=generator,
        )
        return self.mean + torch.matmul(eps, self.chol.T)

    def entropy(self) -> torch.Tensor:
        d = self._dim
        assert d is not None
        log_det = torch.logdet(self.cov)
        return 0.5 * (d * math.log(2 * math.pi * math.e) + log_det)

    def kl_divergence(self, other: "Gaussian") -> torch.Tensor:
        """``KL(self || other)``. Both must be Gaussians."""
        if not isinstance(other, Gaussian):
            raise GeoBrainError("KL divergence only defined between Gaussians")

        d = self._dim
        trace_term = torch.trace(torch.matmul(other.precision, self.cov))
        diff = other.mean - self.mean
        mean_term = torch.einsum(
            "bi,ij,bj->", diff, other.precision, diff
        )
        _, log_det_other = torch.linalg.slogdet(other.cov)
        _, log_det_self = torch.linalg.slogdet(self.cov)
        log_det_term = log_det_other - log_det_self

        return 0.5 * (trace_term + mean_term - d + log_det_term)

    def __repr__(self) -> str:
        return f"Gaussian(dim={self._dim})"


# ---------------------------------------------------------------------------
# Gaussian mixture
# ---------------------------------------------------------------------------


class GaussianMixture(Distribution, _ModuleBase):
    """Weighted mixture of Gaussians.

    Args:
        means: List of mean vectors.
        covs: List of covariance matrices (one per component).
        weights: Component weights (will be normalized). Defaults to
            uniform.
    """

    components: Sequence[Gaussian]
    weights: torch.Tensor
    log_weights: torch.Tensor

    def __init__(
        self,
        means: list[torch.Tensor],
        covs: list[torch.Tensor],
        weights: torch.Tensor | None = None,
    ) -> None:
        if not means:
            raise GeoBrainError(
                "GaussianMixture requires at least one component; "
                "``means`` is empty."
            )
        dim = means[0].shape[-1] if means[0].dim() > 0 else 1
        super().__init__(dim=dim)

        self.n_components = len(means)
        if len(covs) != self.n_components:
            raise GeoBrainError(
                f"Number of means ({self.n_components}) and "
                f"covariances ({len(covs)}) must match."
            )

        # All components must share one dim: otherwise the mixture constructs
        # with dim = means[0]'s and only fails later inside log_prob with a bare
        # torch broadcasting error. Fail early like the other validations here.
        for k, mean in enumerate(means):
            mdim = mean.shape[-1] if mean.dim() > 0 else 1
            if mdim != dim:
                raise GeoBrainError(
                    f"GaussianMixture components must share one dim; component "
                    f"{k} has dim {mdim} but component 0 has dim {dim}."
                )

        self.components = nn.ModuleList(
            [Gaussian(mean, cov) for mean, cov in zip(means, covs)]
        )

        if weights is None:
            # Build at the component dtype/device so a float64 mixture does not
            # silently get float32 weights / log_weights (the explicit-weights
            # branch below casts to means[0].dtype; match it here).
            weights = torch.full(
                (self.n_components,),
                1.0 / self.n_components,
                dtype=means[0].dtype,
                device=means[0].device,
            )
        else:
            weights = torch.as_tensor(weights, dtype=torch.float64)
            if weights.numel() != self.n_components:
                raise GeoBrainError(
                    f"weights length ({weights.numel()}) must match "
                    f"n_components ({self.n_components})."
                )
            if (weights < 0).any():
                raise GeoBrainError(
                    f"weights must be non-negative; got "
                    f"{weights.tolist()}."
                )
            total = float(weights.sum().item())
            if total <= 0.0:
                raise GeoBrainError(
                    f"weights must sum to a positive value; got "
                    f"sum={total!r}."
                )
            weights = (weights / total).to(means[0].dtype)

        self.register_buffer("weights", weights)
        self.register_buffer("log_weights", torch.log(weights))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Mixture log-density via log-sum-exp."""
        log_probs = torch.stack(
            [
                comp.log_prob(x) + self.log_weights[k]
                for k, comp in enumerate(self.components)
            ],
            dim=1,
        )
        return torch.logsumexp(log_probs, dim=1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Posterior-weighted average of component scores."""
        log_probs = torch.stack(
            [
                comp.log_prob(x) + self.log_weights[k]
                for k, comp in enumerate(self.components)
            ],
            dim=1,
        )
        posteriors = torch.softmax(log_probs, dim=1).unsqueeze(-1)
        scores = torch.stack(
            [comp.score(x) for comp in self.components], dim=1
        )
        return (posteriors * scores).sum(dim=1)

    def sample(
        self, n_samples: int, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Two-stage sampling: cluster index, then component."""
        ref = self.components[0].mean
        if n_samples == 0:
            # Match Gaussian.sample(0) (clean empty tensor) instead of letting
            # torch.multinomial raise on n_sample <= 0.
            return torch.empty(
                (0, self._dim), dtype=ref.dtype, device=ref.device
            )
        indices = torch.multinomial(
            self.weights, n_samples, replacement=True, generator=generator
        )
        samples_list = []
        for k in range(self.n_components):
            n_k = (indices == k).sum().item()
            if n_k > 0:
                samples_list.append(
                    self.components[k].sample(n_k, generator=generator)
                )
        samples = torch.cat(samples_list, dim=0)
        # device= so a non-CPU generator (GPU mixture) doesn't mismatch the
        # default-CPU randperm: every other RNG call here threads device too.
        perm = torch.randperm(n_samples, generator=generator, device=ref.device)
        return samples[perm]

    def component_posteriors(self, x: torch.Tensor) -> torch.Tensor:
        """``p(k | x)`` of shape ``(batch, n_components)``."""
        log_probs = torch.stack(
            [
                comp.log_prob(x) + self.log_weights[k]
                for k, comp in enumerate(self.components)
            ],
            dim=1,
        )
        return torch.softmax(log_probs, dim=1)

    def __repr__(self) -> str:
        return (
            f"GaussianMixture(n_components={self.n_components}, "
            f"dim={self._dim})"
        )
