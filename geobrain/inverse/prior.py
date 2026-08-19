"""
Priors for the model-space layer.

A ``Prior`` is anything with ``log_prob(state) -> scalar`` and
``sample(generator) -> ModelState``. ``GaussianPrior`` is the default; further
priors (TV, Laplacian, GMM, NN-decoder) drop in as separate dataclasses with the
same contract.

When *not* to use this layer:
    ``Prior`` operates on a structured :class:`ModelState` (named-field
    semantics, the "model space"). For tensor-level probability machinery on
    flat ``(batch, dim)`` tensors, analytical ``log_prob`` / ``score`` /
    ``sample`` for sampler targets, flipout reparameterizations, or generic
    likelihood components; see :mod:`geobrain.bayes.distributions`
    (``Gaussian``, ``GaussianMixture``).

Three layers, three homes:

- :mod:`geobrain.optim`: deterministic regularizers (no ``log_prob``).
- :mod:`geobrain.inverse.prior`: probabilistic priors over a :class:`ModelState`.
- :mod:`geobrain.bayes.distributions`: probabilistic distributions over raw tensors.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import torch

from ..core.validation import validate_std_positive
from ..core.errors import GeoBrainError
from ..core.containers import ModelState


@runtime_checkable
class Prior(Protocol):
    """A prior over ModelState fields. Used by Bayesian solvers."""

    def log_prob(self, state: ModelState) -> torch.Tensor: ...

    def sample(self, generator: torch.Generator | None = None) -> ModelState: ...


@dataclass(frozen=True)
class GaussianPrior:
    """
    Independent Gaussian prior on a subset of ModelState fields.

    Fields not listed in ``mean`` contribute 0 to ``log_prob``, i.e. they are
    given an improper flat prior.

    Attributes:
        mean: Per-field mean tensors. Each must match the field's expected shape.
        std: Per-field std. Scalars apply to every element; non-scalar tensors
            must have the same shape, dtype, and device as ``mean``.
    """

    mean: Mapping[str, torch.Tensor]
    std: Mapping[str, float | torch.Tensor]

    def __post_init__(self) -> None:
        mean = dict(self.mean)
        std = dict(self.std)
        for field_name, mapping in (("mean", mean), ("std", std)):
            for name in mapping:
                if not isinstance(name, str) or not name:
                    raise GeoBrainError(
                        "GaussianPrior field names must be non-empty strings",
                        object_name="GaussianPrior",
                        field=field_name,
                        expected="non-empty string keys",
                        actual=name,
                    )
        if set(mean) != set(std):
            raise GeoBrainError(
                "GaussianPrior.mean and .std must cover the same fields",
                object_name="GaussianPrior",
                field="std",
                expected=sorted(mean),
                actual=sorted(std),
            )
        for name, m in mean.items():
            if not isinstance(m, torch.Tensor):
                raise GeoBrainError(
                    "GaussianPrior.mean values must be torch.Tensor",
                    object_name="GaussianPrior",
                    field=f"mean[{name!r}]",
                    expected=torch.Tensor,
                    actual=type(m),
                )
            s = std[name]
            if not isinstance(s, (int, float, torch.Tensor)):
                raise GeoBrainError(
                    "GaussianPrior.std values must be scalar or Tensor",
                    object_name="GaussianPrior",
                    field=f"std[{name!r}]",
                    expected="scalar or Tensor",
                    actual=type(s),
                )
            if isinstance(s, torch.Tensor):
                if s.dtype != m.dtype:
                    raise GeoBrainError(
                        "GaussianPrior.std tensor dtype must match mean dtype",
                        object_name="GaussianPrior",
                        field=f"std[{name!r}].dtype",
                        expected=str(m.dtype),
                        actual=str(s.dtype),
                    )
                if s.device != m.device:
                    raise GeoBrainError(
                        "GaussianPrior.std tensor device must match mean device",
                        object_name="GaussianPrior",
                        field=f"std[{name!r}].device",
                        expected=str(m.device),
                        actual=str(s.device),
                    )
                if s.ndim > 0 and tuple(s.shape) != tuple(m.shape):
                    raise GeoBrainError(
                        "GaussianPrior.std tensor must match mean shape",
                        object_name="GaussianPrior",
                        field=f"std[{name!r}]",
                        expected=tuple(m.shape),
                        actual=tuple(s.shape),
                    )
            validate_std_positive(
                s,
                object_name="GaussianPrior",
                field=f"std[{name!r}]",
            )
        object.__setattr__(self, "mean", MappingProxyType(mean))
        object.__setattr__(self, "std", MappingProxyType(std))

    def log_prob(self, state: ModelState) -> torch.Tensor:
        """
        Gaussian log-prior summed over the configured fields.

        Args:
            state: Model parameters. Every field named in ``mean`` must be
                present with exactly matching shape, dtype, and device.

        Returns:
            Scalar log-prior, or ``0`` when no fields are configured.
        """
        total: torch.Tensor | None = None
        for name, m in self.mean.items():
            (value,) = state.fetch(name)
            if tuple(value.shape) != tuple(m.shape):
                raise GeoBrainError(
                    "state field must match prior mean shape",
                    object_name="GaussianPrior.log_prob",
                    field=name,
                    expected=tuple(m.shape),
                    actual=tuple(value.shape),
                )
            if value.dtype != m.dtype:
                raise GeoBrainError(
                    "state field dtype must match prior mean dtype",
                    object_name="GaussianPrior.log_prob",
                    field=f"{name}.dtype",
                    expected=str(m.dtype),
                    actual=str(value.dtype),
                )
            if value.device != m.device:
                raise GeoBrainError(
                    "state field device must match prior mean device",
                    object_name="GaussianPrior.log_prob",
                    field=f"{name}.device",
                    expected=str(m.device),
                    actual=str(value.device),
                )
            s = self.std[name]
            sigma = (
                s
                if isinstance(s, torch.Tensor)
                else value.new_tensor(float(s))
            )
            r2 = ((value - m) / sigma).pow(2)
            if isinstance(s, torch.Tensor):
                log_scale = (
                    sigma.log().sum()
                    if s.ndim > 0
                    else value.numel() * sigma.log()
                )
            else:
                log_scale = value.numel() * math.log(float(s))
            term = -0.5 * r2.sum() - log_scale
            total = term if total is None else total + term
        if total is None:
            # No fields in mean → flat prior (log_prob = 0)
            anchor = next(iter(state.tensors.values()), None)
            if anchor is None:
                return torch.zeros(())
            return torch.zeros((), dtype=anchor.dtype, device=anchor.device)
        return total

    def sample(self, generator: torch.Generator | None = None) -> ModelState:
        """
        Draw a :class:`ModelState` from the prior.

        Args:
            generator: Optional :class:`torch.Generator` for reproducibility.

        Returns:
            A :class:`ModelState` with one sample per configured field.
        """
        tensors: dict[str, torch.Tensor] = {}
        for name, m in self.mean.items():
            s = self.std[name]
            sigma = (
                s
                if isinstance(s, torch.Tensor)
                else m.new_tensor(float(s))
            ).expand_as(m)
            noise = torch.randn(m.shape, generator=generator,
                                dtype=m.dtype, device=m.device)
            tensors[name] = m + sigma * noise
        return ModelState(tensors=tensors)
