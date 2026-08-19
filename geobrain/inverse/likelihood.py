"""
Likelihood models for the data-misfit layer.

A ``Likelihood`` is anything that, given ``(predicted, observed)`` channel
mappings, returns a scalar misfit and a matching ``log_likelihood``.
``GaussianLikelihood`` (i.i.d. Gaussian noise, std σ) is the default; further
models (Laplace, Cauchy, Student-t, robust IRLS) drop in as additional frozen
dataclasses with the same contract.

The contract is a :class:`Protocol`, so user code that supplies its own
``misfit`` / ``log_likelihood`` callable is structurally compatible without
inheritance.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Protocol, runtime_checkable

import torch

from ..core.validation import validate_std_positive
from ..core.errors import GeoBrainError, MissingFieldError


def _validate_common_channel_tensors(
    predicted: Mapping[str, torch.Tensor],
    observed: Mapping[str, torch.Tensor],
    channels: Iterable[str],
    *,
    object_name: str,
) -> None:
    """Reject implicit shape, dtype, or device coercion on common channels.

    Only channels present in both mappings are checked. Callers retain their
    existing, owner-specific missing-channel errors for absent entries.
    """
    for channel in channels:
        if channel not in predicted or channel not in observed:
            continue
        pred = predicted[channel]
        obs = observed[channel]
        if tuple(pred.shape) != tuple(obs.shape):
            raise GeoBrainError(
                "predicted channel shape must match observed channel shape",
                object_name=object_name,
                field=f"{channel}.shape",
                expected=tuple(obs.shape),
                actual=tuple(pred.shape),
            )
        if pred.dtype != obs.dtype:
            raise GeoBrainError(
                "predicted channel dtype must match observed channel dtype",
                object_name=object_name,
                field=f"{channel}.dtype",
                expected=str(obs.dtype),
                actual=str(pred.dtype),
            )
        if pred.device != obs.device:
            raise GeoBrainError(
                "predicted channel device must match observed channel device",
                object_name=object_name,
                field=f"{channel}.device",
                expected=str(obs.device),
                actual=str(pred.device),
            )


@runtime_checkable
class Likelihood(Protocol):
    """Contract: given (predicted_data, observed_data) mappings, return a scalar."""

    def misfit(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor: ...

    def log_likelihood(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class GaussianLikelihood:
    """
    I.i.d. Gaussian likelihood.

    The misfit is ``0.5 * Σ_c Σ_i |(d_c[i] - g_c[i]) / σ_c|²``; the log-likelihood
    is ``-misfit`` up to a dropped additive constant. Using the squared modulus
    ``|·|²`` keeps the chi-squared misfit real and correct for native-complex
    channels (e.g. MT impedance); for real channels ``|x|² == x²``, so real
    misfits are unchanged.

    Attributes:
        std: Scalar applied to all channels, or a per-channel mapping (channel →
            scalar or tensor). A tensor std is either 0-d (a scalar applied to
            every element) or must match that channel's data shape, dtype, and
            device exactly. Partial/broadcastable shapes and implicit tensor
            coercions are rejected by :meth:`misfit`.
    """

    std: float | Mapping[str, float | torch.Tensor]

    def __post_init__(self) -> None:
        """Validate that every ``std`` entry is a positive scalar or tensor."""
        if isinstance(self.std, Mapping):
            std = dict(self.std)
            for name, s in std.items():
                if not isinstance(name, str) or not name:
                    raise GeoBrainError(
                        "GaussianLikelihood.std keys must be non-empty strings",
                        object_name="GaussianLikelihood",
                        field="std",
                        expected="non-empty string keys",
                        actual=name,
                    )
                if not isinstance(s, (int, float, torch.Tensor)):
                    raise GeoBrainError(
                        "GaussianLikelihood.std values must be scalar or Tensor",
                        object_name="GaussianLikelihood",
                        field=f"std[{name!r}]",
                        actual=type(s),
                    )
                if not isinstance(s, torch.Tensor):
                    validate_std_positive(
                        s,
                        object_name="GaussianLikelihood",
                        field=f"std[{name!r}]",
                    )
            object.__setattr__(self, "std", MappingProxyType(std))
        elif not isinstance(self.std, (int, float, torch.Tensor)):
            raise GeoBrainError(
                "GaussianLikelihood.std must be a scalar, Tensor, or Mapping",
                object_name="GaussianLikelihood",
                field="std",
                actual=type(self.std),
            )
        elif not isinstance(self.std, torch.Tensor):
            validate_std_positive(self.std, object_name="GaussianLikelihood", field="std")

    def _std_for(self, channel: str) -> float | torch.Tensor:
        """Return the std for ``channel`` (scalar std applies to every channel)."""
        if isinstance(self.std, Mapping):
            if channel not in self.std:
                raise MissingFieldError(
                    "GaussianLikelihood.std missing entry for channel",
                    object_name="GaussianLikelihood",
                    field=channel,
                    expected=f"present in {sorted(self.std)}",
                )
            return self.std[channel]
        return self.std

    def misfit(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Half-chi-squared data misfit, summed over channels.

        Args:
            predicted: Forward-model output, keyed by channel.
            observed: Observed data, keyed by channel. Every observed channel
                must be present in ``predicted`` with matching shape, dtype, and
                device.

        Returns:
            Scalar ``0.5 * Σ |(predicted - observed) / σ|²``.

        Raises:
            MissingFieldError: If an observed channel is absent from ``predicted``.
            GeoBrainError: On tensor-contract mismatch or empty ``observed``.
        """
        total: torch.Tensor | None = None
        _validate_common_channel_tensors(
            predicted,
            observed,
            observed,
            object_name="GaussianLikelihood.misfit",
        )
        for channel, obs in observed.items():
            if channel not in predicted:
                raise MissingFieldError(
                    "predicted is missing a channel that observed has",
                    object_name="GaussianLikelihood.misfit",
                    field=channel,
                    expected=f"present in predicted {sorted(predicted)}",
                )
            pred = predicted[channel]
            sigma = self._std_for(channel)
            # A non-scalar tensor std must match the data exactly: a partial
            # shape (e.g. per-column on row×column data) would BROADCAST and
            # silently produce the wrong chi-squared. (A 0-d / Python-scalar std
            # is the documented "applies to every element" case; tensor scalars
            # are explicitly aligned to the data contract.) Mirrors
            # GaussianPrior's std-vs-mean shape guard.
            if isinstance(sigma, torch.Tensor):
                if sigma.dtype != obs.dtype:
                    raise GeoBrainError(
                        "GaussianLikelihood.std tensor dtype must match data dtype",
                        object_name="GaussianLikelihood.misfit",
                        field=f"{channel}.std.dtype",
                        expected=str(obs.dtype),
                        actual=str(sigma.dtype),
                    )
                if sigma.device != obs.device:
                    raise GeoBrainError(
                        "GaussianLikelihood.std tensor device must match data device",
                        object_name="GaussianLikelihood.misfit",
                        field=f"{channel}.std.device",
                        expected=str(obs.device),
                        actual=str(sigma.device),
                    )
                if sigma.ndim > 0:
                    if tuple(sigma.shape) != tuple(obs.shape):
                        raise GeoBrainError(
                            "GaussianLikelihood.std tensor must match the data shape",
                            object_name="GaussianLikelihood.misfit",
                            field=f"{channel}.std.shape",
                            expected=tuple(obs.shape),
                            actual=tuple(sigma.shape),
                        )
                std_field = (
                    f"std[{channel!r}]"
                    if isinstance(self.std, Mapping)
                    else "std"
                )
                validate_std_positive(
                    sigma,
                    object_name="GaussianLikelihood",
                    field=std_field,
                )
            else:
                sigma = obs.new_tensor(float(sigma))
            residual = (pred - obs) / sigma
            # ``abs().pow(2)`` gives |residual|² so the chi-squared misfit is
            # real and correct for native-complex channels (e.g. MT impedance).
            # For real tensors |x|² == x², so this is bit-identical there.
            term = 0.5 * residual.abs().pow(2).sum()
            total = term if total is None else total + term
        if total is None:
            raise GeoBrainError(
                "observed is empty; nothing to compute misfit over",
                object_name="GaussianLikelihood.misfit",
                field="observed",
                expected="at least one channel",
                actual={},
            )
        return total

    def log_likelihood(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Log-likelihood ``-misfit`` (up to a dropped additive constant)."""
        return -self.misfit(predicted, observed)
