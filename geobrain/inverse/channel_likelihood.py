"""
Composite likelihood over heterogeneous per-channel children.

:class:`ChannelLikelihood` lets a joint problem (e.g. a seismic gather channel
scored with a waveform misfit alongside a gravity channel scored with a plain
:class:`~geobrain.inverse.likelihood.GaussianLikelihood`) declare ONE
``Likelihood`` for :class:`~geobrain.inverse.inverse_problem.InverseProblem`
even though each channel wants a genuinely different noise model / objective.

There is deliberately no ``weights=`` constructor argument: the value this
class adds is the heterogeneous child *types*, not a per-channel scalar
weight, per-channel noise std (the statistical weight) already belongs on
each child's own constructor (``GaussianLikelihood(std=...)``, a waveform
misfit's own parameters, ...).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from ..core.errors import GeoBrainError, MissingFieldError
from .likelihood import Likelihood, _validate_common_channel_tensors

__all__ = ["ChannelLikelihood"]


@dataclass(frozen=True)
class ChannelLikelihood:
    """
    Sum of per-channel child likelihoods, each scored on its own single-channel
    view of ``(predicted, observed)``.

    ``misfit(predicted, observed) = Σ_c channels[c].misfit({c: predicted[c]},
    {c: observed[c]})`` over the channels declared in :attr:`channels`.
    ``log_likelihood`` is ``-misfit``, matching every other ``Likelihood`` in
    this package.

    Attributes:
        channels: Non-empty mapping of channel name -> ``Likelihood``-protocol
            object scoring that channel alone. Frozen (stored as a
            ``MappingProxyType``) so a ``ChannelLikelihood`` instance is
            immutable, matching :class:`GaussianLikelihood`.
    """

    channels: Mapping[str, Likelihood]

    def __post_init__(self) -> None:
        """Validate ``channels`` is a non-empty mapping of named children."""
        if not isinstance(self.channels, Mapping):
            raise GeoBrainError(
                "ChannelLikelihood.channels must be a Mapping",
                object_name="ChannelLikelihood",
                field="channels",
                expected=Mapping,
                actual=type(self.channels),
            )
        channels = dict(self.channels)
        if not channels:
            raise GeoBrainError(
                "ChannelLikelihood requires at least one channel",
                object_name="ChannelLikelihood",
                field="channels",
                expected="non-empty mapping",
                actual={},
            )
        for name in channels:
            if not isinstance(name, str) or not name:
                raise GeoBrainError(
                    "ChannelLikelihood.channels keys must be non-empty strings",
                    object_name="ChannelLikelihood",
                    field="channels",
                    actual=name,
                )
        object.__setattr__(self, "channels", MappingProxyType(channels))

    def misfit(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        ``Σ_c channels[c].misfit(single-channel views)`` over declared channels.

        Args:
            predicted: Forward-model output, keyed by channel. Must contain
                every channel declared in :attr:`channels`.
            observed: Observed data, keyed by channel. Must contain every
                channel declared in :attr:`channels`.

        Returns:
            Scalar total misfit, summed over channels.

        Raises:
            MissingFieldError: If a declared channel is absent from
                ``predicted`` or from ``observed``.
        """
        total: torch.Tensor | None = None
        _validate_common_channel_tensors(
            predicted,
            observed,
            self.channels,
            object_name="ChannelLikelihood.misfit",
        )
        for name, child in self.channels.items():
            if name not in predicted:
                raise MissingFieldError(
                    "predicted is missing a channel declared on ChannelLikelihood",
                    object_name="ChannelLikelihood.misfit",
                    field=name,
                    expected=f"present in predicted {sorted(predicted)}",
                )
            if name not in observed:
                raise MissingFieldError(
                    "observed is missing a channel declared on ChannelLikelihood",
                    object_name="ChannelLikelihood.misfit",
                    field=name,
                    expected=f"present in observed {sorted(observed)}",
                )
            term = child.misfit({name: predicted[name]}, {name: observed[name]})
            total = term if total is None else total + term
        assert total is not None  # __post_init__ guarantees channels is non-empty
        return total

    def log_likelihood(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Log-likelihood ``-misfit`` (up to a dropped additive constant)."""
        return -self.misfit(predicted, observed)

    @property
    def gradient_singular(self) -> bool:
        """``True`` iff ANY child declares ``gradient_singular = True``.

        Read via ``getattr(child, "gradient_singular", False)`` so a bare
        protocol-conforming child with no such attribute contributes ``False``
        rather than raising. Consumed by
        :class:`~geobrain.optim.Inverter`'s ``nan_policy="guard"``
        auto-detection the same way a plain misfit's own
        ``gradient_singular`` ``ClassVar`` is.
        """
        return any(
            bool(getattr(child, "gradient_singular", False))
            for child in self.channels.values()
        )
