"""
Neural-network architectures for inversion reparameterization.

Three plain ``nn.Module`` architectures whose weights or latent inputs can
serve as inversion variables:

- :class:`ConvDecoder2d` / :class:`ConvDecoder3d`: upsampling conv
  decoders that map a small latent code to a model grid, for
  deep-image-prior (DIP) and latent-space inversion.
- :class:`CoordinateMLP`: a coordinate → field network; its
  ``layer_factory`` hook swaps the linear-layer constructor, so passing
  :class:`~geobrain.nn.LinearFlipout` turns it into a variational
  (Bayesian) network.

None of them is inversion-aware by itself. Wrap one in
:class:`~geobrain.nn.LatentReparameterization` (optimise the latent code)
or :class:`~geobrain.nn.WeightReparameterization` (optimise the weights)
to expose it to an :class:`~geobrain.optim.Inverter` as a differentiable
parameterization.

This module is deliberately small: an architecture earns a place here only
once an in-tree inversion example uses it.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn

from geobrain.core.errors import GeoBrainError

from .activations import ClippedLinearActivation

__all__ = ["ConvDecoder2d", "ConvDecoder3d", "CoordinateMLP"]


def _resolve_final_activation(
    final_activation: str | nn.Module | None, owner: str
) -> nn.Module | None:
    if final_activation is None:
        return None
    if isinstance(final_activation, nn.Module):
        return final_activation
    table: dict[str, Callable[[], nn.Module]] = {
        "sigmoid": nn.Sigmoid,
        "clip01": ClippedLinearActivation,
    }
    if isinstance(final_activation, str) and final_activation in table:
        return table[final_activation]()
    raise GeoBrainError(
        f"{owner} final_activation must be None, 'sigmoid', 'clip01' or an nn.Module",
        object_name=owner,
        field="final_activation",
        expected="None | 'sigmoid' | 'clip01' | nn.Module",
        actual=final_activation,
    )


class _ConvDecoderNd(nn.Module):
    """conv → upsample → ReLU stages, then a final conv (+ optional activation)."""

    _conv_cls: type[nn.Module]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: Sequence[int],
        scale_factor: int = 2,
        kernel_size: int = 3,
        final_activation: str | nn.Module | None = None,
    ) -> None:
        super().__init__()
        owner = type(self).__name__
        hidden = tuple(hidden_channels)
        if not hidden or any((not isinstance(h, int)) or h <= 0 for h in hidden):
            raise GeoBrainError(
                f"{owner} hidden_channels must be a non-empty sequence of positive ints",
                object_name=owner,
                field="hidden_channels",
                expected="non-empty positive ints",
                actual=hidden_channels,
            )
        for label, val in (("in_channels", in_channels), ("out_channels", out_channels)):
            if not isinstance(val, int) or val <= 0:
                raise GeoBrainError(
                    f"{owner} {label} must be a positive integer",
                    object_name=owner,
                    field=label,
                    expected="positive int",
                    actual=val,
                )
        if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
            raise GeoBrainError(
                f"{owner} kernel_size must be a positive odd integer "
                "(same-padding via kernel_size // 2 only holds for odd sizes; "
                "an even size would silently break the documented "
                "scale_factor ** len(hidden_channels) magnification)",
                object_name=owner,
                field="kernel_size",
                expected="positive odd int",
                actual=kernel_size,
            )
        if not isinstance(scale_factor, int) or scale_factor < 1:
            raise GeoBrainError(
                f"{owner} scale_factor must be a positive integer",
                object_name=owner,
                field="scale_factor",
                expected="positive int",
                actual=scale_factor,
            )
        pad = kernel_size // 2
        layers: list[nn.Module] = []
        prev = in_channels
        for h in hidden:
            layers += [
                self._conv_cls(prev, h, kernel_size, 1, pad),
                # nearest-neighbour by design: DIP-style decoders rely on the
                # blocky prior; switch to a bilinear stage explicitly if a
                # smooth interpolant is wanted.
                nn.Upsample(scale_factor=scale_factor),
                nn.ReLU(),
            ]
            prev = h
        layers.append(self._conv_cls(prev, out_channels, kernel_size, 1, pad))
        act = _resolve_final_activation(final_activation, owner)
        if act is not None:
            layers.append(act)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder2d(_ConvDecoderNd):
    """2-D conv decoder: each hidden stage upsamples spatially by ``scale_factor``.

    Args:
        in_channels: latent channels of the input code.
        out_channels: channels of the decoded field.
        hidden_channels: one entry per stage; each stage is
            conv -> nearest-neighbour upsample -> ReLU.
        scale_factor: per-stage spatial magnification; the total is
            ``scale_factor ** len(hidden_channels)``.
        kernel_size: odd conv kernel size (same-padding; even sizes are
            rejected).
        final_activation: ``None`` | ``'sigmoid'`` | ``'clip01'`` | any
            ``nn.Module``, applied after the last conv.

    Shape:
        ``(N, in_channels, H, W)`` -> ``(N, out_channels, H*s**k, W*s**k)``
        with ``s = scale_factor`` and ``k = len(hidden_channels)``.

    Example:
        >>> import torch
        >>> dec = ConvDecoder2d(8, 1, hidden_channels=(32, 16))
        >>> dec(torch.randn(1, 8, 16, 16)).shape
        torch.Size([1, 1, 64, 64])
    """

    _conv_cls = nn.Conv2d


class ConvDecoder3d(_ConvDecoderNd):
    """3-D conv decoder: each hidden stage upsamples spatially by ``scale_factor``.

    Args:
        in_channels: latent channels of the input code.
        out_channels: channels of the decoded field.
        hidden_channels: one entry per stage; each stage is
            conv -> nearest-neighbour upsample -> ReLU.
        scale_factor: per-stage spatial magnification; the total is
            ``scale_factor ** len(hidden_channels)``.
        kernel_size: odd conv kernel size (same-padding; even rejected).
        final_activation: ``None`` | ``'sigmoid'`` | ``'clip01'`` | any
            ``nn.Module``, applied after the last conv.

    Shape:
        ``(N, in_channels, D, H, W)`` ->
        ``(N, out_channels, D*s**k, H*s**k, W*s**k)`` with
        ``s = scale_factor`` and ``k = len(hidden_channels)``.

    Example:
        >>> import torch
        >>> dec = ConvDecoder3d(8, 1, hidden_channels=(32, 32, 16))
        >>> dec(torch.randn(1, 8, 4, 4, 4)).shape   # x8 upsampling
        torch.Size([1, 1, 32, 32, 32])

    The example configuration is the reference DIP decoder the seismic
    inversion examples use.
    """

    _conv_cls = nn.Conv3d


class CoordinateMLP(nn.Module):
    """Coordinate → field MLP; ``layer_factory`` makes it variational.

    Args:
        in_dim: coordinate dimensionality of one input point.
        out_dim: field components predicted per point.
        hidden: width of every hidden layer.
        depth: number of linear layers (>= 1); hidden activations sit
            between them, so ``depth == 1`` is a single linear map.
        activation: zero-argument factory for the hidden activation
            (default ``nn.Tanh``).
        layer_factory: ``layer_factory(in_features, out_features)`` builds
            each linear layer (default ``nn.Linear``); pass
            :class:`~geobrain.nn.LinearFlipout` for a Bayesian network whose
            KL is collected by :func:`~geobrain.nn.get_kl_loss`.

    Shape:
        ``(..., in_dim)`` -> ``(..., out_dim)``, any number of leading
        batch dimensions.

    Example:
        >>> import torch
        >>> mlp = CoordinateMLP(3, 1, hidden=32, depth=3)
        >>> mlp(torch.randn(100, 3)).shape
        torch.Size([100, 1])
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden: int = 64,
        depth: int = 3,
        activation: Callable[[], nn.Module] = nn.Tanh,
        layer_factory: Callable[[int, int], nn.Module] = nn.Linear,
    ) -> None:
        super().__init__()
        if not isinstance(depth, int) or depth < 1:
            raise GeoBrainError(
                "CoordinateMLP depth must be an integer >= 1",
                object_name="CoordinateMLP",
                field="depth",
                expected=">= 1",
                actual=depth,
            )
        for label, val in (("in_dim", in_dim), ("out_dim", out_dim), ("hidden", hidden)):
            if not isinstance(val, int) or val <= 0:
                raise GeoBrainError(
                    f"CoordinateMLP {label} must be a positive integer",
                    object_name="CoordinateMLP",
                    field=label,
                    expected="positive int",
                    actual=val,
                )
        layers: list[nn.Module] = []
        if depth == 1:
            layers.append(layer_factory(in_dim, out_dim))
        else:
            layers.append(layer_factory(in_dim, hidden))
            for _ in range(depth - 2):
                layers += [activation(), layer_factory(hidden, hidden)]
            layers += [activation(), layer_factory(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
