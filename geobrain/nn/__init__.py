"""
GeoBrain neural network layers.

This subpackage hosts the neural-network building blocks used across
GeoBrain wherever they do not belong to a specific physics family.

The single canonical :class:`LinearFlipout` (and its
:class:`Conv2dFlipout` / :class:`Conv3dFlipout` siblings) lives in
:mod:`geobrain.nn.layers`, alongside :class:`BaseVariationalLayer`, the
:func:`get_kl_loss` aggregator, the standalone :func:`gaussian_kl`
closed-form KL, and a general-purpose :class:`Reshape` block. Every
Flipout ``forward`` returns a plain tensor; KL is obtained via
``kl_loss`` / :func:`get_kl_loss`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .activations import ClippedLinearActivation
from .layers import (
    BaseVariationalLayer,
    Conv2dFlipout,
    Conv3dFlipout,
    LinearFlipout,
    Reshape,
    count_variational_parameters,
    gaussian_kl,
    get_kl_loss,
    kl_regularizer,
)
from .arch import ConvDecoder2d, ConvDecoder3d, CoordinateMLP
from .reparam import LatentReparameterization, WeightReparameterization

__all__ = [
    "ConvDecoder2d",
    "ConvDecoder3d",
    "CoordinateMLP",
    "ClippedLinearActivation",
    "LinearFlipout",
    "gaussian_kl",
    "get_kl_loss",
    "kl_regularizer",
    "BaseVariationalLayer",
    "Conv2dFlipout",
    "Conv3dFlipout",
    "Reshape",
    "count_variational_parameters",
    "LatentReparameterization",
    "WeightReparameterization",
]
