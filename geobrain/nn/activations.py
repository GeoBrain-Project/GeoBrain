# pyright: reportPrivateImportUsage=false
"""
Custom activation functions for GeoBrain.

Provides activations for Bayesian networks (KL divergence is collected
via :func:`geobrain.nn.get_kl_loss`; activations themselves are plain
``nn.Module`` passes).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ClippedLinearActivation"]


class ClippedLinearActivation(nn.Module):
    """
    Clipped linear activation that constrains outputs to ``[0, 1]``.

    Equivalent to ``torch.clamp(x, 0.0, 1.0)``, wrapped as an
    :class:`nn.Module` so it can slot into :class:`nn.Sequential`
    pipelines. For a post-step parameter projection instead of a network
    layer, prefer :class:`geobrain.optim.processing.BoundsClamp`.

    Example:
        >>> activation = ClippedLinearActivation()
        >>> x = torch.tensor([-0.5, 0.3, 0.8, 1.5])
        >>> output = activation(x)
        >>> print(output)  # tensor([0.0, 0.3, 0.8, 1.0])
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0)
