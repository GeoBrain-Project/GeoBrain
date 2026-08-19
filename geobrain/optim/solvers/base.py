"""
Optimizer ABC and the shared execution-contract seam.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch

from ...core import ForwardContext
from ..config import AdamConfig, OptimizerConfig
from ..execution import CancellationToken, OptimizationCallback
from ..results import OptimizationResult

__all__ = ["Optimizer", "OptimizationResult"]


def _compile_optimizer(
    config: OptimizerConfig,
    params: Iterable[torch.Tensor],
) -> torch.optim.Optimizer:
    """Compile every immutable config field into its torch optimizer."""
    if isinstance(config, AdamConfig):
        return torch.optim.Adam(
            params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    return torch.optim.LBFGS(
        params,
        lr=config.lr,
        history_size=config.history_size,
        max_iter=config.max_iter,
        tolerance_grad=config.tolerance_grad,
        tolerance_change=config.tolerance_change,
        line_search_fn=config.line_search_fn,
    )


class Optimizer(ABC):
    """
    Optimizer ABC. Subclasses implement :meth:`run`.

    Constructor convention:
        ``__init__(self, problem, params: Mapping[str, Tensor], **kwargs)``,
        ``params`` maps ModelState field name → leaf tensor with
        ``requires_grad=True``. On each iteration the optimizer builds
        ``ModelState(tensors=params)`` and lets autograd flow back to those
        tensors. The shared :data:`OptimizerConfig` records compile
        into the concrete torch optimizer; the alias is imported here so the
        base seam and implementations share one configuration vocabulary.
    """

    optimizer_config_type: type[OptimizerConfig] | None = None

    @abstractmethod
    def run(
        self,
        n_iters: int,
        ctx: ForwardContext | None = None,
        callback: OptimizationCallback | None = None,
        cancellation: CancellationToken | None = None,
    ) -> OptimizationResult:
        """Run ``n_iters`` optimization steps and return an :class:`OptimizationResult`."""
        ...
