"""Small injected contracts for decision workflows.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

import torch

from geobrain.core.context import ForwardContext


@runtime_checkable
class EnsembleUpdater(Protocol):
    """Stateful ensemble update operation injected into a closed loop."""

    @property
    def ensemble(self) -> torch.Tensor:
        """Return the updater's current ensemble."""
        ...

    def update(
        self,
        target: Any,
        *,
        n_steps: int,
        ctx: ForwardContext | None,
    ) -> torch.Tensor:
        """Update against ``target`` and return the current ensemble."""
        ...


@runtime_checkable
class CancellationCheck(Protocol):
    """Cooperative cancellation check evaluated only at safe boundaries."""

    def __call__(self) -> bool:
        """Return ``True`` when the pending workflow should stop."""
        ...


class HistoryPolicy(str, Enum):
    """Tensor-retention policy for closed-loop history."""

    NONE = "none"
    LATEST = "latest"
    ALL = "all"


__all__ = ["CancellationCheck", "EnsembleUpdater", "HistoryPolicy"]
