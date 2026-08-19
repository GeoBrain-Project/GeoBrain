"""Wave memory-strategy selection.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import cast

from ...errors import WaveContractError
from ..contracts import WaveMemoryProtocol
from .boundary import BoundaryMemory
from .checkpoint import CheckpointMemory
from .full import FullMemory
from .recursive import RecursiveMemory


def create_memory_strategy(strategy: str) -> WaveMemoryProtocol:
    """Return the exact strategy object selected by immutable configuration."""
    strategies = {
        "full": FullMemory,
        "checkpoint": CheckpointMemory,
        "recursive": RecursiveMemory,
        "boundary": BoundaryMemory,
    }
    try:
        strategy_type = strategies[strategy]
    except KeyError as exc:
        raise WaveContractError(
            "unknown Wave memory strategy",
            object_name="create_memory_strategy",
            field="strategy",
            expected=tuple(strategies),
            actual=strategy,
        ) from exc
    return cast(WaveMemoryProtocol, strategy_type())


__all__ = [
    "BoundaryMemory",
    "CheckpointMemory",
    "FullMemory",
    "RecursiveMemory",
    "create_memory_strategy",
]
