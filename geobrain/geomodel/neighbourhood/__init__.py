"""Exact deterministic spatial neighbourhood backends.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .contracts import NeighbourhoodBackend, NeighbourhoodSelection, NeighbourhoodSpec
from .exhaustive import ExhaustiveNeighbourhood
from .kdtree import DynamicKDTreeNeighbourhood, StaticKDTreeNeighbourhood
from .regular_grid import RegularGridNeighbourhood

__all__ = [
    "DynamicKDTreeNeighbourhood",
    "ExhaustiveNeighbourhood",
    "NeighbourhoodBackend",
    "NeighbourhoodSelection",
    "NeighbourhoodSpec",
    "RegularGridNeighbourhood",
    "StaticKDTreeNeighbourhood",
]
