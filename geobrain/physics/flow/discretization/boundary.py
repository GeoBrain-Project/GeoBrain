"""
Flow boundary conditions.

Currently implemented: constant-pressure Dirichlet on a single cell,
modelled as a half-cell transmissibility boundary flux. Positive boundary
flux is always cell-to-exterior and therefore contributes positively to the
canonical cell divergence. More complex BCs
(no-flow at domain edges is the default; aquifer support; time-varying
BHP) can be added without changing the residual interface.

Field units: pressure [psi], face_area [ft²], half_dist [ft],
permeability [mD]. Transmissibility ``T = A · k / d`` is in [mD·ft];
combine with Darcy's ``ALPHA`` and phase mobility ``ρ/(μ·B)`` to get
[bbl/day].

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .._defaults import DEVICE, DTYPE


if TYPE_CHECKING:
    class _ModuleBase:
        """Static interface for ``torch.nn.Module`` when imports are skipped."""

        def __init__(self) -> None:
            pass
else:
    _ModuleBase = nn.Module


@dataclass
class FlowBoundary:
    """
    Constant-pressure Dirichlet BC on one cell face.

    Applied as a canonical outward boundary flux::

        q_out = T_bc · mobility · (p_cell − p_bc)

    where the half-cell transmissibility ``T_bc = A · k / (d/2)`` uses
    only the cell-side geometry (no opposite half-cell).
    """

    cell: int
    pressure: float  # [psi] target boundary pressure
    face_area: float  # [ft²]
    half_dist: float  # [ft] cell-centre to face distance
    permeability: float  # [mD] cell perm projected on the face normal

    @property
    def transmissibility(self) -> float:
        """Half-cell transmissibility ``T_bc`` [mD·ft]."""
        return self.face_area * self.permeability / self.half_dist

    def outward_pressure_drop(self, cell_pressure: torch.Tensor) -> torch.Tensor:
        """Return ``p_cell - p_bc`` in the canonical cell-to-exterior direction."""

        return cell_pressure - cell_pressure.new_tensor(self.pressure)


class FlowBoundaryGroup(_ModuleBase):
    """Container for multiple :class:`FlowBoundary` instances."""

    def __init__(
        self,
        bcs: Iterable[FlowBoundary] | None = None,
        device: torch.device | str = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.bcs: list[FlowBoundary] = list(bcs or [])

    def add(self, bc: FlowBoundary) -> "FlowBoundaryGroup":
        self.bcs.append(bc)
        return self

    def __len__(self) -> int:
        return len(self.bcs)

    def cells_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [b.cell for b in self.bcs],
            dtype=torch.int64,
            device=self.device,
        )

    def pressures_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [b.pressure for b in self.bcs],
            dtype=self.dtype,
            device=self.device,
        )

    def transmissibilities_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [b.transmissibility for b in self.bcs],
            dtype=self.dtype,
            device=self.device,
        )


__all__ = ["FlowBoundary", "FlowBoundaryGroup"]
