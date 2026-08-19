"""
Resistivity ``ComponentModel`` implementations.

Currently one model
(:class:`ArchieResistivity`) plus inverse / helper methods.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from torch import Tensor

from ._categories import ResistivityModel
from ._registry import register
from ..petrophysics import (
    archie_formation_factor,
    archie_resistivity,
    archie_resistivity_index,
    archie_water_saturation,
)


@register("ArchieResistivity", aliases=["archie", "Archie"])
class ArchieResistivity(ResistivityModel):
    """
    Archie's law ``R_t = a · R_w · φ^(−m) · S_w^(−n)``.

    Args:
        Rw: Water resistivity (ohm·m).
        a:  Tortuosity factor.
    """

    def __init__(self, Rw: float = 0.17, a: float = 1.0) -> None:
        super().__init__()
        self.Rw = float(Rw)
        self.a = float(a)

    def forward(
        self, poro: Tensor, Sw: Tensor, m: float | Tensor, n: float | Tensor,
    ) -> Tensor:
        return archie_resistivity(
            poro,
            Sw,
            self.Rw,
            tortuosity_factor=self.a,
            cementation_exponent=m,
            saturation_exponent=n,
        )

    def formation_factor(self, poro: Tensor, m: float | Tensor) -> Tensor:
        """``F = a / φ^m``."""
        return archie_formation_factor(
            poro, tortuosity_factor=self.a, cementation_exponent=m
        )

    def resistivity_index(self, Sw: Tensor, n: float | Tensor) -> Tensor:
        """``I = Sw^(−n)``."""
        return archie_resistivity_index(Sw, saturation_exponent=n)

    def water_saturation(
        self, Rt: Tensor, poro: Tensor,
        m: float | Tensor, n: float | Tensor,
    ) -> Tensor:
        """Solve Archie for ``Sw`` and reject results outside ``(0, 1]``."""
        return archie_water_saturation(
            Rt,
            poro,
            self.Rw,
            tortuosity_factor=self.a,
            cementation_exponent=m,
            saturation_exponent=n,
        )

    def __repr__(self) -> str:
        return f"ArchieResistivity(Rw={self.Rw}, a={self.a})"


__all__ = ["ArchieResistivity"]
