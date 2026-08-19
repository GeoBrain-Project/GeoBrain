"""
Permeability ``ComponentModel`` implementations.

Ten correlations for permeability
from porosity / grain-size / saturation / pore-geometry parameters.

Pure-math layer; downstream operator wrappers (or :class:`RockPhysicsTransform`)
adapt these to the ``ModelState`` contract as needed.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from torch import Tensor

from ._categories import PermeabilityModel
from ._registry import register
from ..petrophysics import (
    bernabe_permeability,
    bloch_porosity_permeability,
    fredrich_permeability,
    kozeny_carman_percolation_permeability,
    kozeny_carman_permeability,
    owolabi_permeability,
    panda_lake_cemented_permeability,
    panda_lake_permeability,
    perm_logs,
    revil_permeability,
)


@register("KozenyCarman", aliases=["kc", "kozeny_carman"])
class KozenyCarman(PermeabilityModel):
    """``k = d² φ³ / (180 (1 − φ)²)``."""

    def forward(self, phi: Tensor, d: Tensor) -> Tensor:
        return kozeny_carman_permeability(phi, d)


@register("KozenyCarmanPercolation", aliases=["kc_perc"])
class KozenyCarmanPercolation(PermeabilityModel):
    """
    KC with percolation threshold: ``k = B d² (φ − φ_c)³ / (1 + φ_c − φ)²``.

    Clamps ``(φ − φ_c)`` to ≥ 0, below the percolation threshold the
    formula has no physical meaning; the lower bound k = 0 keeps
    gradients well-behaved during inversion.
    """

    def forward(
        self, phi: Tensor, phi_c: Tensor, d: Tensor, B: Tensor,
    ) -> Tensor:
        return kozeny_carman_percolation_permeability(phi, phi_c, d, B)


@register("Owolabi", aliases=["owolabi"])
class Owolabi(PermeabilityModel):
    """Owolabi unconsolidated-sand permeability: returns ``(k_oil, k_gas)``."""

    def forward(self, phi: Tensor, Swi: Tensor) -> tuple[Tensor, Tensor]:
        return owolabi_permeability(phi, Swi)


@register("PermLogs", aliases=["perm_logs"])
class PermLogs(PermeabilityModel):
    """
    Tixier / Timur / Coates / Coates-Dumanoir log correlations.

    Returns the 4-tuple ``(k_Tixier, k_Timur, k_Coates, k_CoatesDumanoir)``.
    """

    def forward(
        self, phi: Tensor, Swi: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return perm_logs(phi, Swi)


@register("PandaLake", aliases=["panda_lake"])
class PandaLake(PermeabilityModel):
    """Panda-Lake modified KC with grain-size + cementation parameters."""

    def forward(
        self, d: Tensor, C: Tensor, S: Tensor, tau: Tensor, phi: Tensor,
    ) -> Tensor:
        return panda_lake_permeability(d, C, S, tau, phi)


@register("PandaLakeCem", aliases=["panda_lake_cem"])
class PandaLakeCem(PermeabilityModel):
    """Cemented-sand variant: ``k = 3.34 d² φ³ / (1 − φ)²``."""

    def forward(self, phi: Tensor, d: Tensor) -> Tensor:
        return panda_lake_cemented_permeability(phi, d)


@register("Revil", aliases=["revil"])
class Revil(PermeabilityModel):
    """Revil shaly-rock permeability: ``k = 1000 d² φ^4.5 / 24``."""

    def forward(self, phi: Tensor, d: Tensor) -> Tensor:
        return revil_permeability(phi, d)


@register("Fredrich", aliases=["fredrich"])
class Fredrich(PermeabilityModel):
    """Fredrich pore-geometry permeability with formation-factor correction."""

    def forward(self, phi: Tensor, d: Tensor, b: Tensor) -> Tensor:
        return fredrich_permeability(phi, d, b)


@register("Bloch", aliases=["bloch"])
class Bloch(PermeabilityModel):
    """Bloch relation from clay fraction; return ``(φ, k_m2)`` jointly."""

    def forward(
        self, S: Tensor, C: Tensor, D: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return bloch_porosity_permeability(S, C, D)


@register("Bernabe", aliases=["bernabe"])
class Bernabe(PermeabilityModel):
    """Bernabe dual-porosity (cracks + tubes) permeability."""

    def forward(
        self, phi: Tensor, crf: Tensor, w: Tensor, r: Tensor,
    ) -> Tensor:
        return bernabe_permeability(phi, crf, w, r)


__all__ = [
    "Bernabe", "Bloch",
    "Fredrich",
    "KozenyCarman", "KozenyCarmanPercolation",
    "Owolabi",
    "PandaLake", "PandaLakeCem",
    "PermLogs",
    "Revil",
]
