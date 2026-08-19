"""Agent-discoverable Rock operator registry.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from ._base import RockForwardOperator, RockOperatorDeclaration
from .anisotropy import Hudson, SayersKachanov
from .elastic import ModuliFromVelocities, VelocitiesFromModuli
from .empirical import ArchieResistivity, Gardner, KozenyCarman
from .fluids import BatzleWangBrine, BiotHighFrequency, Gassmann, WoodFluidMix
from .frames import HertzMindlin, SelfConsistent
from .qi import RockPhysicsTemplate

ROCK_OPERATOR_TYPES: tuple[type[RockForwardOperator], ...] = (
    VelocitiesFromModuli,
    ModuliFromVelocities,
    WoodFluidMix,
    BatzleWangBrine,
    Gassmann,
    BiotHighFrequency,
    HertzMindlin,
    SelfConsistent,
    Hudson,
    SayersKachanov,
    Gardner,
    ArchieResistivity,
    KozenyCarman,
    RockPhysicsTemplate,
)

_ROCK_OPERATOR_BY_NAME = {
    operator.declaration.model: operator for operator in ROCK_OPERATOR_TYPES
}


def get_rock_operator(name: str) -> type[RockForwardOperator]:
    """Return a facade class by stable model id."""

    try:
        return _ROCK_OPERATOR_BY_NAME[name]
    except KeyError as error:
        choices = ", ".join(sorted(_ROCK_OPERATOR_BY_NAME))
        raise KeyError(f"unknown Rock operator {name!r}; available: {choices}") from error


__all__ = [
    "ArchieResistivity",
    "BatzleWangBrine",
    "BiotHighFrequency",
    "Gardner",
    "Gassmann",
    "HertzMindlin",
    "Hudson",
    "KozenyCarman",
    "ModuliFromVelocities",
    "ROCK_OPERATOR_TYPES",
    "RockForwardOperator",
    "RockOperatorDeclaration",
    "RockPhysicsTemplate",
    "SayersKachanov",
    "SelfConsistent",
    "VelocitiesFromModuli",
    "WoodFluidMix",
    "get_rock_operator",
]
