"""SI material-parameter descriptors used by electromagnetic operators.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Conductivity:
    """Conductivity ``σ`` in S/m, conventionally stored as ``sigma``."""

    field_name: str = "sigma"


@dataclass(frozen=True, slots=True)
class Resistivity:
    """Electrical resistivity ``ρ`` in Ω·m, conventionally ``rho_e``."""

    field_name: str = "rho_e"


@dataclass(frozen=True, slots=True)
class Permeability:
    """Relative magnetic permeability, conventionally ``mu_r``.

    Attributes:
        field_name: ModelState field carrying the property.
        default: value used when the field is absent.
    """

    field_name: str = "mu_r"
    default: float = 1.0


@dataclass(frozen=True, slots=True)
class Permittivity:
    """Relative electric permittivity, conventionally ``eps_r``.

    Attributes:
        field_name: ModelState field carrying the property.
        default: value used when the field is absent.
    """

    field_name: str = "eps_r"
    default: float = 1.0


__all__ = ["Conductivity", "Permeability", "Permittivity", "Resistivity"]
