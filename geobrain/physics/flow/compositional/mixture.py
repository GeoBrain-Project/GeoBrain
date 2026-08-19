"""
Multi-component mixture: per-species critical properties + Wilson K-values.

A compositional fluid is a set of :class:`Component` species, each carrying the
critical properties a cubic EOS needs (critical pressure / temperature, acentric
factor, molar mass), plus an optional binary-interaction-parameter (BIP) matrix.
:class:`Mixture` stacks them into tensors and provides the Wilson correlation for
the equilibrium-ratio (K-value) initial guess that seeds the flash.

Units are **SI** (the EOS convention): pressure in Pa, temperature in K, molar
mass in kg/mol, critical volume in m³/mol.

Wilson (1969) K-value estimate::

    K_i = (p_c,i / p) · exp[ 5.37·(1 + ω_i)·(1 − T_c,i / T) ]

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from ....core import GeoBrainError
from .._defaults import DEVICE, DTYPE
from ..errors import FlowContractError


@dataclass(frozen=True, slots=True)
class Component:
    """Critical properties of one species (SI units).

    Args:
        name:  species label (e.g. ``"CO2"``, ``"C1"``).
        molar_mass_kg_mol: molar mass [kg/mol].
        critical_pressure_pa: critical pressure [Pa].
        critical_temperature_k: critical temperature [K].
        acentric_factor: acentric factor [-].
        critical_volume_m3_mol: critical volume [m³/mol] (optional).
    """

    name: str
    molar_mass_kg_mol: float
    critical_pressure_pa: float
    critical_temperature_k: float
    acentric_factor: float
    critical_volume_m3_mol: float = 0.0

    def __post_init__(self) -> None:
        dimensional = (
            self.molar_mass_kg_mol,
            self.critical_pressure_pa,
            self.critical_temperature_k,
            self.critical_volume_m3_mol,
        )
        if not all(math.isfinite(value) for value in dimensional):
            raise FlowContractError(
                "component properties must be finite",
                object_name=f"Component[{self.name}]",
                field=(
                    "molar_mass_kg_mol/critical_pressure_pa/"
                    "critical_temperature_k/critical_volume_m3_mol"
                ),
                expected="finite canonical SI values",
                actual=dimensional,
            )
        if not (
            self.molar_mass_kg_mol > 0
            and self.critical_pressure_pa > 0
            and self.critical_temperature_k > 0
            and self.critical_volume_m3_mol >= 0
        ):
            raise GeoBrainError(
                f"Component {self.name}: molar mass and critical properties must be positive"
            )
        if not math.isfinite(self.acentric_factor) or self.acentric_factor < -1.0:
            raise GeoBrainError(f"Component {self.name}: acentric factor must be ≥ −1")


class Mixture(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """A multi-component mixture: stacked critical properties + BIP matrix.

    Args:
        components: sequence of :class:`Component`.
        bip:        optional ``(n, n)`` binary-interaction-parameter matrix
            (symmetric, zero diagonal); defaults to all zeros.
    """

    def __init__(
        self,
        components: Sequence[Component],
        bip: torch.Tensor | None = None,
        device: str | torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        components = list(components)
        if len(components) < 2:
            raise GeoBrainError("Mixture needs at least 2 components for a flash")
        self.components = components
        self.names = [c.name for c in components]
        self.device = torch.device(device)
        self.dtype = dtype
        n = len(components)

        def _buf(vals: Sequence[float]) -> torch.Tensor:
            return torch.tensor(vals, device=self.device, dtype=self.dtype)

        self.register_buffer(
            "critical_pressure_pa", _buf([c.critical_pressure_pa for c in components])
        )
        self.register_buffer(
            "critical_temperature_k", _buf([c.critical_temperature_k for c in components])
        )
        self.register_buffer("acentric_factor", _buf([c.acentric_factor for c in components]))
        self.register_buffer("molar_mass_kg_mol", _buf([c.molar_mass_kg_mol for c in components]))
        self.register_buffer(
            "critical_volume_m3_mol", _buf([c.critical_volume_m3_mol for c in components])
        )
        if bip is None:
            bip_t = torch.zeros((n, n), device=self.device, dtype=self.dtype)
        else:
            if not isinstance(bip, torch.Tensor) or not bip.is_floating_point():
                raise FlowContractError(
                    "bip must be a floating tensor",
                    object_name="Mixture",
                    field="bip",
                    expected="floating torch.Tensor",
                    actual=type(bip).__name__,
                )
            if bip.dtype != self.dtype:
                raise FlowContractError(
                    "bip dtype must match the declared mixture dtype",
                    object_name="Mixture",
                    field="bip.dtype",
                    expected=str(self.dtype),
                    actual=str(bip.dtype),
                )
            if bip.device != self.device:
                raise FlowContractError(
                    "bip device must match the declared mixture device",
                    object_name="Mixture",
                    field="bip.device",
                    expected=str(self.device),
                    actual=str(bip.device),
                )
            bip_t = bip
            if bip_t.shape != (n, n):
                raise GeoBrainError(f"bip must be ({n}, {n}), got {tuple(bip_t.shape)}")
            if not bool(torch.isfinite(bip_t).all()):
                raise FlowContractError(
                    "bip must contain only finite values",
                    object_name="Mixture",
                    field="bip",
                    expected="finite values",
                    actual="contains NaN or infinity",
                )
        self.register_buffer("bip", bip_t)

    @property
    def n_components(self) -> int:
        return len(self.components)

    def wilson_k(self, p: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Wilson K-value estimate ``K_i(p, T)`` [-].

        ``p`` and ``T`` may be scalars or shape ``(...,)`` (e.g. per cell); the
        result is ``(..., n_components)`` broadcasting the species axis last.
        """
        self.validate_state_tensors(p, T, object_name="Mixture.wilson_k")
        p_e = p.unsqueeze(-1)
        T_e = T.unsqueeze(-1)
        return (self.critical_pressure_pa / p_e) * torch.exp(
            5.37 * (1.0 + self.acentric_factor) * (1.0 - self.critical_temperature_k / T_e)
        )

    def validate_state_tensors(
        self,
        p: object,
        T: object,
        *,
        object_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate canonical-SI state tensors without casting or moving them."""

        checked: list[torch.Tensor] = []
        for field, value in (("pressure_pa", p), ("temperature_k", T)):
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise FlowContractError(
                    f"{field} must be a floating tensor",
                    object_name=object_name,
                    field=field,
                    expected="floating torch.Tensor",
                    actual=type(value).__name__,
                )
            if value.dtype != self.dtype:
                raise FlowContractError(
                    f"{field} dtype must match the mixture dtype",
                    object_name=object_name,
                    field=f"{field}.dtype",
                    expected=str(self.dtype),
                    actual=str(value.dtype),
                )
            if value.device != self.device:
                raise FlowContractError(
                    f"{field} device must match the mixture device",
                    object_name=object_name,
                    field=f"{field}.device",
                    expected=str(self.device),
                    actual=str(value.device),
                )
            if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
                raise FlowContractError(
                    f"{field} must be positive and finite",
                    object_name=object_name,
                    field=field,
                    expected="> 0 in canonical SI units",
                    actual="contains a non-positive or non-finite value",
                )
            checked.append(value)
        return checked[0], checked[1]

    def validate_composition_tensor(
        self,
        composition: object,
        *,
        object_name: str,
    ) -> torch.Tensor:
        """Validate a component-last mole-fraction tensor in place."""

        if (
            not isinstance(composition, torch.Tensor)
            or not composition.is_floating_point()
            or composition.ndim < 1
            or composition.shape[-1] != self.n_components
        ):
            raise FlowContractError(
                "composition must be a floating component-last tensor",
                object_name=object_name,
                field="composition",
                expected=f"floating [..., {self.n_components}] tensor",
                actual=(
                    type(composition).__name__,
                    tuple(getattr(composition, "shape", ())),
                ),
            )
        if composition.dtype != self.dtype or composition.device != self.device:
            raise FlowContractError(
                "composition metadata must match the mixture",
                object_name=object_name,
                field="composition.dtype/device",
                expected=(str(self.dtype), str(self.device)),
                actual=(str(composition.dtype), str(composition.device)),
            )
        if not bool(torch.isfinite(composition).all()) or bool((composition < 0).any()):
            raise FlowContractError(
                "composition must be finite and non-negative",
                object_name=object_name,
                field="composition",
                expected="finite mole fractions >= 0",
                actual="contains a negative or non-finite value",
            )
        return composition
