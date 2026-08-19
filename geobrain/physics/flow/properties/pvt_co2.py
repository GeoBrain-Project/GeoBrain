"""
CO₂ / brine flow PVT backed by the rock-physics fluid models.

``PVTCO2Brine`` exposes the flow :class:`PVT` interface (``density``,
``viscosity``, ``fvf`` as functions of pressure in Pa). By default it takes its
**density** straight from the rock-physics CO₂ / brine equation of state
(:class:`~geobrain.physics.rock.models._fluid_co2.CO2Properties`, a modified
Batzle-Wang EOS, and the Batzle-Wang brine model). So the *same* differentiable
torch density operator drives the flow residual (buoyancy in the gravity term,
phase mobility, accumulation) **and** the Gassmann/Archie rock-physics forward,
exactly the coupled, end-to-end-differentiable hydro-geophysical path that
motivates a flow module inside an inversion platform (and that a separate
reservoir simulator cannot offer the PyTorch ecosystem).

The density correlation is selectable per phase via ``density_eos``:
``"batzle_wang"`` (default for BOTH phases, the modified Batzle-Wang CO₂ model
and the Batzle-Wang brine model that keep the Gassmann/Archie coupling),
``"redlich_kwong"`` (the adjusted Redlich-Kwong CO₂ EOS of
Spycher-Pruess-Ennis-King, 2003) or ``"rowe_chou"`` (the Rowe & Chou, 1970,
brine density). The last two reproduce the CO₂-storage density tables used by
reservoir simulators. ``"redlich_kwong"`` is a CO₂ EOS and ``"rowe_chou"`` a
brine EOS: requesting an EOS that does not apply to the chosen phase (e.g.
``"rowe_chou"`` for ``"co2"``) falls back to that phase's Batzle-Wang default.

Units are canonical SI: pressure in **Pa**, temperature fixed at construction in
**K**, density returned in **kg/m³**, viscosity in **Pa·s**
(pressure-and-temperature dependent, Fenghour-Wakeham-Vesovic for CO₂,
Mao-Duan for brine), and ``fvf`` on a reservoir-volume basis
``B(p) = ρ(p_ref)/ρ(p)`` (so ``B(p_ref)=1`` and ``B`` carries the phase
compressibility). The CO₂-in-brine solubility (aqueous mole fraction) and the
CO₂ equilibrium ratio ``K`` are exposed via :meth:`PVTCO2Brine.solubility` and
:meth:`PVTCO2Brine.kvalue` (Spycher-Pruess-Ennis-King mutual-solubility model).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from ....core import GeoBrainError
from ...rock.models._fluid_batzlewang import BatzleWang
from ...rock.models._fluid_co2 import CO2Properties
from ..errors import FlowContractError
from .co2_correlations import (
    brine_density_rowe_chou as _brine_density_rowe_chou,
    brine_viscosity as _brine_viscosity,
    co2_density_rk as _co2_density_rk,
    co2_kvalue as _co2_kvalue,
    co2_solubility as _co2_solubility,
    co2_viscosity as _co2_viscosity,
)
from .pvt import PVT

_KELVIN_OFFSET = 273.15


class PVTCO2Brine(PVT):
    """Flow PVT for a CO₂ or brine phase, density from the rock CO₂/brine EOS.

    Args:
        phase: ``"co2"`` or ``"brine"``, which single-phase density to expose.
        temperature_k: reservoir temperature [K] (fixed; isothermal flow).
        salinity_mass_fraction: brine NaCl mass fraction, brine only.
        gas_specific_gravity: CO₂ gas specific gravity, co2 only.
        density_eos: density correlation, selected per phase. ``"batzle_wang"``
            (default for both phases) keeps the modified Batzle-Wang CO₂ density
            / Batzle-Wang brine density that the Gassmann/Archie rock-physics
            forward is coupled to. ``"redlich_kwong"`` selects the adjusted
            Redlich-Kwong CO₂ EOS (Spycher-Pruess-Ennis-King, 2003) for the CO₂
            phase; ``"rowe_chou"`` selects the Rowe & Chou (1970) brine density.
            A phase-incompatible EOS is rejected.
        constant_viscosity_pa_s: constant-viscosity fallback [Pa·s], used only when
            ``constant_viscosity=True`` (kept for backward compatibility with
            callers that pinned a fixed μ).
        constant_viscosity: if ``True`` return ``constant_viscosity_pa_s`` regardless of
            pressure; default ``False`` uses the pressure-temperature
            correlation (Fenghour-Wakeham-Vesovic CO₂ / Mao-Duan brine).
        reference_pressure_pa: reference pressure [Pa] where ``B = 1``.
    """

    _MW_NACL = 0.0584428  # NaCl molar mass [kg/mol]

    def __init__(
        self,
        phase: Literal["co2", "brine"] = "co2",
        *,
        temperature_k: float = 313.15,
        salinity_mass_fraction: float = 0.03,
        gas_specific_gravity: float = 1.5349,
        density_eos: Literal["batzle_wang", "redlich_kwong", "rowe_chou"] = "batzle_wang",
        constant_viscosity_pa_s: float = 5.0e-5,
        constant_viscosity: bool = False,
        reference_pressure_pa: float = 27_579_029.172672,
    ) -> None:
        super().__init__()
        if phase not in ("co2", "brine"):
            raise GeoBrainError(f"phase must be 'co2' or 'brine', got {phase!r}")
        if density_eos not in ("batzle_wang", "redlich_kwong", "rowe_chou"):
            raise GeoBrainError(
                "density_eos must be 'batzle_wang', 'redlich_kwong' or "
                f"'rowe_chou', got {density_eos!r}"
            )
        compatible = {
            "co2": {"batzle_wang", "redlich_kwong"},
            "brine": {"batzle_wang", "rowe_chou"},
        }
        if density_eos not in compatible[phase]:
            raise FlowContractError(
                "density_eos is incompatible with the selected phase",
                object_name="PVTCO2Brine",
                field="density_eos",
                expected=tuple(sorted(compatible[phase])),
                actual=density_eos,
            )
        scalar_domains = {
            "temperature_k": (temperature_k, 0.0, None),
            "salinity_mass_fraction": (salinity_mass_fraction, 0.0, 1.0),
            "gas_specific_gravity": (gas_specific_gravity, 0.0, None),
            "constant_viscosity_pa_s": (constant_viscosity_pa_s, 0.0, None),
            "reference_pressure_pa": (reference_pressure_pa, 0.0, None),
        }
        for field, (value, lower, upper) in scalar_domains.items():
            valid = math.isfinite(float(value)) and (
                float(value) >= lower if field == "salinity_mass_fraction" else float(value) > lower
            )
            if upper is not None:
                valid = valid and float(value) < upper
            if not valid:
                raise FlowContractError(
                    f"{field} is outside its physical domain",
                    object_name="PVTCO2Brine",
                    field=field,
                    expected=(f"> {lower}", None if upper is None else f"< {upper}"),
                    actual=value,
                )
        self.phase = phase
        self.temperature_k = float(temperature_k)
        self.salinity_mass_fraction = float(salinity_mass_fraction)
        self.gas_specific_gravity = float(gas_specific_gravity)
        self.density_eos = density_eos
        self.constant_viscosity_pa_s = float(constant_viscosity_pa_s)
        self.constant_viscosity = bool(constant_viscosity)
        self.reference_pressure_pa = float(reference_pressure_pa)
        self._co2 = CO2Properties()
        self._bw = BatzleWang()

    def _validate_pressure(self, pressure_pa: torch.Tensor) -> torch.Tensor:
        if not isinstance(pressure_pa, torch.Tensor) or not pressure_pa.is_floating_point():
            raise FlowContractError(
                "pressure_pa must be a floating tensor",
                object_name="PVTCO2Brine",
                field="pressure_pa",
                expected="floating torch.Tensor",
                actual=type(pressure_pa).__name__,
            )
        if not bool(torch.isfinite(pressure_pa).all()) or bool((pressure_pa <= 0).any()):
            raise FlowContractError(
                "pressure_pa must be positive and finite",
                object_name="PVTCO2Brine",
                field="pressure_pa",
                expected="> 0 Pa",
                actual="contains a non-positive or non-finite value",
            )
        return pressure_pa

    def _temperature_kelvin(self, pressure_pa: torch.Tensor) -> torch.Tensor:
        return pressure_pa.new_tensor(self.temperature_k)

    def _nacl_molality(self) -> float:
        """NaCl molality [mol/kg solvent] from the brine mass fraction."""
        w = self.salinity_mass_fraction
        if w <= 0.0:
            return 0.0
        return w / (self._MW_NACL * (1.0 - w))

    def _density_kg_m3(self, pressure_pa: torch.Tensor) -> torch.Tensor:
        """Phase density [kg/m³] from the selected EOS at pressure [Pa].

        CO₂: modified Batzle-Wang (``density_eos="batzle_wang"``, default) or the
        adjusted Redlich-Kwong EOS (``density_eos="redlich_kwong"``). Brine:
        Batzle-Wang brine EOS (default) or the Rowe & Chou (1970) brine density
        (``density_eos="rowe_chou"``). An EOS that does not apply to the chosen
        phase falls back to that phase's Batzle-Wang default.
        """
        p = self._validate_pressure(pressure_pa)
        T_K = self._temperature_kelvin(p)
        T_C = p.new_tensor(self.temperature_k - _KELVIN_OFFSET)
        if self.phase == "co2":
            if self.density_eos == "redlich_kwong":
                return _co2_density_rk(p, T_K)
            rho, _ = self._co2(p * 1.0e-6, T_C, self.gas_specific_gravity)
        else:
            if self.density_eos == "rowe_chou":
                return _brine_density_rowe_chou(p, T_K, self.salinity_mass_fraction)
            S = p.new_tensor(self.salinity_mass_fraction)
            rho, _ = self._bw.brine(T_C, p * 1.0e-6, S)
        return rho

    def density(self, p: torch.Tensor) -> torch.Tensor:
        """Phase density [kg/m³] at pressure ``p`` [Pa]."""
        return self._density_kg_m3(p)

    def viscosity(self, p: torch.Tensor) -> torch.Tensor:
        """Phase viscosity [Pa·s] at pressure ``p`` [Pa].

        CO₂: Fenghour-Wakeham-Vesovic (1998) on the Redlich-Kwong density.
        Brine: Mao-Duan (2009) pure-water viscosity with the NaCl factor and
        the Islam-Carlson dissolved-CO₂ term (salt-free reduces to pure water).
        Set ``constant_viscosity=True`` to recover the legacy fixed value.
        """
        p = self._validate_pressure(p)
        if self.constant_viscosity:
            return torch.full_like(p, self.constant_viscosity_pa_s)
        T_K = self._temperature_kelvin(p)
        if self.phase == "co2":
            mu_pas = _co2_viscosity(p, T_K)
        else:
            mu_pas = _brine_viscosity(p, T_K, self._nacl_molality(), 0.0)
        return mu_pas

    def solubility(self, p: torch.Tensor) -> torch.Tensor:
        """Aqueous CO₂ solubility: CO₂ mole fraction ``x_CO2`` in the brine.

        Spycher-Pruess-Ennis-King (2003) mutual-solubility model. Increases with
        pressure; salt-free here (the salt activity correction is not applied).
        """
        p = self._validate_pressure(p)
        return _co2_solubility(p, self._temperature_kelvin(p))

    def kvalue(self, p: torch.Tensor) -> torch.Tensor:
        """CO₂ equilibrium ratio ``K_CO2 = (1 − y_H2O) / x_CO2`` at ``p`` [Pa].

        The vapour/aqueous mole-fraction ratio of the CO₂ component
        (Spycher-Pruess-Ennis-King, 2003), falls steeply with pressure as more
        CO₂ dissolves into the brine.
        """
        p = self._validate_pressure(p)
        return _co2_kvalue(p, self._temperature_kelvin(p))

    def fvf(self, p: torch.Tensor) -> torch.Tensor:
        """Formation volume factor on a reservoir-volume basis, ``B = ρ(p_ref)/ρ(p)``."""
        p = self._validate_pressure(p)
        p_ref = p.new_tensor(self.reference_pressure_pa)
        return self._density_kg_m3(p_ref) / self._density_kg_m3(p)
