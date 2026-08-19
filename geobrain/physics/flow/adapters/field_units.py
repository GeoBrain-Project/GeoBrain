"""Named tensor-preserving conversions between FIELD units and canonical SI.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from ..errors import FlowContractError

_PA_PER_PSI = 6894.757293168
_M_PER_FT = 0.3048
_M2_PER_MD = 9.869233e-16
_PA_S_PER_CP = 1.0e-3
_S_PER_DAY = 86400.0
_M3_PER_STB = 0.158987294928
_M3_PER_SCF = 0.028316846592
_KG_M3_PER_LBM_FT3 = 16.01846337396014


def _scale(value: torch.Tensor, factor: float, *, object_name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise FlowContractError(
            "Flow unit adapter requires a torch.Tensor",
            object_name=object_name,
            field="value",
            expected="torch.Tensor",
            actual=type(value).__qualname__,
        )
    if not value.is_floating_point():
        raise FlowContractError(
            "Flow unit adapter requires a floating tensor",
            object_name=object_name,
            field="dtype",
            expected="floating torch.dtype",
            actual=str(value.dtype),
        )
    return value * factor


def pressure_psi_to_pa(value: torch.Tensor) -> torch.Tensor:
    """Convert pressure from psi to Pa without changing tensor metadata."""
    return _scale(value, _PA_PER_PSI, object_name="pressure_psi_to_pa")


def pressure_pa_to_psi(value: torch.Tensor) -> torch.Tensor:
    """Convert pressure from Pa to psi without changing tensor metadata."""
    return _scale(value, 1.0 / _PA_PER_PSI, object_name="pressure_pa_to_psi")


def compressibility_psi_inv_to_pa_inv(value: torch.Tensor) -> torch.Tensor:
    """Convert compressibility from psi⁻¹ to Pa⁻¹."""
    return _scale(
        value,
        1.0 / _PA_PER_PSI,
        object_name="compressibility_psi_inv_to_pa_inv",
    )


def compressibility_pa_inv_to_psi_inv(value: torch.Tensor) -> torch.Tensor:
    """Convert compressibility from Pa⁻¹ to psi⁻¹."""
    return _scale(
        value,
        _PA_PER_PSI,
        object_name="compressibility_pa_inv_to_psi_inv",
    )


def density_lbm_ft3_to_kg_m3(value: torch.Tensor) -> torch.Tensor:
    """Convert density from lbm/ft³ to kg/m³."""
    return _scale(value, _KG_M3_PER_LBM_FT3, object_name="density_lbm_ft3_to_kg_m3")


def density_kg_m3_to_lbm_ft3(value: torch.Tensor) -> torch.Tensor:
    """Convert density from kg/m³ to lbm/ft³."""
    return _scale(value, 1.0 / _KG_M3_PER_LBM_FT3, object_name="density_kg_m3_to_lbm_ft3")


def temperature_c_to_k(value: torch.Tensor) -> torch.Tensor:
    """Convert degrees Celsius to kelvin without changing tensor metadata."""
    return _scale(value, 1.0, object_name="temperature_c_to_k") + 273.15


def temperature_k_to_c(value: torch.Tensor) -> torch.Tensor:
    """Convert kelvin to degrees Celsius without changing tensor metadata."""
    return _scale(value, 1.0, object_name="temperature_k_to_c") - 273.15


def length_ft_to_m(value: torch.Tensor) -> torch.Tensor:
    """Convert length from international feet to metres."""
    return _scale(value, _M_PER_FT, object_name="length_ft_to_m")


def length_m_to_ft(value: torch.Tensor) -> torch.Tensor:
    """Convert length from metres to international feet."""
    return _scale(value, 1.0 / _M_PER_FT, object_name="length_m_to_ft")


def permeability_md_to_m2(value: torch.Tensor) -> torch.Tensor:
    """Convert permeability from millidarcy to square metres."""
    return _scale(value, _M2_PER_MD, object_name="permeability_md_to_m2")


def permeability_m2_to_md(value: torch.Tensor) -> torch.Tensor:
    """Convert permeability from square metres to millidarcy."""
    return _scale(value, 1.0 / _M2_PER_MD, object_name="permeability_m2_to_md")


def viscosity_cp_to_pa_s(value: torch.Tensor) -> torch.Tensor:
    """Convert dynamic viscosity from cP to Pa·s."""
    return _scale(value, _PA_S_PER_CP, object_name="viscosity_cp_to_pa_s")


def viscosity_pa_s_to_cp(value: torch.Tensor) -> torch.Tensor:
    """Convert dynamic viscosity from Pa·s to cP."""
    return _scale(value, 1.0 / _PA_S_PER_CP, object_name="viscosity_pa_s_to_cp")


def time_day_to_s(value: torch.Tensor) -> torch.Tensor:
    """Convert time from days to seconds."""
    return _scale(value, _S_PER_DAY, object_name="time_day_to_s")


def time_s_to_day(value: torch.Tensor) -> torch.Tensor:
    """Convert time from seconds to days."""
    return _scale(value, 1.0 / _S_PER_DAY, object_name="time_s_to_day")


def liquid_rate_stb_day_to_m3_s(value: torch.Tensor) -> torch.Tensor:
    """Convert stock-tank liquid rate from STB/day to m³/s."""
    return _scale(
        value,
        _M3_PER_STB / _S_PER_DAY,
        object_name="liquid_rate_stb_day_to_m3_s",
    )


def liquid_rate_m3_s_to_stb_day(value: torch.Tensor) -> torch.Tensor:
    """Convert stock-tank liquid rate from m³/s to STB/day."""
    return _scale(
        value,
        _S_PER_DAY / _M3_PER_STB,
        object_name="liquid_rate_m3_s_to_stb_day",
    )


def gas_rate_scf_day_to_m3_s(value: torch.Tensor) -> torch.Tensor:
    """Convert standard-gas rate from scf/day to m³/s."""
    return _scale(
        value,
        _M3_PER_SCF / _S_PER_DAY,
        object_name="gas_rate_scf_day_to_m3_s",
    )


def gas_rate_m3_s_to_scf_day(value: torch.Tensor) -> torch.Tensor:
    """Convert standard-gas rate from m³/s to scf/day."""
    return _scale(
        value,
        _S_PER_DAY / _M3_PER_SCF,
        object_name="gas_rate_m3_s_to_scf_day",
    )


__all__ = [
    "compressibility_pa_inv_to_psi_inv",
    "compressibility_psi_inv_to_pa_inv",
    "density_kg_m3_to_lbm_ft3",
    "density_lbm_ft3_to_kg_m3",
    "gas_rate_m3_s_to_scf_day",
    "gas_rate_scf_day_to_m3_s",
    "length_ft_to_m",
    "length_m_to_ft",
    "liquid_rate_m3_s_to_stb_day",
    "liquid_rate_stb_day_to_m3_s",
    "permeability_m2_to_md",
    "permeability_md_to_m2",
    "pressure_pa_to_psi",
    "pressure_psi_to_pa",
    "temperature_c_to_k",
    "temperature_k_to_c",
    "time_day_to_s",
    "time_s_to_day",
    "viscosity_cp_to_pa_s",
    "viscosity_pa_s_to_cp",
]
