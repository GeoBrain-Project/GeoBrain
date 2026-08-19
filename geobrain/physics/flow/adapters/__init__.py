"""Explicit external-unit and reservoir-deck adapters for Flow.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .eclipse import read_eclipse_case, read_eclipse_deck_si
from .field_units import (
    compressibility_pa_inv_to_psi_inv,
    compressibility_psi_inv_to_pa_inv,
    density_kg_m3_to_lbm_ft3,
    density_lbm_ft3_to_kg_m3,
    gas_rate_m3_s_to_scf_day,
    gas_rate_scf_day_to_m3_s,
    length_ft_to_m,
    length_m_to_ft,
    liquid_rate_m3_s_to_stb_day,
    liquid_rate_stb_day_to_m3_s,
    permeability_m2_to_md,
    permeability_md_to_m2,
    pressure_pa_to_psi,
    pressure_psi_to_pa,
    temperature_c_to_k,
    temperature_k_to_c,
    time_day_to_s,
    time_s_to_day,
    viscosity_cp_to_pa_s,
    viscosity_pa_s_to_cp,
)
from .grdecl import parse_grdecl_si, read_grdecl_grid_si

__all__ = (
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
    "parse_grdecl_si",
    "permeability_m2_to_md",
    "permeability_md_to_m2",
    "pressure_pa_to_psi",
    "pressure_psi_to_pa",
    "read_eclipse_case",
    "read_eclipse_deck_si",
    "read_grdecl_grid_si",
    "time_day_to_s",
    "time_s_to_day",
    "temperature_c_to_k",
    "temperature_k_to_c",
    "viscosity_cp_to_pa_s",
    "viscosity_pa_s_to_cp",
)
