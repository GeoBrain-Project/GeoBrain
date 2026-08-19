"""
Lohrenz-Bray-Clark (LBC) compositional phase viscosity.

A PyTorch, differentiable, cell-vectorized implementation of the LBC viscosity
correlation, the standard correlation for compositional phase viscosities,
replacing the constant-μ placeholder:

    μ = 1e-3·(μ* + (Σ_k a_k ρ_r^k)^4 − 1e-4) / ξ_mix)      [Pa·s]

where ξ = T_c^{1/6} / (√(1000·MW)·(p_c/101325)^{2/3}) is the viscosity-reducing
parameter (paper units: g/mol, atm), ρ_r = V_pc / V the reduced density (V_pc =
Σ z_i V_{c,i} the pseudo-critical molar volume, V = Z·R·T/p the phase molar
volume), μ* the dilute-gas (atmospheric) mixture viscosity (Stiel-Thodos per
component, mole-/√MW-weighted), and ``a_k`` the LBC polynomial coefficients.

Units are SI (Pa, K, kg/mol, m³/mol). Differentiable in pressure, composition and
the phase compressibility ``Z``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from numbers import Real

import torch

from ..errors import FlowContractError
from .mixture import Mixture

_R_GAS = 8.31446261815324  # J/(mol·K)
_LBC_COEFF = (0.1023, 0.023364, 0.058533, -0.040758, 0.0093324)
_LBC_SHIFT = -1e-4
_ATM = 101325.0  # Pa


def _viscosity_parameter(
    mw: torch.Tensor,
    T_c: torch.Tensor,
    p_c: torch.Tensor,
) -> torch.Tensor:
    """ξ = T_c^{1/6} / (√(1000·MW)·(p_c/atm)^{2/3}) (kg/mol, Pa in; paper g/mol, atm)."""
    return T_c.pow(1.0 / 6.0) / (torch.sqrt(1000.0 * mw) * (p_c / _ATM).pow(2.0 / 3.0))


def lbc_viscosity(
    mixture: Mixture,
    p: torch.Tensor,
    temperature: Real | torch.Tensor,
    composition: torch.Tensor,
    Z: torch.Tensor,
) -> torch.Tensor:
    """LBC phase viscosity [Pa·s], vectorized over cells.

    Args:
        mixture: a :class:`Mixture` with canonical-SI molar mass, critical
            pressure, critical temperature, and critical volume buffers.
        p: ``(...)`` pressure [Pa].
        temperature: T [K], a scalar (isothermal) **or** a per-cell ``(...)`` tensor
            (thermal: ``T`` is a primary variable, so viscosity tracks it cell-by-cell
            and stays differentiable in ``T``).
        composition: ``(..., n_c)`` phase mole fractions (liquid ``x`` or vapor ``y``).
        Z: ``(...)`` phase compressibility factor.
    """
    composition = mixture.validate_composition_tensor(
        composition,
        object_name="lbc_viscosity",
    )
    if not isinstance(p, torch.Tensor) or not p.is_floating_point():
        raise FlowContractError(
            "pressure_pa must be a floating tensor",
            object_name="lbc_viscosity",
            field="pressure_pa",
            expected="floating torch.Tensor",
            actual=type(p).__name__,
        )
    if isinstance(temperature, torch.Tensor):
        T = temperature
    elif isinstance(temperature, Real) and not isinstance(temperature, bool):
        if not math.isfinite(float(temperature)) or float(temperature) <= 0:
            raise FlowContractError(
                "temperature_k must be positive and finite",
                object_name="lbc_viscosity",
                field="temperature_k",
                expected="> 0 K",
                actual=temperature,
            )
        T = p.new_tensor(float(temperature))
    else:
        raise FlowContractError(
            "temperature_k must be a scalar or floating tensor",
            object_name="lbc_viscosity",
            field="temperature_k",
            expected="real scalar or floating torch.Tensor",
            actual=type(temperature).__name__,
        )
    p, T = mixture.validate_state_tensors(
        p,
        T,
        object_name="lbc_viscosity",
    )
    if (
        not isinstance(Z, torch.Tensor)
        or not Z.is_floating_point()
        or Z.dtype != mixture.dtype
        or Z.device != mixture.device
    ):
        raise FlowContractError(
            "compressibility metadata must match the mixture",
            object_name="lbc_viscosity",
            field="Z.dtype/device",
            expected=(str(mixture.dtype), str(mixture.device)),
            actual=(
                type(Z).__name__,
                str(getattr(Z, "dtype", None)),
                str(getattr(Z, "device", None)),
            ),
        )
    if not bool(torch.isfinite(Z).all()) or bool((Z <= 0).any()):
        raise FlowContractError(
            "compressibility factor must be positive and finite",
            object_name="lbc_viscosity",
            field="Z",
            expected="> 0",
            actual="contains a non-positive or non-finite value",
        )
    batch_shape = composition.shape[:-1]
    for field, value in (("pressure_pa", p), ("temperature_k", T), ("Z", Z)):
        try:
            result_shape = torch.broadcast_shapes(value.shape, batch_shape)
        except RuntimeError as error:
            raise FlowContractError(
                f"{field} does not broadcast over the composition batch",
                object_name="lbc_viscosity",
                field=field,
                expected=tuple(batch_shape),
                actual=tuple(value.shape),
            ) from error
        if result_shape != batch_shape:
            raise FlowContractError(
                f"{field} would expand the composition batch",
                object_name="lbc_viscosity",
                field=field,
                expected=tuple(batch_shape),
                actual=tuple(result_shape),
            )

    mw = mixture.molar_mass_kg_mol
    pc = mixture.critical_pressure_pa
    Tc = mixture.critical_temperature_k
    Vc = mixture.critical_volume_m3_mol

    # pseudo-critical (mole-fraction-weighted) mixture properties
    mw_mix = (composition * mw).sum(dim=-1)
    P_pc = (composition * pc).sum(dim=-1)
    T_pc = (composition * Tc).sum(dim=-1)
    V_pc = (composition * Vc).sum(dim=-1)  # pseudo-critical molar volume

    # dilute-gas (atmospheric) mixture viscosity μ*: Stiel-Thodos per component.
    # A per-cell T broadcasts against the per-component axis → ``(..., n_c)``; a scalar
    # T collapses to ``(n_c,)`` (the isothermal path, numerically unchanged).
    Tr = (T.unsqueeze(-1) if T.ndim > 0 else T) / Tc  # (..., n_c) or (n_c,)
    e_i = _viscosity_parameter(mw, Tc, pc)  # (n_c,)
    mu_i = torch.where(
        Tr > 1.5, 17.78e-5 * (4.58 * Tr - 1.67).clamp_min(0.0).pow(0.625), 34e-5 * Tr.pow(0.94)
    )  # (n_c,)
    tmp = torch.sqrt(1000.0 * mw) * composition  # (..., n_c)
    mu_atm = (tmp * (mu_i / e_i)).sum(dim=-1) / tmp.sum(dim=-1)  # (...)

    e_mix = _viscosity_parameter(mw_mix, T_pc, P_pc)  # (...)
    V = _R_GAS * T * Z / p  # phase molar volume (...)
    rho_r = V_pc / V  # reduced density (...)
    corr = torch.zeros_like(rho_r)
    for k, c in enumerate(_LBC_COEFF):
        corr = corr + c * rho_r.pow(k)
    mu_correction = (corr.pow(4) + _LBC_SHIFT) / e_mix
    return 1e-3 * (mu_atm + mu_correction)  # cP → Pa·s


__all__ = ["lbc_viscosity"]
