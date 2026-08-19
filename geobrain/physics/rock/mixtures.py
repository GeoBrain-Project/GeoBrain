"""Canonical two-phase solid and fluid mixture equations.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TypeAlias, cast

import torch

from geobrain.core import ErrorCode

from .contracts import require_compatible_tensors
from .errors import RockContractError

TensorInput: TypeAlias = torch.Tensor | int | float


def _validated_inputs(
    object_name: str,
    *fields: tuple[str, TensorInput],
) -> tuple[torch.Tensor, ...]:
    tensors = cast(
        tuple[torch.Tensor, ...], require_compatible_tensors(object_name, *fields)
    )
    for (name, _), tensor in zip(fields, tensors):
        if tensor.layout is not torch.strided:
            raise RockContractError(
                "Rock mixture kernels require strided tensors",
                object_name=object_name,
                field=name,
                expected="torch.strided layout",
                actual=str(tensor.layout),
            )
        if tensor.device.type == "meta":
            raise RockContractError(
                "Rock mixture kernels require materialized values",
                object_name=object_name,
                field=name,
                expected="a materialized CPU or accelerator tensor",
                actual=str(tensor.device),
                code=ErrorCode.DEVICE_UNAVAILABLE,
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RockContractError(
                "Rock mixture input must be finite",
                object_name=object_name,
                field=name,
                expected="finite values",
                actual="non-finite value(s)",
            )
    return tensors


def _require_positive(
    object_name: str,
    field: str,
    value: torch.Tensor,
) -> None:
    if bool(torch.any(value <= 0.0)):
        raise RockContractError(
            "Rock mixture input must be positive",
            object_name=object_name,
            field=field,
            expected="> 0",
            actual={"minimum": value.amin().item(), "maximum": value.amax().item()},
        )


def _require_unit_interval(
    object_name: str,
    field: str,
    value: torch.Tensor,
) -> None:
    if bool(torch.any((value < 0.0) | (value > 1.0))):
        raise RockContractError(
            "Rock mixture fraction must lie in the unit interval",
            object_name=object_name,
            field=field,
            expected="0 <= value <= 1",
            actual={"minimum": value.amin().item(), "maximum": value.amax().item()},
            hint="supply physical fractions without clamping or implicit normalization",
        )


def _two_phase_inputs(
    object_name: str,
    modulus_1: torch.Tensor,
    modulus_2: TensorInput,
    phase_1_fraction: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    modulus_1, modulus_2, phase_1_fraction = _validated_inputs(
        object_name,
        ("modulus_1", modulus_1),
        ("modulus_2", modulus_2),
        ("phase_1_fraction", phase_1_fraction),
    )
    _require_positive(object_name, "modulus_1", modulus_1)
    _require_positive(object_name, "modulus_2", modulus_2)
    _require_unit_interval(object_name, "phase_1_fraction", phase_1_fraction)
    return modulus_1, modulus_2, phase_1_fraction


def voigt_average(
    modulus_1: torch.Tensor,
    modulus_2: TensorInput,
    phase_1_fraction: TensorInput,
) -> torch.Tensor:
    """Return the two-phase Voigt (iso-strain) average."""

    modulus_1, modulus_2, phase_1_fraction = _two_phase_inputs(
        "voigt_average", modulus_1, modulus_2, phase_1_fraction
    )
    phase_2_fraction = phase_1_fraction.new_tensor(1.0) - phase_1_fraction
    return phase_1_fraction * modulus_1 + phase_2_fraction * modulus_2


def reuss_average(
    modulus_1: torch.Tensor,
    modulus_2: TensorInput,
    phase_1_fraction: TensorInput,
) -> torch.Tensor:
    """Return the two-phase Reuss (iso-stress) average."""

    modulus_1, modulus_2, phase_1_fraction = _two_phase_inputs(
        "reuss_average", modulus_1, modulus_2, phase_1_fraction
    )
    phase_2_fraction = phase_1_fraction.new_tensor(1.0) - phase_1_fraction
    reciprocal = phase_1_fraction / modulus_1 + phase_2_fraction / modulus_2
    return reciprocal.reciprocal()


def vrh_average(
    modulus_1: torch.Tensor,
    modulus_2: TensorInput,
    phase_1_fraction: TensorInput,
) -> torch.Tensor:
    """Return the Hill mean of the canonical Voigt and Reuss averages."""

    modulus_1, modulus_2, phase_1_fraction = _two_phase_inputs(
        "vrh_average", modulus_1, modulus_2, phase_1_fraction
    )
    voigt = voigt_average(modulus_1, modulus_2, phase_1_fraction)
    reuss = reuss_average(modulus_1, modulus_2, phase_1_fraction)
    return (voigt + reuss) * modulus_1.new_tensor(0.5)


def hashin_shtrikman_bounds(
    bulk_modulus_1: torch.Tensor,
    shear_modulus_1: TensorInput,
    bulk_modulus_2: TensorInput,
    shear_modulus_2: TensorInput,
    phase_1_fraction: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ordered bulk-lower, bulk-upper, shear-lower, shear-upper bounds."""

    (
        bulk_modulus_1,
        shear_modulus_1,
        bulk_modulus_2,
        shear_modulus_2,
        phase_1_fraction,
    ) = _validated_inputs(
        "hashin_shtrikman_bounds",
        ("bulk_modulus_1", bulk_modulus_1),
        ("shear_modulus_1", shear_modulus_1),
        ("bulk_modulus_2", bulk_modulus_2),
        ("shear_modulus_2", shear_modulus_2),
        ("phase_1_fraction", phase_1_fraction),
    )
    for name, value in (
        ("bulk_modulus_1", bulk_modulus_1),
        ("shear_modulus_1", shear_modulus_1),
        ("bulk_modulus_2", bulk_modulus_2),
        ("shear_modulus_2", shear_modulus_2),
    ):
        _require_positive("hashin_shtrikman_bounds", name, value)
    _require_unit_interval(
        "hashin_shtrikman_bounds", "phase_1_fraction", phase_1_fraction
    )
    (
        bulk_modulus_1,
        shear_modulus_1,
        bulk_modulus_2,
        shear_modulus_2,
        phase_1_fraction,
    ) = torch.broadcast_tensors(
        bulk_modulus_1,
        shear_modulus_1,
        bulk_modulus_2,
        shear_modulus_2,
        phase_1_fraction,
    )
    crossed_order = (bulk_modulus_1 - bulk_modulus_2) * (
        shear_modulus_1 - shear_modulus_2
    ) < 0.0
    if bool(torch.any(crossed_order)):
        raise RockContractError(
            "Hashin-Shtrikman phases have crossed stiffness ordering",
            object_name="hashin_shtrikman_bounds",
            field="phase_moduli",
            expected="one phase is no softer in both bulk and shear modulus",
            actual="crossed bulk/shear ordering",
        )

    phase_1_is_stiff = (bulk_modulus_1 > bulk_modulus_2) | (
        (bulk_modulus_1 == bulk_modulus_2)
        & (shear_modulus_1 >= shear_modulus_2)
    )
    bulk_stiff = torch.where(phase_1_is_stiff, bulk_modulus_1, bulk_modulus_2)
    shear_stiff = torch.where(phase_1_is_stiff, shear_modulus_1, shear_modulus_2)
    bulk_soft = torch.where(phase_1_is_stiff, bulk_modulus_2, bulk_modulus_1)
    shear_soft = torch.where(phase_1_is_stiff, shear_modulus_2, shear_modulus_1)
    fraction_stiff = torch.where(
        phase_1_is_stiff,
        phase_1_fraction,
        phase_1_fraction.new_tensor(1.0) - phase_1_fraction,
    )
    fraction_soft = fraction_stiff.new_tensor(1.0) - fraction_stiff

    bulk_delta = bulk_stiff - bulk_soft
    shear_delta = shear_stiff - shear_soft
    p_wave_soft = bulk_soft + bulk_soft.new_tensor(4.0 / 3.0) * shear_soft
    p_wave_stiff = bulk_stiff + bulk_stiff.new_tensor(4.0 / 3.0) * shear_stiff
    bulk_lower = bulk_soft + (
        fraction_stiff * bulk_delta * p_wave_soft
        / (p_wave_soft + fraction_soft * bulk_delta)
    )
    bulk_upper = bulk_stiff - (
        fraction_soft * bulk_delta * p_wave_stiff
        / (p_wave_stiff - fraction_stiff * bulk_delta)
    )

    shear_factor_soft = (
        2.0
        * fraction_soft
        * (bulk_soft + 2.0 * shear_soft)
        / (5.0 * shear_soft * p_wave_soft)
    )
    shear_factor_stiff = (
        2.0
        * fraction_stiff
        * (bulk_stiff + 2.0 * shear_stiff)
        / (5.0 * shear_stiff * p_wave_stiff)
    )
    shear_lower = shear_soft + (
        fraction_stiff * shear_delta
        / (1.0 + shear_factor_soft * shear_delta)
    )
    shear_upper = shear_stiff - (
        fraction_soft * shear_delta
        / (1.0 - shear_factor_stiff * shear_delta)
    )
    return bulk_lower, bulk_upper, shear_lower, shear_upper


def wood_fluid_mix(
    bulk_modulus_water: torch.Tensor,
    bulk_modulus_other: TensorInput,
    rho_water: TensorInput,
    rho_other: TensorInput,
    water_saturation: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Wood bulk modulus and volume-weighted density for two fluids.

    Args:
        bulk_modulus_water: water bulk modulus [Pa].
        bulk_modulus_other: second-fluid bulk modulus [Pa].
        rho_water: water density [kg/m^3].
        rho_other: second-fluid density [kg/m^3].
        water_saturation: water volume fraction of the pore fill.
    """

    (
        bulk_modulus_water,
        bulk_modulus_other,
        rho_water,
        rho_other,
        water_saturation,
    ) = _validated_inputs(
        "wood_fluid_mix",
        ("bulk_modulus_water", bulk_modulus_water),
        ("bulk_modulus_other", bulk_modulus_other),
        ("rho_water", rho_water),
        ("rho_other", rho_other),
        ("water_saturation", water_saturation),
    )
    for name, value in (
        ("bulk_modulus_water", bulk_modulus_water),
        ("bulk_modulus_other", bulk_modulus_other),
        ("rho_water", rho_water),
        ("rho_other", rho_other),
    ):
        _require_positive("wood_fluid_mix", name, value)
    _require_unit_interval("wood_fluid_mix", "water_saturation", water_saturation)
    other_saturation = water_saturation.new_tensor(1.0) - water_saturation
    reciprocal_bulk_modulus = (
        water_saturation / bulk_modulus_water
        + other_saturation / bulk_modulus_other
    )
    density = water_saturation * rho_water + other_saturation * rho_other
    return reciprocal_bulk_modulus.reciprocal(), density


def brie_fluid_mix(
    bulk_modulus_water: torch.Tensor,
    bulk_modulus_other: TensorInput,
    rho_water: TensorInput,
    rho_other: TensorInput,
    water_saturation: TensorInput,
    *,
    exponent: TensorInput = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Brie bulk modulus and volume-weighted density for two fluids."""

    (
        bulk_modulus_water,
        bulk_modulus_other,
        rho_water,
        rho_other,
        water_saturation,
        exponent,
    ) = _validated_inputs(
        "brie_fluid_mix",
        ("bulk_modulus_water", bulk_modulus_water),
        ("bulk_modulus_other", bulk_modulus_other),
        ("rho_water", rho_water),
        ("rho_other", rho_other),
        ("water_saturation", water_saturation),
        ("exponent", exponent),
    )
    for name, value in (
        ("bulk_modulus_water", bulk_modulus_water),
        ("bulk_modulus_other", bulk_modulus_other),
        ("rho_water", rho_water),
        ("rho_other", rho_other),
        ("exponent", exponent),
    ):
        _require_positive("brie_fluid_mix", name, value)
    _require_unit_interval("brie_fluid_mix", "water_saturation", water_saturation)
    other_saturation = water_saturation.new_tensor(1.0) - water_saturation
    bulk_modulus = bulk_modulus_other + (
        bulk_modulus_water - bulk_modulus_other
    ) * water_saturation.pow(exponent)
    density = water_saturation * rho_water + other_saturation * rho_other
    return bulk_modulus, density


__all__ = [
    "brie_fluid_mix",
    "hashin_shtrikman_bounds",
    "reuss_average",
    "voigt_average",
    "vrh_average",
    "wood_fluid_mix",
]
