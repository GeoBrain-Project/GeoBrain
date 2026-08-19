"""Canonical SI Batzle-Wang, live-oil, and CO₂ fluid equations.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeAlias, cast

import torch

from geobrain.core import ErrorCode

from .contracts import require_compatible_tensors
from .errors import RockContractError, RockNumericsError
from .mixtures import brie_fluid_mix, wood_fluid_mix

TensorInput: TypeAlias = torch.Tensor | int | float


@dataclass(frozen=True, slots=True)
class FluidProperties:
    """Fluid bulk modulus in Pa and density in kg/m³."""

    bulk_modulus: torch.Tensor
    rho: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.bulk_modulus
        yield self.rho


def _validated_inputs(
    object_name: str,
    *fields: tuple[str, TensorInput],
) -> tuple[torch.Tensor, ...]:
    first_name, first_value = fields[0]
    if not isinstance(first_value, torch.Tensor):
        raise RockContractError(
            "Rock fluid-EOS reference input must be a tensor",
            object_name=object_name,
            field=first_name,
            expected="torch.Tensor reference with dtype float32 or float64",
            actual={"type": type(first_value).__qualname__, "unit": "K"},
            hint="pass temperature as the first materialized SI tensor",
        )
    tensors = cast(tuple[torch.Tensor, ...], require_compatible_tensors(object_name, *fields))
    for (name, _), tensor in zip(fields, tensors):
        if tensor.layout is not torch.strided:
            raise RockContractError(
                "Rock fluid-EOS kernels require strided tensors",
                object_name=object_name,
                field=name,
                expected="torch.strided layout",
                actual={"layout": str(tensor.layout), "unit": "SI"},
            )
        if tensor.device.type == "meta":
            raise RockContractError(
                "Rock fluid-EOS kernels require materialized values",
                object_name=object_name,
                field=name,
                expected="a materialized CPU or accelerator tensor",
                actual={"device": str(tensor.device), "unit": "SI"},
                code=ErrorCode.DEVICE_UNAVAILABLE,
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RockContractError(
                "Rock fluid-EOS input must be finite",
                object_name=object_name,
                field=name,
                expected="finite SI values",
                actual={"value": "non-finite value(s)", "unit": "SI"},
            )
    return tensors


def _extrema(value: torch.Tensor, unit: str) -> dict[str, object]:
    return {
        "minimum": value.amin().item(),
        "maximum": value.amax().item(),
        "unit": unit,
    }


def _require_range(
    object_name: str,
    field: str,
    value: torch.Tensor,
    lower: float,
    upper: float,
    unit: str,
    *,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> None:
    below = value < lower if lower_inclusive else value <= lower
    above = value > upper if upper_inclusive else value >= upper
    if bool(torch.any(below | above)):
        left = "[" if lower_inclusive else "("
        right = "]" if upper_inclusive else ")"
        raise RockContractError(
            "fluid EOS input lies outside its cited calibration range",
            object_name=object_name,
            field=field,
            expected=f"{left}{lower}, {upper}{right} {unit}",
            actual=_extrema(value, unit),
            hint="supply an unclamped value inside the cited Batzle-Wang range",
        )


def _require_common_domain(
    object_name: str,
    temperature: torch.Tensor,
    pressure: torch.Tensor,
    *,
    maximum_temperature: float,
) -> None:
    _require_range(
        object_name,
        "temperature",
        temperature,
        273.15,
        maximum_temperature,
        "K",
    )
    _require_range(
        object_name,
        "pressure",
        pressure,
        0.1e6,
        100.0e6,
        "Pa",
    )


def _require_positive_result(
    object_name: str,
    field: str,
    value: torch.Tensor,
    unit: str,
) -> None:
    invalid = (~torch.isfinite(value)) | (value <= 0.0)
    if bool(torch.any(invalid)):
        raise RockNumericsError(
            "fluid EOS produced a non-positive or non-finite result",
            object_name=object_name,
            field=field,
            expected=f"finite value > 0 {unit}",
            actual={"value": "invalid EOS result", "unit": unit},
            hint="choose inputs away from the edge of the calibrated domain",
        )


_BW_WATER_VELOCITY = (
    (1402.85, 1.524, 3.437e-3, -1.197e-5),
    (4.871, -0.0111, 1.739e-4, -1.628e-6),
    (-0.04783, 2.747e-4, -2.135e-6, 1.237e-8),
    (1.487e-4, -6.503e-7, -1.455e-8, 1.327e-10),
    (-2.197e-7, 7.987e-10, 5.230e-11, -4.614e-13),
)


def _water_velocity(
    temperature_celsius: torch.Tensor,
    pressure_megapascal: torch.Tensor,
) -> torch.Tensor:
    velocity = torch.zeros_like(temperature_celsius)
    for temperature_power, row in enumerate(_BW_WATER_VELOCITY):
        powered_temperature = temperature_celsius.pow(temperature_power)
        for pressure_power, coefficient in enumerate(row):
            velocity = velocity + (
                coefficient * powered_temperature * pressure_megapascal.pow(pressure_power)
            )
    return velocity


def _water_density_native(
    temperature_celsius: torch.Tensor,
    pressure_megapascal: torch.Tensor,
) -> torch.Tensor:
    temperature = temperature_celsius
    pressure = pressure_megapascal
    return 1.0 + 1.0e-6 * (
        -80.0 * temperature
        - 3.3 * temperature.square()
        + 0.00175 * temperature.pow(3)
        + 489.0 * pressure
        - 2.0 * temperature * pressure
        + 0.016 * temperature.square() * pressure
        - 1.3e-5 * temperature.pow(3) * pressure
        - 0.333 * pressure.square()
        - 0.002 * temperature * pressure.square()
    )


def _brine_properties(
    temperature: torch.Tensor,
    pressure: torch.Tensor,
    salinity: torch.Tensor,
) -> FluidProperties:
    temperature_celsius = temperature - temperature.new_tensor(273.15)
    pressure_megapascal = pressure * pressure.new_tensor(1.0e-6)
    density_water_g_per_cm3 = _water_density_native(temperature_celsius, pressure_megapascal)
    density_brine_g_per_cm3 = density_water_g_per_cm3 + salinity * (
        0.668
        + 0.44 * salinity
        + 1.0e-6
        * (
            300.0 * pressure_megapascal
            - 2400.0 * pressure_megapascal * salinity
            + temperature_celsius
            * (
                80.0
                + 3.0 * temperature_celsius
                - 3300.0 * salinity
                - 13.0 * pressure_megapascal
                + 47.0 * pressure_megapascal * salinity
            )
        )
    )
    velocity_water = _water_velocity(temperature_celsius, pressure_megapascal)
    velocity_brine = (
        velocity_water
        + salinity
        * (
            1170.0
            - 9.6 * temperature_celsius
            + 0.055 * temperature_celsius.square()
            - 8.5e-5 * temperature_celsius.pow(3)
            + 2.6 * pressure_megapascal
            - 0.0029 * temperature_celsius * pressure_megapascal
            - 0.0476 * pressure_megapascal.square()
        )
        + salinity.pow(1.5)
        * (780.0 - 10.0 * pressure_megapascal + 0.16 * pressure_megapascal.square())
        - 1820.0 * salinity.square()
    )
    density = density_brine_g_per_cm3 * density_brine_g_per_cm3.new_tensor(1000.0)
    bulk_modulus = density * velocity_brine.square()
    return FluidProperties(bulk_modulus=bulk_modulus, rho=density)


def batzle_wang_brine(
    temperature: torch.Tensor,
    pressure: TensorInput,
    salinity: TensorInput = 0.035,
) -> FluidProperties:
    """Return brine properties from Batzle & Wang (1992), Table 1.

    Public inputs are temperature in K, pressure in Pa, and NaCl mass fraction.
    The cited polynomial is calibrated here for 273.15–623.15 K,
    0.1–100 MPa, and salinity 0–0.35.

    Args:
        temperature: temperature [degC].
        pressure: pore pressure [Pa].
        salinity: NaCl weight fraction.
    """

    temperature, pressure, salinity = _validated_inputs(
        "batzle_wang_brine",
        ("temperature", temperature),
        ("pressure", pressure),
        ("salinity", salinity),
    )
    temperature, pressure, salinity = torch.broadcast_tensors(temperature, pressure, salinity)
    _require_common_domain(
        "batzle_wang_brine",
        temperature,
        pressure,
        maximum_temperature=623.15,
    )
    _require_range("batzle_wang_brine", "salinity", salinity, 0.0, 0.35, "1")
    result = _brine_properties(temperature, pressure, salinity)
    _require_positive_result("batzle_wang_brine", "bulk_modulus", result.bulk_modulus, "Pa")
    _require_positive_result("batzle_wang_brine", "rho", result.rho, "kg/m^3")
    return result


def _gas_properties(
    temperature: torch.Tensor,
    pressure: torch.Tensor,
    gas_gravity: torch.Tensor,
) -> FluidProperties:
    temperature_celsius = temperature - temperature.new_tensor(273.15)
    pressure_megapascal = pressure * pressure.new_tensor(1.0e-6)
    absolute_temperature = temperature_celsius + temperature.new_tensor(273.15)
    pseudo_critical_temperature = 94.72 + 170.75 * gas_gravity
    pseudo_critical_pressure_megapascal = 4.892 - 0.4048 * gas_gravity
    reduced_temperature = absolute_temperature / pseudo_critical_temperature
    reduced_pressure = pressure_megapascal / pseudo_critical_pressure_megapascal

    coefficient_a = 0.03 + 0.00527 * (3.5 - reduced_temperature).pow(3)
    coefficient_b = 0.642 * reduced_temperature - 0.007 * reduced_temperature.pow(4) - 0.52
    alpha = 0.45 + 8.0 * (0.56 - reduced_temperature.reciprocal()).square()
    beta = alpha / reduced_temperature
    coefficient_c = 0.109 * (3.85 - reduced_temperature).square()
    exponential = coefficient_c * torch.exp(-beta * reduced_pressure.pow(1.2))
    compressibility_factor = coefficient_a * reduced_pressure + coefficient_b + exponential
    derivative_exponential = -1.2 * exponential * beta * reduced_pressure.pow(0.2)
    derivative_z = coefficient_a + derivative_exponential
    adiabatic_index = (
        0.85
        + 5.6 / (reduced_pressure + 2.0)
        + 27.1 / (reduced_pressure + 3.5).square()
        - 8.7 * torch.exp(-0.65 * (reduced_pressure + 1.0))
    )
    compressibility_denominator = 1.0 - reduced_pressure * derivative_z / compressibility_factor
    _require_positive_result(
        "batzle_wang_gas",
        "compressibility_factor",
        compressibility_factor,
        "1",
    )
    _require_positive_result(
        "batzle_wang_gas",
        "compressibility_denominator",
        compressibility_denominator,
        "1",
    )
    molar_mass_air = temperature.new_tensor(28.97e-3)
    gas_constant = temperature.new_tensor(8.314)
    density = (
        molar_mass_air
        * gas_gravity
        * pressure
        / (compressibility_factor * gas_constant * absolute_temperature)
    )
    bulk_modulus = adiabatic_index * pressure / compressibility_denominator
    return FluidProperties(bulk_modulus=bulk_modulus, rho=density)


def batzle_wang_gas(
    temperature: torch.Tensor,
    pressure: TensorInput,
    gas_gravity: TensorInput = 0.65,
) -> FluidProperties:
    """Return hydrocarbon-gas properties using Mavko (2009), eqs. 7.30–7.37."""

    temperature, pressure, gas_gravity = _validated_inputs(
        "batzle_wang_gas",
        ("temperature", temperature),
        ("pressure", pressure),
        ("gas_gravity", gas_gravity),
    )
    temperature, pressure, gas_gravity = torch.broadcast_tensors(temperature, pressure, gas_gravity)
    _require_common_domain("batzle_wang_gas", temperature, pressure, maximum_temperature=473.15)
    _require_range("batzle_wang_gas", "gas_gravity", gas_gravity, 0.55, 1.5, "1")
    result = _gas_properties(temperature, pressure, gas_gravity)
    _require_positive_result("batzle_wang_gas", "bulk_modulus", result.bulk_modulus, "Pa")
    _require_positive_result("batzle_wang_gas", "rho", result.rho, "kg/m^3")
    return result


def _dead_oil_properties(
    temperature: torch.Tensor,
    pressure: torch.Tensor,
    reference_density: torch.Tensor,
) -> FluidProperties:
    temperature_celsius = temperature - temperature.new_tensor(273.15)
    pressure_megapascal = pressure * pressure.new_tensor(1.0e-6)
    reference_density_g_per_cm3 = reference_density * reference_density.new_tensor(1.0e-3)
    density_at_pressure = (
        reference_density_g_per_cm3
        + (0.00277 * pressure_megapascal - 1.71e-7 * pressure_megapascal.pow(3))
        * (reference_density_g_per_cm3 - 1.15).square()
        + 3.49e-4 * pressure_megapascal
    )
    density_g_per_cm3 = density_at_pressure / (
        0.972 + 3.81e-4 * (temperature_celsius + 17.78).pow(1.175)
    )
    velocity = (
        2096.0 * torch.sqrt(reference_density_g_per_cm3 / (2.6 - reference_density_g_per_cm3))
        - 3.7 * temperature_celsius
        + 4.64 * pressure_megapascal
        + 0.0115
        * (4.12 * torch.sqrt(1.08 / reference_density_g_per_cm3 - 1.0) - 1.0)
        * temperature_celsius
        * pressure_megapascal
    )
    density = density_g_per_cm3 * density_g_per_cm3.new_tensor(1000.0)
    return FluidProperties(bulk_modulus=density * velocity.square(), rho=density)


def batzle_wang_dead_oil(
    temperature: torch.Tensor,
    pressure: TensorInput,
    reference_density: TensorInput = 850.0,
) -> FluidProperties:
    """Return dead-oil properties using Mavko (2009), eqs. 7.41–7.42.

    ``reference_density`` is public SI kg/m³; conversion to the correlation's
    native g/cm³ occurs only in the private calculation.  # SI-EXEMPT
    """

    temperature, pressure, reference_density = _validated_inputs(
        "batzle_wang_dead_oil",
        ("temperature", temperature),
        ("pressure", pressure),
        ("reference_density", reference_density),
    )
    temperature, pressure, reference_density = torch.broadcast_tensors(
        temperature, pressure, reference_density
    )
    _require_common_domain(
        "batzle_wang_dead_oil",
        temperature,
        pressure,
        maximum_temperature=473.15,
    )
    _require_range(
        "batzle_wang_dead_oil",
        "reference_density",
        reference_density,
        600.0,
        1_000.0,
        "kg/m^3",
        lower_inclusive=False,
    )
    result = _dead_oil_properties(temperature, pressure, reference_density)
    _require_positive_result("batzle_wang_dead_oil", "bulk_modulus", result.bulk_modulus, "Pa")
    _require_positive_result("batzle_wang_dead_oil", "rho", result.rho, "kg/m^3")
    return result


def _live_oil_properties(
    temperature: torch.Tensor,
    pressure: torch.Tensor,
    reference_density: torch.Tensor,
    gas_oil_ratio: torch.Tensor,
    gas_gravity: torch.Tensor,
) -> FluidProperties:
    temperature_celsius = temperature - temperature.new_tensor(273.15)
    pressure_megapascal = pressure * pressure.new_tensor(1.0e-6)
    reference_density_g_per_cm3 = reference_density * reference_density.new_tensor(1.0e-3)
    formation_volume_factor = 0.972 + 3.8e-4 * (
        2.4 * gas_oil_ratio * torch.sqrt(gas_gravity / reference_density_g_per_cm3)
        + temperature_celsius
        + 17.8
    ).pow(1.175)
    reservoir_reference_density = (
        reference_density_g_per_cm3 + 0.0012 * gas_oil_ratio * gas_gravity
    ) / formation_volume_factor
    pseudo_density = (
        reference_density_g_per_cm3 / formation_volume_factor / (1.0 + 0.001 * gas_oil_ratio)
    )
    density_at_pressure = (
        reservoir_reference_density
        + (0.00277 * pressure_megapascal - 1.71e-7 * pressure_megapascal.pow(3))
        * (reservoir_reference_density - 1.15).square()
        + 3.49e-4 * pressure_megapascal
    )
    density_g_per_cm3 = density_at_pressure
    velocity_base = pseudo_density / (2.6 - pseudo_density)
    velocity_correction_base = 1.08 / pseudo_density - 1.0
    _require_positive_result("batzle_wang_live_oil", "velocity_base", velocity_base, "1")
    _require_positive_result(
        "batzle_wang_live_oil",
        "velocity_correction_base",
        velocity_correction_base,
        "1",
    )
    velocity = (
        2096.0 * torch.sqrt(velocity_base)
        - 3.7 * temperature_celsius
        + 4.64 * pressure_megapascal
        + 0.0115
        * (4.12 * torch.sqrt(velocity_correction_base) - 1.0)
        * temperature_celsius
        * pressure_megapascal
    )
    density = density_g_per_cm3 * density_g_per_cm3.new_tensor(1000.0)
    return FluidProperties(bulk_modulus=density * velocity.square(), rho=density)


def batzle_wang_live_oil(
    temperature: torch.Tensor,
    pressure: TensorInput,
    reference_density: TensorInput = 850.0,
    gas_oil_ratio: TensorInput = 85.0,
    gas_gravity: TensorInput = 0.65,
) -> FluidProperties:
    """Return live-oil properties using Mavko (2009), eqs. 7.43–7.47."""

    values = _validated_inputs(
        "batzle_wang_live_oil",
        ("temperature", temperature),
        ("pressure", pressure),
        ("reference_density", reference_density),
        ("gas_oil_ratio", gas_oil_ratio),
        ("gas_gravity", gas_gravity),
    )
    (
        temperature,
        pressure,
        reference_density,
        gas_oil_ratio,
        gas_gravity,
    ) = torch.broadcast_tensors(*values)
    _require_common_domain(
        "batzle_wang_live_oil",
        temperature,
        pressure,
        maximum_temperature=473.15,
    )
    _require_range(
        "batzle_wang_live_oil",
        "reference_density",
        reference_density,
        600.0,
        1_100.0,
        "kg/m^3",
        lower_inclusive=False,
        upper_inclusive=False,
    )
    _require_range(
        "batzle_wang_live_oil",
        "gas_oil_ratio",
        gas_oil_ratio,
        0.0,
        500.0,
        "1",
    )
    _require_range(
        "batzle_wang_live_oil",
        "gas_gravity",
        gas_gravity,
        0.55,
        1.5,
        "1",
    )
    result = _live_oil_properties(
        temperature,
        pressure,
        reference_density,
        gas_oil_ratio,
        gas_gravity,
    )
    _require_positive_result("batzle_wang_live_oil", "bulk_modulus", result.bulk_modulus, "Pa")
    _require_positive_result("batzle_wang_live_oil", "rho", result.rho, "kg/m^3")
    return result


def solution_gas_oil_ratio(
    temperature: torch.Tensor,
    pressure: TensorInput,
    reference_density: TensorInput = 850.0,
    gas_gravity: TensorInput = 0.65,
) -> torch.Tensor:
    """Return the Batzle-Wang solution gas/oil ratio for a saturated oil."""

    values = _validated_inputs(
        "solution_gas_oil_ratio",
        ("temperature", temperature),
        ("pressure", pressure),
        ("reference_density", reference_density),
        ("gas_gravity", gas_gravity),
    )
    temperature, pressure, reference_density, gas_gravity = torch.broadcast_tensors(*values)
    _require_common_domain(
        "solution_gas_oil_ratio",
        temperature,
        pressure,
        maximum_temperature=473.15,
    )
    _require_range(
        "solution_gas_oil_ratio",
        "reference_density",
        reference_density,
        600.0,
        1_100.0,
        "kg/m^3",
        lower_inclusive=False,
        upper_inclusive=False,
    )
    _require_range(
        "solution_gas_oil_ratio",
        "gas_gravity",
        gas_gravity,
        0.55,
        1.5,
        "1",
    )
    temperature_celsius = temperature - temperature.new_tensor(273.15)
    pressure_megapascal = pressure * pressure.new_tensor(1.0e-6)
    reference_density_g_per_cm3 = reference_density * reference_density.new_tensor(1.0e-3)
    ratio = (
        0.02123
        * gas_gravity
        * (
            pressure_megapascal
            * torch.exp(4.072 / reference_density_g_per_cm3 - 0.00377 * temperature_celsius)
        ).pow(1.205)
    )
    if bool(torch.any(ratio > 500.0)):
        raise RockNumericsError(
            "computed solution gas/oil ratio exceeds live-oil calibration",
            object_name="solution_gas_oil_ratio",
            field="gas_oil_ratio",
            expected="0 <= gas_oil_ratio <= 500 (dimensionless)",
            actual=_extrema(ratio, "1"),
        )
    return ratio


def live_oil_properties(
    temperature: torch.Tensor,
    pressure: TensorInput,
    reference_density: TensorInput = 850.0,
    gas_gravity: TensorInput = 0.65,
    gas_oil_ratio: TensorInput | None = None,
) -> FluidProperties:
    """Return live oil, deriving solution gas content when it is omitted."""

    resolved_ratio = (
        solution_gas_oil_ratio(temperature, pressure, reference_density, gas_gravity)
        if gas_oil_ratio is None
        else gas_oil_ratio
    )
    return batzle_wang_live_oil(
        temperature,
        pressure,
        reference_density,
        resolved_ratio,
        gas_gravity,
    )


def _co2_properties(
    temperature: torch.Tensor,
    pressure: torch.Tensor,
    gas_gravity: torch.Tensor,
) -> FluidProperties:
    pressure_megapascal = pressure * pressure.new_tensor(1.0e-6)
    reduced_pressure = pressure_megapascal / pressure.new_tensor(7.4)
    reduced_temperature = temperature / temperature.new_tensor(304.6)
    alpha = 0.45 + 8.0 * (0.56 - reduced_temperature.reciprocal()).square()
    exponential = (
        0.109
        * (3.85 - reduced_temperature).square()
        * torch.exp(-alpha * reduced_pressure.pow(1.2) / reduced_temperature)
    )
    coefficient_a = 0.03 + 0.00527 * (3.5 - reduced_temperature).pow(3)
    compressibility_factor = (
        coefficient_a * reduced_pressure
        + 0.642 * reduced_temperature
        - 0.007 * reduced_temperature.pow(4)
        - 0.52
        + exponential
    )
    derivative_z = coefficient_a + (
        0.109
        * (3.85 - reduced_temperature).square()
        * 1.2
        * reduced_pressure.pow(0.2)
        * (-alpha / reduced_temperature)
        * torch.exp(-alpha * reduced_pressure.pow(1.2) / reduced_temperature)
    )
    adiabatic_index = (
        0.85
        + 5.6 / (reduced_pressure + 2.0)
        + 27.1 / (reduced_pressure + 3.5).square()
        - 8.7 * torch.exp(-0.65 * (reduced_pressure + 1.0))
    )
    denominator = 1.0 - reduced_pressure * derivative_z / compressibility_factor
    _require_positive_result(
        "co2_properties", "compressibility_factor", compressibility_factor, "1"
    )
    _require_positive_result("co2_properties", "compressibility_denominator", denominator, "1")
    density_g_per_cm3 = (
        28.8 * gas_gravity * pressure_megapascal / (compressibility_factor * 8.3145 * temperature)
    )
    density = density_g_per_cm3 * density_g_per_cm3.new_tensor(1000.0)
    bulk_modulus = pressure * adiabatic_index / denominator
    return FluidProperties(bulk_modulus=bulk_modulus, rho=density)


def co2_properties(
    temperature: torch.Tensor,
    pressure: TensorInput,
    gas_gravity: TensorInput = 1.5349,
) -> FluidProperties:
    """Return supercritical CO₂ properties from the modified BW correlation."""

    temperature, pressure, gas_gravity = _validated_inputs(
        "co2_properties",
        ("temperature", temperature),
        ("pressure", pressure),
        ("gas_gravity", gas_gravity),
    )
    temperature, pressure, gas_gravity = torch.broadcast_tensors(temperature, pressure, gas_gravity)
    _require_range("co2_properties", "temperature", temperature, 304.15, 473.15, "K")
    _require_range("co2_properties", "pressure", pressure, 7.4e6, 100.0e6, "Pa")
    _require_range("co2_properties", "gas_gravity", gas_gravity, 1.5, 1.6, "1")
    result = _co2_properties(temperature, pressure, gas_gravity)
    _require_positive_result("co2_properties", "bulk_modulus", result.bulk_modulus, "Pa")
    _require_positive_result("co2_properties", "rho", result.rho, "kg/m^3")
    return result


def co2_brine_mix(
    temperature: torch.Tensor,
    pressure: TensorInput,
    salinity: TensorInput,
    co2_saturation: TensorInput,
    *,
    brie_exponent: TensorInput | None = None,
) -> FluidProperties:
    """Mix canonical CO₂ and brine with Wood or an explicit Brie exponent."""

    fields: list[tuple[str, TensorInput]] = [
        ("temperature", temperature),
        ("pressure", pressure),
        ("salinity", salinity),
        ("co2_saturation", co2_saturation),
    ]
    if brie_exponent is not None:
        fields.append(("brie_exponent", brie_exponent))
    values = _validated_inputs("co2_brine_mix", *fields)
    values = torch.broadcast_tensors(*values)
    temperature, pressure, salinity, co2_saturation = values[:4]
    _require_range("co2_brine_mix", "temperature", temperature, 304.15, 473.15, "K")
    _require_range("co2_brine_mix", "pressure", pressure, 7.4e6, 100.0e6, "Pa")
    _require_range("co2_brine_mix", "salinity", salinity, 0.0, 0.35, "1")
    _require_range("co2_brine_mix", "co2_saturation", co2_saturation, 0.0, 1.0, "1")
    if brie_exponent is not None:
        exponent = values[4]
        if bool(torch.any(exponent <= 0.0)):
            raise RockContractError(
                "Brie exponent must be positive",
                object_name="co2_brine_mix",
                field="brie_exponent",
                expected="> 0 (dimensionless)",
                actual=_extrema(exponent, "1"),
                hint="supply a positive explicit Brie mixing exponent",
            )
    brine = batzle_wang_brine(temperature, pressure, salinity)
    co2 = co2_properties(temperature, pressure)
    brine_saturation = co2_saturation.new_tensor(1.0) - co2_saturation
    if brie_exponent is None:
        bulk_modulus, density = wood_fluid_mix(
            brine.bulk_modulus,
            co2.bulk_modulus,
            brine.rho,
            co2.rho,
            brine_saturation,
        )
    else:
        bulk_modulus, density = brie_fluid_mix(
            brine.bulk_modulus,
            co2.bulk_modulus,
            brine.rho,
            co2.rho,
            brine_saturation,
            exponent=exponent,
        )
    return FluidProperties(bulk_modulus=bulk_modulus, rho=density)


__all__ = [
    "FluidProperties",
    "batzle_wang_brine",
    "batzle_wang_dead_oil",
    "batzle_wang_gas",
    "batzle_wang_live_oil",
    "co2_brine_mix",
    "co2_properties",
    "live_oil_properties",
    "solution_gas_oil_ratio",
]
