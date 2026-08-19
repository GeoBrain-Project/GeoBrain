"""Canonical SI Biot and Geertsma-Smit poroelastic kernels.

The elastic/mass coefficients follow Biot (1956), JASA 28, parts I and II
(DOI 10.1121/1.1908239 and 10.1121/1.1908241). The continuous circular-pore
viscodynamic approximation is Chandrasekaran et al. (2022), JASA 151,
Eq. (2), with pore-shape factor 1/16 (DOI 10.1121/10.0010164).
Geertsma-Smit limits follow Geertsma & Smit (1961), Geophysics 26,
169--181 (DOI 10.1190/1.1438855).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

import torch

from geobrain.core import ErrorCode

from .contracts import require_compatible_tensors
from .errors import RockContractError, RockNumericsError

TensorInput: TypeAlias = torch.Tensor | int | float


@dataclass(frozen=True, slots=True)
class BiotVelocityLimits:
    """Biot high-frequency velocities in m/s."""

    fast_p_velocity: torch.Tensor
    slow_p_velocity: torch.Tensor
    shear_velocity: torch.Tensor


@dataclass(frozen=True, slots=True)
class BiotDispersionResult:
    """Complex SI wavenumbers, phase velocities, and inverse quality factors."""

    fast_p_wavenumber: torch.Tensor
    slow_p_wavenumber: torch.Tensor
    shear_wavenumber: torch.Tensor
    fast_p_velocity: torch.Tensor
    slow_p_velocity: torch.Tensor
    shear_velocity: torch.Tensor
    fast_p_inverse_q: torch.Tensor
    slow_p_inverse_q: torch.Tensor
    shear_inverse_q: torch.Tensor


@dataclass(frozen=True, slots=True)
class _BiotCoefficients:
    bulk_density: torch.Tensor
    rho_11: torch.Tensor
    rho_12: torch.Tensor
    rho_22: torch.Tensor
    p_modulus: torch.Tensor
    q_modulus: torch.Tensor
    r_modulus: torch.Tensor
    storage_modulus: torch.Tensor
    coupling_modulus: torch.Tensor
    saturated_p_modulus: torch.Tensor
    dry_p_modulus: torch.Tensor
    biot_alpha: torch.Tensor
    geertsma_density: torch.Tensor
    geertsma_storage: torch.Tensor


def _validated_inputs(
    object_name: str,
    *fields: tuple[str, TensorInput],
) -> tuple[torch.Tensor, ...]:
    first_name, first_value = fields[0]
    if not isinstance(first_value, torch.Tensor):
        raise RockContractError(
            "Rock poroelastic reference input must be a tensor",
            object_name=object_name,
            field=first_name,
            expected="torch.Tensor reference with dtype float32 or float64",
            actual={"type": type(first_value).__qualname__, "unit": "SI"},
            hint="pass the first documented input as a materialized SI tensor",
        )
    tensors: tuple[torch.Tensor, ...] = require_compatible_tensors(object_name, *fields)
    for (name, _), tensor in zip(fields, tensors):
        if tensor.layout is not torch.strided:
            raise RockContractError(
                "Rock poroelastic kernels require strided tensors",
                object_name=object_name,
                field=name,
                expected="torch.strided layout",
                actual={"layout": str(tensor.layout), "unit": "SI"},
            )
        if tensor.device.type == "meta":
            raise RockContractError(
                "Rock poroelastic kernels require materialized values",
                object_name=object_name,
                field=name,
                expected="a materialized CPU or accelerator tensor",
                actual={"device": str(tensor.device), "unit": "SI"},
                code=ErrorCode.DEVICE_UNAVAILABLE,
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RockContractError(
                "Rock poroelastic input must be finite",
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


def _require_positive(
    object_name: str,
    field: str,
    value: torch.Tensor,
    unit: str,
) -> None:
    if bool(torch.any(value <= 0.0)):
        raise RockContractError(
            "Rock poroelastic input must be positive",
            object_name=object_name,
            field=field,
            expected=f"> 0 {unit}",
            actual=_extrema(value, unit),
            hint=f"supply an unclamped physical {field} in {unit}",
        )


def _require_medium_domain(
    object_name: str,
    k_dry: torch.Tensor,
    shear_modulus: torch.Tensor,
    k_mineral: torch.Tensor,
    k_fluid: torch.Tensor,
    rho_mineral: torch.Tensor,
    rho_fluid: torch.Tensor,
    porosity: torch.Tensor,
    tortuosity: torch.Tensor,
) -> None:
    for field, value, unit in (
        ("k_dry", k_dry, "Pa"),
        ("shear_modulus", shear_modulus, "Pa"),
        ("k_mineral", k_mineral, "Pa"),
        ("k_fluid", k_fluid, "Pa"),
        ("rho_mineral", rho_mineral, "kg/m³"),
        ("rho_fluid", rho_fluid, "kg/m³"),
    ):
        _require_positive(object_name, field, value, unit)
    if bool(torch.any(k_dry > k_mineral)):
        raise RockContractError(
            "Biot dry-frame modulus cannot exceed the mineral modulus",
            object_name=object_name,
            field="k_dry",
            expected="0 < k_dry <= k_mineral in Pa",
            actual=_extrema(k_dry, "Pa"),
        )
    if bool(torch.any(k_fluid > k_mineral)):
        raise RockContractError(
            "Biot fluid modulus cannot exceed the mineral modulus",
            object_name=object_name,
            field="k_fluid",
            expected="0 < k_fluid <= k_mineral in Pa",
            actual=_extrema(k_fluid, "Pa"),
        )
    if bool(torch.any((porosity <= 0.0) | (porosity >= 1.0))):
        raise RockContractError(
            "Biot porosity must lie in the open unit interval",
            object_name=object_name,
            field="porosity",
            expected="0 < porosity < 1",
            actual=_extrema(porosity, "1"),
        )
    if bool(torch.any(tortuosity < 1.0)):
        raise RockContractError(
            "Biot tortuosity cannot be below unity",
            object_name=object_name,
            field="tortuosity",
            expected=">= 1",
            actual=_extrema(tortuosity, "1"),
        )


def _require_non_singular(
    object_name: str,
    field: str,
    value: torch.Tensor,
    scale: torch.Tensor,
    unit: str,
) -> None:
    tolerance = value.new_tensor(64.0 * torch.finfo(value.dtype).eps)
    if bool(torch.any(value.abs() <= tolerance * scale)):
        raise RockNumericsError(
            "Biot coefficient assembly is physically singular",
            object_name=object_name,
            field=field,
            expected="non-zero value relative to the equation scale",
            actual=_extrema(value, unit),
            hint="choose a non-degenerate porous medium inside the Biot domain",
        )


def _require_finite_derived(
    object_name: str,
    field: str,
    value: torch.Tensor,
    unit: str,
) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RockNumericsError(
            "Rock poroelastic calculation produced a non-finite value",
            object_name=object_name,
            field=field,
            expected=f"finite derived value in {unit}",
            actual={"value": "non-finite value(s)", "unit": unit},
            hint="use a wider dtype or choose inputs with a representable physical result",
        )


def _assemble_biot_coefficients(
    object_name: str,
    k_dry: torch.Tensor,
    shear_modulus: torch.Tensor,
    k_mineral: torch.Tensor,
    k_fluid: torch.Tensor,
    rho_mineral: torch.Tensor,
    rho_fluid: torch.Tensor,
    porosity: torch.Tensor,
    tortuosity: torch.Tensor,
) -> _BiotCoefficients:
    """Assemble the one SI coefficient set consumed by every public limit."""

    one = k_dry.new_tensor(1.0)
    bulk_density = (one - porosity) * rho_mineral + porosity * rho_fluid
    rho_12 = (one - tortuosity) * porosity * rho_fluid
    rho_22 = tortuosity * porosity * rho_fluid
    rho_11 = (one - porosity) * rho_mineral - rho_12
    mass_determinant = rho_11 * rho_22 - rho_12.square()
    mass_scale = (rho_11 * rho_22).abs() + rho_12.square()
    _require_non_singular(
        object_name,
        "mass_determinant",
        mass_determinant,
        mass_scale,
        "kg²/m⁶",
    )
    if bool(torch.any(mass_determinant < 0.0)):
        raise RockNumericsError(
            "Biot effective mass matrix is not positive definite",
            object_name=object_name,
            field="mass_determinant",
            expected="> 0 kg²/m⁶",
            actual=_extrema(mass_determinant, "kg²/m⁶"),
        )

    biot_alpha = one - k_dry / k_mineral
    frame_contrast = biot_alpha - porosity
    fluid_compliance = porosity * k_mineral / k_fluid
    elastic_denominator = frame_contrast + fluid_compliance
    _require_non_singular(
        object_name,
        "elastic_denominator",
        elastic_denominator,
        frame_contrast.abs() + fluid_compliance.abs(),
        "1",
    )
    dry_p_modulus = k_dry + k_dry.new_tensor(4.0 / 3.0) * shear_modulus
    geertsma_density = (one - porosity) * rho_mineral + porosity * rho_fluid * (
        one - one / tortuosity
    )
    _require_positive(object_name, "geertsma_density", geertsma_density, "kg/m³")
    geertsma_storage = (biot_alpha - porosity) / k_mineral + porosity / k_fluid
    _require_non_singular(
        object_name,
        "geertsma_storage",
        geertsma_storage,
        ((biot_alpha - porosity) / k_mineral).abs() + (porosity / k_fluid).abs(),
        "Pa⁻¹",
    )
    if bool(torch.any(geertsma_storage < 0.0)):
        raise RockNumericsError(
            "Geertsma-Smit storage coefficient must be positive",
            object_name=object_name,
            field="geertsma_storage",
            expected="> 0 Pa⁻¹",
            actual=_extrema(geertsma_storage, "Pa⁻¹"),
        )
    # Form M from the dimensionless denominator rather than 1/S.  The two are
    # algebraically identical, while reciprocal-compliance backpropagation
    # squares a tiny SI compliance and overflows for valid float32 moduli.
    storage_modulus = k_mineral / elastic_denominator
    coupling_modulus = biot_alpha * storage_modulus
    saturated_p_modulus = dry_p_modulus + biot_alpha * coupling_modulus
    r_modulus = porosity.square() * storage_modulus
    q_modulus = porosity * frame_contrast * storage_modulus
    p_modulus = dry_p_modulus + frame_contrast.square() * storage_modulus
    for field, value, unit in (
        ("storage_modulus", storage_modulus, "Pa"),
        ("coupling_modulus", coupling_modulus, "Pa"),
        ("saturated_p_modulus", saturated_p_modulus, "Pa"),
        ("p_modulus", p_modulus, "Pa"),
        ("q_modulus", q_modulus, "Pa"),
        ("r_modulus", r_modulus, "Pa"),
    ):
        _require_finite_derived(object_name, field, value, unit)
    return _BiotCoefficients(
        bulk_density=bulk_density,
        rho_11=rho_11,
        rho_12=rho_12,
        rho_22=rho_22,
        p_modulus=p_modulus,
        q_modulus=q_modulus,
        r_modulus=r_modulus,
        storage_modulus=storage_modulus,
        coupling_modulus=coupling_modulus,
        saturated_p_modulus=saturated_p_modulus,
        dry_p_modulus=dry_p_modulus,
        biot_alpha=biot_alpha,
        geertsma_density=geertsma_density,
        geertsma_storage=geertsma_storage,
    )


def _validated_medium(
    object_name: str,
    k_dry: torch.Tensor,
    shear_modulus: TensorInput,
    k_mineral: TensorInput,
    k_fluid: TensorInput,
    rho_mineral: TensorInput,
    rho_fluid: TensorInput,
    porosity: TensorInput,
    tortuosity: TensorInput,
) -> tuple[tuple[torch.Tensor, ...], _BiotCoefficients]:
    values = _validated_inputs(
        object_name,
        ("k_dry", k_dry),
        ("shear_modulus", shear_modulus),
        ("k_mineral", k_mineral),
        ("k_fluid", k_fluid),
        ("rho_mineral", rho_mineral),
        ("rho_fluid", rho_fluid),
        ("porosity", porosity),
        ("tortuosity", tortuosity),
    )
    broadcast = torch.broadcast_tensors(*values)
    _require_medium_domain(object_name, *broadcast)
    return broadcast, _assemble_biot_coefficients(object_name, *broadcast)


def biot_high_frequency_limits(
    k_dry: torch.Tensor,
    shear_modulus: TensorInput,
    k_mineral: TensorInput,
    k_fluid: TensorInput,
    rho_mineral: TensorInput,
    rho_fluid: TensorInput,
    porosity: TensorInput,
    tortuosity: TensorInput,
) -> BiotVelocityLimits:
    """Return the Biot (1956 II) high-frequency SI velocity limits in m/s.

    Args:
        k_dry: dry-frame bulk modulus [Pa].
        shear_modulus: frame shear modulus [Pa].
        k_mineral: mineral bulk modulus [Pa].
        k_fluid: pore-fluid bulk modulus [Pa].
        rho_mineral: mineral density [kg/m^3].
        rho_fluid: fluid density [kg/m^3].
        porosity: fractional porosity.
        tortuosity: pore-space tortuosity (>= 1).
    """

    values, coefficients = _validated_medium(
        "biot_high_frequency_limits",
        k_dry,
        shear_modulus,
        k_mineral,
        k_fluid,
        rho_mineral,
        rho_fluid,
        porosity,
        tortuosity,
    )
    shear_modulus = values[1]
    modulus_scale = torch.maximum(
        torch.maximum(coefficients.p_modulus.abs(), coefficients.q_modulus.abs()),
        torch.maximum(coefficients.r_modulus.abs(), coefficients.dry_p_modulus.abs()),
    )
    normalized_p = coefficients.p_modulus / modulus_scale
    normalized_q = coefficients.q_modulus / modulus_scale
    normalized_r = coefficients.r_modulus / modulus_scale
    normalized_dry_p = coefficients.dry_p_modulus / modulus_scale
    normalized_delta = (
        normalized_p * coefficients.rho_22
        + normalized_r * coefficients.rho_11
        - 2.0 * normalized_q * coefficients.rho_12
    )
    _require_finite_derived(
        "biot_high_frequency_limits",
        "compressional_delta",
        normalized_delta,
        "kg/m³",
    )
    if bool(torch.any(normalized_delta <= 0.0)):
        raise RockNumericsError(
            "Biot high-frequency compressional root sum is non-positive",
            object_name="biot_high_frequency_limits",
            field="compressional_delta",
            expected="> 0 after modulus normalization",
            actual=_extrema(normalized_delta, "kg/m³"),
        )
    mass_determinant = coefficients.rho_11 * coefficients.rho_22 - coefficients.rho_12.square()
    normalized_elastic_determinant = normalized_r * normalized_dry_p
    normalized_root_product = (
        mass_determinant * normalized_elastic_determinant / normalized_delta.square()
    )
    normalized_discriminant = 1.0 - 4.0 * normalized_root_product
    _require_finite_derived(
        "biot_high_frequency_limits",
        "compressional_discriminant",
        normalized_discriminant,
        "1",
    )
    if bool(torch.any(normalized_discriminant <= 0.0)):
        raise RockNumericsError(
            "Biot high-frequency compressional roots are not real and distinct",
            object_name="biot_high_frequency_limits",
            field="compressional_discriminant",
            expected="> 0 after SI coefficient normalization",
            actual=_extrema(normalized_discriminant, "1"),
        )
    normalized_root = torch.sqrt(normalized_discriminant)
    normalized_fast_squared = 0.5 * (1.0 + normalized_root)
    normalized_slow_squared = normalized_root_product / normalized_fast_squared
    velocity_squared_scale = modulus_scale * (normalized_delta / mass_determinant)
    fast_squared = velocity_squared_scale * normalized_fast_squared
    slow_squared = velocity_squared_scale * normalized_slow_squared
    shear_squared = shear_modulus / coefficients.geertsma_density
    for field, value in (
        ("fast_p_velocity_squared", fast_squared),
        ("slow_p_velocity_squared", slow_squared),
        ("shear_velocity_squared", shear_squared),
    ):
        _require_finite_derived("biot_high_frequency_limits", field, value, "m²/s²")
        if bool(torch.any(value <= 0.0)):
            raise RockNumericsError(
                "Biot high-frequency velocity is non-propagating",
                object_name="biot_high_frequency_limits",
                field=field,
                expected="> 0 m²/s²",
                actual=_extrema(value, "m²/s²"),
            )
    result = BiotVelocityLimits(
        fast_p_velocity=torch.sqrt(fast_squared),
        slow_p_velocity=torch.sqrt(slow_squared),
        shear_velocity=torch.sqrt(shear_squared),
    )
    for field, value in (
        ("fast_p_velocity", result.fast_p_velocity),
        ("slow_p_velocity", result.slow_p_velocity),
        ("shear_velocity", result.shear_velocity),
    ):
        _require_finite_derived("biot_high_frequency_limits", field, value, "m/s")
    return result


def biot_viscodynamic_factor(zeta: torch.Tensor) -> torch.Tensor:
    """Return ``sqrt(1 + i*zeta**2/16)`` for every positive ``zeta``.

    This is the continuous circular-pore square-root approximation in
    Chandrasekaran et al. (2022), Eq. (2), with shape factor ``1/16``.
    It has value and first derivative continuity for the entire positive axis.
    """

    (zeta,) = _validated_inputs("biot_viscodynamic_factor", ("zeta", zeta))
    _require_positive("biot_viscodynamic_factor", "zeta", zeta, "1")
    t = zeta / zeta.new_tensor(4.0)
    use_large_form = t > 1.0
    small_t = torch.where(use_large_form, torch.ones_like(t), t)
    small_argument = torch.complex(
        torch.ones_like(small_t),
        small_t.square(),
    )
    small_factor = torch.sqrt(small_argument)
    large_t = torch.where(use_large_form, t, torch.ones_like(t))
    inverse_large_t = large_t.reciprocal()
    large_argument = torch.complex(
        inverse_large_t.square(),
        torch.ones_like(inverse_large_t),
    )
    large_factor = large_t * torch.sqrt(large_argument)
    factor = torch.where(use_large_form, large_factor, small_factor)
    _require_finite_derived("biot_viscodynamic_factor", "factor", factor, "1")
    return factor


def _passive_root(value: torch.Tensor) -> torch.Tensor:
    root = torch.sqrt(value)
    return torch.where(root.imag < 0.0, -root, root)


def _stable_quadratic_roots(
    quadratic_a: torch.Tensor,
    quadratic_b: torch.Tensor,
    quadratic_c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both roots after common scaling and cancellation-safe recovery."""

    coefficient_scale = torch.maximum(
        torch.maximum(quadratic_a.abs(), quadratic_b.abs()),
        quadratic_c.abs(),
    )
    scaled_a = quadratic_a / coefficient_scale
    scaled_b = quadratic_b / coefficient_scale
    scaled_c = quadratic_c / coefficient_scale
    discriminant_root = torch.sqrt(scaled_b.square() - 4.0 * scaled_a * scaled_c)
    plus = scaled_b + discriminant_root
    minus = scaled_b - discriminant_root
    stable_discriminant_root = torch.where(
        plus.abs() >= minus.abs(), discriminant_root, -discriminant_root
    )
    stable_q = -0.5 * (scaled_b + stable_discriminant_root)
    first_root = stable_q / scaled_a
    second_root = scaled_c / stable_q
    return first_root, second_root


def _phase_and_inverse_q(
    object_name: str,
    omega: torch.Tensor,
    wavenumber: torch.Tensor,
    field: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not bool(torch.isfinite(wavenumber).all()):
        raise RockNumericsError(
            "Biot root is non-finite",
            object_name=object_name,
            field=field,
            expected="finite passive complex wavenumber",
            actual={"value": "non-finite complex result", "unit": "rad/m"},
        )
    real_wavenumber = wavenumber.real
    if bool(torch.any(real_wavenumber <= 0.0)):
        raise RockNumericsError(
            "Biot root has non-positive propagation constant",
            object_name=object_name,
            field=f"{field}.real",
            expected="> 0 rad/m",
            actual=_extrema(real_wavenumber, "rad/m"),
        )
    velocity = omega / real_wavenumber
    inverse_q = 2.0 * wavenumber.imag / real_wavenumber
    _require_finite_derived(object_name, f"{field}.phase_velocity", velocity, "m/s")
    _require_finite_derived(object_name, f"{field}.inverse_q", inverse_q, "1")
    if bool(torch.any(inverse_q < 0.0)):
        raise RockNumericsError(
            "Biot root violates the passive attenuation convention",
            object_name=object_name,
            field=f"{field}.imag",
            expected=">= 0 rad/m",
            actual=_extrema(wavenumber.imag, "rad/m"),
        )
    return velocity, inverse_q


def biot_wavenumbers(
    frequency: torch.Tensor,
    k_dry: TensorInput,
    shear_modulus: TensorInput,
    k_mineral: TensorInput,
    k_fluid: TensorInput,
    rho_mineral: TensorInput,
    rho_fluid: TensorInput,
    viscosity: TensorInput,
    porosity: TensorInput,
    permeability: TensorInput,
    pore_radius: TensorInput,
    tortuosity: TensorInput,
) -> BiotDispersionResult:
    """Return passive complex Biot wavenumbers and real SI observables.

    ``frequency`` is in Hz, moduli in Pa, densities in kg/m³, viscosity in
    Pa.s, permeability in m², and pore radius in m. The harmonic convention is
    chosen so passive propagation has ``imag(k) >= 0``.
    """

    values = _validated_inputs(
        "biot_wavenumbers",
        ("frequency", frequency),
        ("k_dry", k_dry),
        ("shear_modulus", shear_modulus),
        ("k_mineral", k_mineral),
        ("k_fluid", k_fluid),
        ("rho_mineral", rho_mineral),
        ("rho_fluid", rho_fluid),
        ("viscosity", viscosity),
        ("porosity", porosity),
        ("permeability", permeability),
        ("pore_radius", pore_radius),
        ("tortuosity", tortuosity),
    )
    (
        frequency,
        k_dry,
        shear_modulus,
        k_mineral,
        k_fluid,
        rho_mineral,
        rho_fluid,
        viscosity,
        porosity,
        permeability,
        pore_radius,
        tortuosity,
    ) = torch.broadcast_tensors(*values)
    _require_positive("biot_wavenumbers", "frequency", frequency, "Hz")
    angular_factor = frequency.new_tensor(2.0 * math.pi)
    maximum_frequency = frequency.new_tensor(torch.finfo(frequency.dtype).max) / angular_factor
    if not bool(torch.isfinite(angular_factor * maximum_frequency)):
        maximum_frequency = torch.nextafter(
            maximum_frequency,
            frequency.new_tensor(-torch.inf),
        )
    if bool(torch.any(frequency > maximum_frequency)):
        raise RockContractError(
            "Biot frequency exceeds the representable angular-frequency range",
            object_name="biot_wavenumbers",
            field="frequency",
            expected=f"<= {maximum_frequency.item()!r} Hz for {frequency.dtype}",
            actual=_extrema(frequency, "Hz"),
            hint="reduce frequency or use a wider floating-point dtype",
        )
    _require_positive("biot_wavenumbers", "viscosity", viscosity, "Pa·s")
    _require_positive("biot_wavenumbers", "permeability", permeability, "m²")
    _require_positive("biot_wavenumbers", "pore_radius", pore_radius, "m")
    _require_medium_domain(
        "biot_wavenumbers",
        k_dry,
        shear_modulus,
        k_mineral,
        k_fluid,
        rho_mineral,
        rho_fluid,
        porosity,
        tortuosity,
    )
    coefficients = _assemble_biot_coefficients(
        "biot_wavenumbers",
        k_dry,
        shear_modulus,
        k_mineral,
        k_fluid,
        rho_mineral,
        rho_fluid,
        porosity,
        tortuosity,
    )

    omega = angular_factor * frequency
    zeta = torch.sqrt(omega * pore_radius.square() * rho_fluid / viscosity)
    _require_finite_derived("biot_wavenumbers", "dimensionless_frequency", zeta, "1")
    factor = biot_viscodynamic_factor(zeta)
    drag_density = viscosity * factor / (omega * permeability)
    _require_finite_derived("biot_wavenumbers", "drag_density", drag_density, "kg/m³")
    inertial_density = tortuosity * rho_fluid / porosity
    dynamic_mass = torch.complex(inertial_density, torch.zeros_like(inertial_density))
    dynamic_mass = dynamic_mass + 1j * drag_density
    _require_finite_derived("biot_wavenumbers", "dynamic_mass", dynamic_mass, "kg/m³")

    modulus_scale = torch.maximum(
        torch.maximum(
            coefficients.storage_modulus.abs(),
            coefficients.coupling_modulus.abs(),
        ),
        torch.maximum(
            coefficients.saturated_p_modulus.abs(),
            torch.maximum(coefficients.dry_p_modulus.abs(), shear_modulus.abs()),
        ),
    )
    density_scale = torch.maximum(
        torch.maximum(dynamic_mass.abs(), coefficients.bulk_density.abs()),
        rho_fluid.abs(),
    )
    normalized_storage = coefficients.storage_modulus / modulus_scale
    normalized_coupling = coefficients.coupling_modulus / modulus_scale
    normalized_saturated_p = coefficients.saturated_p_modulus / modulus_scale
    normalized_dry_p = coefficients.dry_p_modulus / modulus_scale
    normalized_shear = shear_modulus / modulus_scale
    normalized_dynamic_mass = dynamic_mass / density_scale
    normalized_bulk_density = coefficients.bulk_density / density_scale
    normalized_fluid_density = rho_fluid / density_scale
    quadratic_a = -(normalized_storage * normalized_dry_p)
    _require_non_singular(
        "biot_wavenumbers",
        "compressional_quadratic_a",
        quadratic_a,
        normalized_storage.abs() * normalized_dry_p.abs(),
        "1",
    )
    quadratic_b = (
        normalized_saturated_p * normalized_dynamic_mass
        + normalized_storage * normalized_bulk_density
        - 2.0 * normalized_coupling * normalized_fluid_density
    )
    quadratic_c = normalized_fluid_density.square() - (
        normalized_bulk_density * normalized_dynamic_mass
    )
    first_normalized_slowness, second_normalized_slowness = _stable_quadratic_roots(
        quadratic_a,
        quadratic_b,
        quadratic_c,
    )
    # Apply the dimensional scale after taking square roots.  Forming
    # density/modulus first is forward-representable but its modulus derivative
    # underflows in float32; reverse mode then encounters 0 * inf.  The split
    # square-root ratio is the same slowness scale with a finite derivative.
    slowness_scale = torch.sqrt(density_scale) / torch.sqrt(modulus_scale)
    shear_normalized_slowness = (
        normalized_bulk_density * normalized_dynamic_mass - normalized_fluid_density.square()
    ) / (normalized_shear * normalized_dynamic_mass)
    for field, value in (
        ("first_normalized_squared_slowness", first_normalized_slowness),
        ("second_normalized_squared_slowness", second_normalized_slowness),
        ("shear_normalized_squared_slowness", shear_normalized_slowness),
    ):
        _require_finite_derived("biot_wavenumbers", field, value, "1")
    _require_finite_derived("biot_wavenumbers", "slowness_scale", slowness_scale, "s/m")

    first_p_wavenumber = omega * slowness_scale * _passive_root(first_normalized_slowness)
    second_p_wavenumber = omega * slowness_scale * _passive_root(second_normalized_slowness)
    shear_wavenumber = omega * slowness_scale * _passive_root(shear_normalized_slowness)
    first_p_velocity, first_p_inverse_q = _phase_and_inverse_q(
        "biot_wavenumbers", omega, first_p_wavenumber, "first_p_wavenumber"
    )
    second_p_velocity, second_p_inverse_q = _phase_and_inverse_q(
        "biot_wavenumbers", omega, second_p_wavenumber, "second_p_wavenumber"
    )
    shear_velocity, shear_inverse_q = _phase_and_inverse_q(
        "biot_wavenumbers", omega, shear_wavenumber, "shear_wavenumber"
    )
    first_is_fast = first_p_velocity >= second_p_velocity
    fast_p_wavenumber = torch.where(first_is_fast, first_p_wavenumber, second_p_wavenumber)
    slow_p_wavenumber = torch.where(first_is_fast, second_p_wavenumber, first_p_wavenumber)
    fast_p_velocity = torch.where(first_is_fast, first_p_velocity, second_p_velocity)
    slow_p_velocity = torch.where(first_is_fast, second_p_velocity, first_p_velocity)
    fast_p_inverse_q = torch.where(first_is_fast, first_p_inverse_q, second_p_inverse_q)
    slow_p_inverse_q = torch.where(first_is_fast, second_p_inverse_q, first_p_inverse_q)
    result = BiotDispersionResult(
        fast_p_wavenumber=fast_p_wavenumber,
        slow_p_wavenumber=slow_p_wavenumber,
        shear_wavenumber=shear_wavenumber,
        fast_p_velocity=fast_p_velocity,
        slow_p_velocity=slow_p_velocity,
        shear_velocity=shear_velocity,
        fast_p_inverse_q=fast_p_inverse_q,
        slow_p_inverse_q=slow_p_inverse_q,
        shear_inverse_q=shear_inverse_q,
    )
    for field, value, unit in (
        ("fast_p_wavenumber", result.fast_p_wavenumber, "rad/m"),
        ("slow_p_wavenumber", result.slow_p_wavenumber, "rad/m"),
        ("shear_wavenumber", result.shear_wavenumber, "rad/m"),
        ("fast_p_velocity", result.fast_p_velocity, "m/s"),
        ("slow_p_velocity", result.slow_p_velocity, "m/s"),
        ("shear_velocity", result.shear_velocity, "m/s"),
        ("fast_p_inverse_q", result.fast_p_inverse_q, "1"),
        ("slow_p_inverse_q", result.slow_p_inverse_q, "1"),
        ("shear_inverse_q", result.shear_inverse_q, "1"),
    ):
        _require_finite_derived("biot_wavenumbers", field, value, unit)
    return result


def _geertsma_smit_high_kernel(
    object_name: str,
    coefficients: _BiotCoefficients,
    shear_modulus: torch.Tensor,
    rho_fluid: torch.Tensor,
    porosity: torch.Tensor,
    tortuosity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inertial_fraction = porosity * coefficients.bulk_density / (rho_fluid * tortuosity)
    correction_numerator = inertial_fraction + coefficients.biot_alpha * (
        coefficients.biot_alpha - 2.0 * porosity / tortuosity
    )
    correction = correction_numerator * coefficients.storage_modulus
    _require_finite_derived(object_name, "inertial_correction", correction, "Pa")
    fast_p_squared = (coefficients.dry_p_modulus + correction) / coefficients.geertsma_density
    shear_squared = shear_modulus / coefficients.geertsma_density
    _require_finite_derived(object_name, "fast_p_velocity_squared", fast_p_squared, "m²/s²")
    _require_finite_derived(object_name, "shear_velocity_squared", shear_squared, "m²/s²")
    if bool(torch.any((fast_p_squared <= 0.0) | (shear_squared <= 0.0))):
        raise RockNumericsError(
            "Geertsma-Smit high-frequency limit is non-propagating",
            object_name=object_name,
            field="velocity_squared",
            expected="> 0 m²/s²",
            actual={
                "fast_p": _extrema(fast_p_squared, "m²/s²"),
                "shear": _extrema(shear_squared, "m²/s²"),
            },
        )
    fast_p_velocity = torch.sqrt(fast_p_squared)
    shear_velocity = torch.sqrt(shear_squared)
    _require_finite_derived(object_name, "fast_p_velocity", fast_p_velocity, "m/s")
    _require_finite_derived(object_name, "shear_velocity", shear_velocity, "m/s")
    return fast_p_velocity, shear_velocity


def geertsma_smit_high_frequency(
    k_dry: torch.Tensor,
    shear_modulus: TensorInput,
    k_mineral: TensorInput,
    k_fluid: TensorInput,
    rho_mineral: TensorInput,
    rho_fluid: TensorInput,
    porosity: TensorInput,
    tortuosity: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Geertsma-Smit (1961) high-frequency P/S limits in m/s."""

    values, coefficients = _validated_medium(
        "geertsma_smit_high_frequency",
        k_dry,
        shear_modulus,
        k_mineral,
        k_fluid,
        rho_mineral,
        rho_fluid,
        porosity,
        tortuosity,
    )
    return _geertsma_smit_high_kernel(
        "geertsma_smit_high_frequency",
        coefficients,
        values[1],
        values[5],
        values[6],
        values[7],
    )


def geertsma_smit_low_frequency(
    frequency: torch.Tensor,
    k_dry: TensorInput,
    shear_modulus: TensorInput,
    k_mineral: TensorInput,
    k_fluid: TensorInput,
    rho_mineral: TensorInput,
    rho_fluid: TensorInput,
    porosity: TensorInput,
    tortuosity: TensorInput,
    viscosity: TensorInput,
    permeability: TensorInput,
) -> torch.Tensor:
    """Return the Geertsma-Smit first-wave low/mid-frequency interpolation.

    The zero- and infinite-frequency velocities are derived from the shared SI
    Biot coefficient assembly; callers cannot supply inconsistent limits.
    """

    values = _validated_inputs(
        "geertsma_smit_low_frequency",
        ("frequency", frequency),
        ("k_dry", k_dry),
        ("shear_modulus", shear_modulus),
        ("k_mineral", k_mineral),
        ("k_fluid", k_fluid),
        ("rho_mineral", rho_mineral),
        ("rho_fluid", rho_fluid),
        ("porosity", porosity),
        ("tortuosity", tortuosity),
        ("viscosity", viscosity),
        ("permeability", permeability),
    )
    (
        frequency,
        k_dry,
        shear_modulus,
        k_mineral,
        k_fluid,
        rho_mineral,
        rho_fluid,
        porosity,
        tortuosity,
        viscosity,
        permeability,
    ) = torch.broadcast_tensors(*values)
    _require_positive("geertsma_smit_low_frequency", "frequency", frequency, "Hz")
    _require_positive("geertsma_smit_low_frequency", "viscosity", viscosity, "Pa·s")
    _require_positive("geertsma_smit_low_frequency", "permeability", permeability, "m²")
    _require_medium_domain(
        "geertsma_smit_low_frequency",
        k_dry,
        shear_modulus,
        k_mineral,
        k_fluid,
        rho_mineral,
        rho_fluid,
        porosity,
        tortuosity,
    )
    coefficients = _assemble_biot_coefficients(
        "geertsma_smit_low_frequency",
        k_dry,
        shear_modulus,
        k_mineral,
        k_fluid,
        rho_mineral,
        rho_fluid,
        porosity,
        tortuosity,
    )
    low_velocity_squared = coefficients.saturated_p_modulus / coefficients.bulk_density
    _require_finite_derived(
        "geertsma_smit_low_frequency",
        "low_velocity_squared",
        low_velocity_squared,
        "m²/s²",
    )
    if bool(torch.any(low_velocity_squared <= 0.0)):
        raise RockNumericsError(
            "Geertsma-Smit low-frequency limit is non-propagating",
            object_name="geertsma_smit_low_frequency",
            field="low_velocity_squared",
            expected="> 0 m²/s²",
            actual=_extrema(low_velocity_squared, "m²/s²"),
        )
    low_velocity = torch.sqrt(low_velocity_squared)
    high_velocity, _ = _geertsma_smit_high_kernel(
        "geertsma_smit_low_frequency",
        coefficients,
        shear_modulus,
        rho_fluid,
        porosity,
        tortuosity,
    )
    corner_frequency = (
        porosity * viscosity / (frequency.new_tensor(2.0 * math.pi) * rho_fluid * permeability)
    )
    velocity_weight_ratio = (low_velocity / high_velocity) * (corner_frequency / frequency)
    use_inverse_weight = velocity_weight_ratio > 1.0
    direct_ratio = torch.where(
        use_inverse_weight,
        torch.ones_like(velocity_weight_ratio),
        velocity_weight_ratio,
    )
    direct_squared = direct_ratio.square()
    inverse_source = torch.where(
        use_inverse_weight,
        velocity_weight_ratio,
        torch.ones_like(velocity_weight_ratio),
    )
    inverse_ratio = inverse_source.reciprocal()
    inverse_squared = inverse_ratio.square()
    low_weight = torch.where(
        use_inverse_weight,
        1.0 / (1.0 + inverse_squared),
        direct_squared / (1.0 + direct_squared),
    )
    interpolated_squared = (
        1.0 - low_weight
    ) * high_velocity.square() + low_weight * low_velocity.square()
    _require_finite_derived(
        "geertsma_smit_low_frequency",
        "interpolated_velocity_squared",
        interpolated_squared,
        "m²/s²",
    )
    interpolated_velocity = torch.sqrt(interpolated_squared)
    _require_finite_derived(
        "geertsma_smit_low_frequency",
        "interpolated_velocity",
        interpolated_velocity,
        "m/s",
    )
    return interpolated_velocity


__all__ = [
    "BiotDispersionResult",
    "BiotVelocityLimits",
    "biot_high_frequency_limits",
    "biot_viscodynamic_factor",
    "biot_wavenumbers",
    "geertsma_smit_high_frequency",
    "geertsma_smit_low_frequency",
]
