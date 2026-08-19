"""Canonical SI Gassmann fluid-substitution kernels.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import torch

from geobrain.core import ErrorCode

from .contracts import require_compatible_tensors
from .anisotropy import isotropic_compliance, stiffness_from_compliance
from .elastic import velocities_from_moduli
from .errors import RockContractError, RockNumericsError

TensorInput: TypeAlias = torch.Tensor | int | float


@dataclass(frozen=True, slots=True)
class GassmannResult:
    """Saturated isotropic properties in Pa, kg/m³, and m/s."""

    k_sat: torch.Tensor
    mu_sat: torch.Tensor
    rho: torch.Tensor
    vp: torch.Tensor
    vs: torch.Tensor


def _validated_inputs(
    object_name: str,
    *fields: tuple[str, TensorInput],
) -> tuple[torch.Tensor, ...]:
    first_name, first_value = fields[0]
    if not isinstance(first_value, torch.Tensor):
        raise RockContractError(
            "Rock fluid-substitution reference input must be a tensor",
            object_name=object_name,
            field=first_name,
            expected="torch.Tensor reference with dtype float32 or float64",
            actual={"type": type(first_value).__qualname__, "unit": "SI"},
            hint="pass the first documented input as a materialized SI tensor",
        )
    tensors = cast(tuple[torch.Tensor, ...], require_compatible_tensors(object_name, *fields))
    for (name, _), tensor in zip(fields, tensors):
        if tensor.layout is not torch.strided:
            raise RockContractError(
                "Rock fluid-substitution kernels require strided tensors",
                object_name=object_name,
                field=name,
                expected="torch.strided layout",
                actual={"layout": str(tensor.layout), "unit": "SI"},
            )
        if tensor.device.type == "meta":
            raise RockContractError(
                "Rock fluid-substitution kernels require materialized values",
                object_name=object_name,
                field=name,
                expected="a materialized CPU or accelerator tensor",
                actual={"device": str(tensor.device), "unit": "SI"},
                code=ErrorCode.DEVICE_UNAVAILABLE,
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RockContractError(
                "Rock fluid-substitution input must be finite",
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
            "Rock fluid-substitution input must be positive",
            object_name=object_name,
            field=field,
            expected=f"> 0 {unit}",
            actual=_extrema(value, unit),
            hint=f"supply an unclamped physical {field} in {unit}",
        )


def _require_porosity(
    object_name: str,
    porosity: torch.Tensor,
) -> None:
    if bool(torch.any((porosity < 0.0) | (porosity >= 1.0))):
        raise RockContractError(
            "Rock porosity must lie in the half-open unit interval",
            object_name=object_name,
            field="porosity",
            expected="0 <= porosity < 1 (dimensionless)",
            actual=_extrema(porosity, "1"),
            hint="supply physical pore fractions without clamping",
        )


def _require_not_stiffer_than_mineral(
    object_name: str,
    field: str,
    value: torch.Tensor,
    mineral: torch.Tensor,
) -> None:
    if bool(torch.any(value > mineral)):
        raise RockContractError(
            "Gassmann constituent cannot exceed the mineral bulk modulus",
            object_name=object_name,
            field=field,
            expected="0 < value <= k_mineral in Pa",
            actual=_extrema(value, "Pa"),
            hint="use a dry-frame or fluid modulus inside the Gassmann domain",
        )


def _validate_forward_domain(
    object_name: str,
    k_dry: torch.Tensor,
    k_mineral: torch.Tensor,
    k_fluid: torch.Tensor,
    porosity: torch.Tensor,
) -> None:
    _require_positive(object_name, "k_dry", k_dry, "Pa")
    _require_positive(object_name, "k_mineral", k_mineral, "Pa")
    _require_positive(object_name, "k_fluid", k_fluid, "Pa")
    _require_porosity(object_name, porosity)
    _require_not_stiffer_than_mineral(object_name, "k_dry", k_dry, k_mineral)
    _require_not_stiffer_than_mineral(object_name, "k_fluid", k_fluid, k_mineral)


def _gassmann_bulk_kernel(
    k_dry: torch.Tensor,
    k_mineral: torch.Tensor,
    k_fluid: torch.Tensor,
    porosity: torch.Tensor,
) -> torch.Tensor:
    dry_compliance_contrast = k_dry.new_tensor(1.0) - k_dry / k_mineral
    denominator = dry_compliance_contrast + porosity * (k_mineral / k_fluid - k_dry.new_tensor(1.0))
    removable_joint_singularity = (dry_compliance_contrast == 0.0) & (denominator == 0.0)
    safe_denominator = torch.where(
        removable_joint_singularity,
        torch.ones_like(denominator),
        denominator,
    )
    correction = k_mineral * dry_compliance_contrast.square() / safe_denominator
    interior = k_dry + correction
    return torch.where(removable_joint_singularity, k_mineral, interior)


def gassmann_bulk_modulus(
    k_dry: torch.Tensor,
    k_mineral: TensorInput,
    k_fluid: TensorInput,
    porosity: TensorInput,
) -> torch.Tensor:
    """Return saturated bulk modulus in Pa from Gassmann's stiffness form.

    This is the sole forward denominator implementation. Inputs follow
    Gassmann (1951): ``0 < k_dry,k_fluid <= k_mineral`` and
    ``0 <= porosity < 1``.
    """

    k_dry, k_mineral, k_fluid, porosity = _validated_inputs(
        "gassmann_bulk_modulus",
        ("k_dry", k_dry),
        ("k_mineral", k_mineral),
        ("k_fluid", k_fluid),
        ("porosity", porosity),
    )
    k_dry, k_mineral, k_fluid, porosity = torch.broadcast_tensors(
        k_dry, k_mineral, k_fluid, porosity
    )
    _validate_forward_domain("gassmann_bulk_modulus", k_dry, k_mineral, k_fluid, porosity)
    return _gassmann_bulk_kernel(k_dry, k_mineral, k_fluid, porosity)


def _gassmann_dry_kernel(
    object_name: str,
    k_saturated: torch.Tensor,
    k_mineral: torch.Tensor,
    k_fluid: torch.Tensor,
    porosity: torch.Tensor,
    *,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    fluid_compliance_contrast = porosity * (k_mineral / k_fluid - k_saturated.new_tensor(1.0))
    saturated_compliance_contrast = k_saturated.new_tensor(1.0) - k_saturated / k_mineral
    denominator = fluid_compliance_contrast - saturated_compliance_contrast
    scale = fluid_compliance_contrast.abs() + saturated_compliance_contrast.abs()
    tolerance = k_saturated.new_tensor(64.0 * torch.finfo(k_saturated.dtype).eps)
    non_identifiable = (porosity == 0.0) | (k_fluid == k_mineral)
    singular = non_identifiable | (denominator.abs() <= tolerance * scale)
    if active is not None:
        singular = singular & active
    if bool(torch.any(singular)):
        raise RockNumericsError(
            "inverse Gassmann configuration is physically singular",
            object_name=object_name,
            field="inverse_denominator",
            expected="non-zero denominator relative to its compressibility scale",
            actual={
                "minimum_absolute_denominator": denominator.abs().amin().item(),
                "unit": "1",
            },
            hint="use non-zero porosity and a fluid modulus distinct from the mineral modulus",
        )
    if active is None:
        active = torch.ones_like(denominator, dtype=torch.bool)
    safe_denominator = torch.where(active, denominator, torch.ones_like(denominator))
    dry_compliance_contrast = (
        saturated_compliance_contrast * fluid_compliance_contrast / safe_denominator
    )
    recovered = k_mineral * (k_saturated.new_tensor(1.0) - dry_compliance_contrast)
    return torch.where(active, recovered, k_saturated)


def gassmann_dry_bulk_modulus(
    k_saturated: torch.Tensor,
    k_mineral: TensorInput,
    k_fluid: TensorInput,
    porosity: TensorInput,
) -> torch.Tensor:
    """Recover dry bulk modulus in Pa using the documented closed-form inverse."""

    k_saturated, k_mineral, k_fluid, porosity = _validated_inputs(
        "gassmann_dry_bulk_modulus",
        ("k_saturated", k_saturated),
        ("k_mineral", k_mineral),
        ("k_fluid", k_fluid),
        ("porosity", porosity),
    )
    k_saturated, k_mineral, k_fluid, porosity = torch.broadcast_tensors(
        k_saturated, k_mineral, k_fluid, porosity
    )
    _require_positive("gassmann_dry_bulk_modulus", "k_saturated", k_saturated, "Pa")
    _require_positive("gassmann_dry_bulk_modulus", "k_mineral", k_mineral, "Pa")
    _require_positive("gassmann_dry_bulk_modulus", "k_fluid", k_fluid, "Pa")
    _require_porosity("gassmann_dry_bulk_modulus", porosity)
    _require_not_stiffer_than_mineral(
        "gassmann_dry_bulk_modulus", "k_saturated", k_saturated, k_mineral
    )
    _require_not_stiffer_than_mineral("gassmann_dry_bulk_modulus", "k_fluid", k_fluid, k_mineral)
    recovered = _gassmann_dry_kernel(
        "gassmann_dry_bulk_modulus",
        k_saturated,
        k_mineral,
        k_fluid,
        porosity,
    )
    if bool(torch.any((recovered <= 0.0) | (recovered > k_mineral))):
        raise RockNumericsError(
            "inverse Gassmann result lies outside the dry-frame domain",
            object_name="gassmann_dry_bulk_modulus",
            field="k_dry",
            expected="0 < k_dry <= k_mineral in Pa",
            actual=_extrema(recovered, "Pa"),
        )
    return recovered


def gassmann_saturated_properties(
    k_dry: torch.Tensor,
    mu_dry: TensorInput,
    k_mineral: TensorInput,
    k_fluid: TensorInput,
    porosity: TensorInput,
    rho_mineral: TensorInput,
    rho_fluid: TensorInput,
) -> GassmannResult:
    """Return saturated moduli, SI density, and isotropic velocities.

    Args:
        k_dry: dry-frame bulk modulus [Pa].
        mu_dry: dry-frame shear modulus [Pa].
        k_mineral: mineral bulk modulus [Pa].
        k_fluid: pore-fluid bulk modulus [Pa].
        porosity: fractional porosity.
        rho_mineral: mineral density [kg/m^3].
        rho_fluid: fluid density [kg/m^3].
    """

    (
        k_dry,
        mu_dry,
        k_mineral,
        k_fluid,
        porosity,
        rho_mineral,
        rho_fluid,
    ) = _validated_inputs(
        "gassmann_saturated_properties",
        ("k_dry", k_dry),
        ("mu_dry", mu_dry),
        ("k_mineral", k_mineral),
        ("k_fluid", k_fluid),
        ("porosity", porosity),
        ("rho_mineral", rho_mineral),
        ("rho_fluid", rho_fluid),
    )
    (
        k_dry,
        mu_dry,
        k_mineral,
        k_fluid,
        porosity,
        rho_mineral,
        rho_fluid,
    ) = torch.broadcast_tensors(
        k_dry,
        mu_dry,
        k_mineral,
        k_fluid,
        porosity,
        rho_mineral,
        rho_fluid,
    )
    _validate_forward_domain(
        "gassmann_saturated_properties",
        k_dry,
        k_mineral,
        k_fluid,
        porosity,
    )
    _require_positive("gassmann_saturated_properties", "mu_dry", mu_dry, "Pa")
    _require_positive("gassmann_saturated_properties", "rho_mineral", rho_mineral, "kg/m^3")
    _require_positive("gassmann_saturated_properties", "rho_fluid", rho_fluid, "kg/m^3")
    k_sat = gassmann_bulk_modulus(k_dry, k_mineral, k_fluid, porosity)
    rho = (porosity.new_tensor(1.0) - porosity) * rho_mineral + (porosity * rho_fluid)
    mu_sat = mu_dry + torch.zeros_like(k_sat)
    vp, vs = velocities_from_moduli(k_sat, mu_sat, rho)
    return GassmannResult(k_sat=k_sat, mu_sat=mu_sat, rho=rho, vp=vp, vs=vs)


def gassmann_substitute_fluid(
    k_saturated: torch.Tensor,
    mu_saturated: TensorInput,
    k_mineral: TensorInput,
    k_fluid_initial: TensorInput,
    k_fluid_final: TensorInput,
    porosity: TensorInput,
    rho_saturated: TensorInput,
    rho_fluid_initial: TensorInput,
    rho_fluid_final: TensorInput,
) -> GassmannResult:
    """Replace one pore fluid with another through a dry-frame intermediate."""

    values = _validated_inputs(
        "gassmann_substitute_fluid",
        ("k_saturated", k_saturated),
        ("mu_saturated", mu_saturated),
        ("k_mineral", k_mineral),
        ("k_fluid_initial", k_fluid_initial),
        ("k_fluid_final", k_fluid_final),
        ("porosity", porosity),
        ("rho_saturated", rho_saturated),
        ("rho_fluid_initial", rho_fluid_initial),
        ("rho_fluid_final", rho_fluid_final),
    )
    (
        k_saturated,
        mu_saturated,
        k_mineral,
        k_fluid_initial,
        k_fluid_final,
        porosity,
        rho_saturated,
        rho_fluid_initial,
        rho_fluid_final,
    ) = torch.broadcast_tensors(*values)
    for name, value, unit in (
        ("k_saturated", k_saturated, "Pa"),
        ("mu_saturated", mu_saturated, "Pa"),
        ("k_mineral", k_mineral, "Pa"),
        ("k_fluid_initial", k_fluid_initial, "Pa"),
        ("k_fluid_final", k_fluid_final, "Pa"),
        ("rho_saturated", rho_saturated, "kg/m^3"),
        ("rho_fluid_initial", rho_fluid_initial, "kg/m^3"),
        ("rho_fluid_final", rho_fluid_final, "kg/m^3"),
    ):
        _require_positive("gassmann_substitute_fluid", name, value, unit)
    _require_porosity("gassmann_substitute_fluid", porosity)
    for name, value in (
        ("k_saturated", k_saturated),
        ("k_fluid_initial", k_fluid_initial),
        ("k_fluid_final", k_fluid_final),
    ):
        _require_not_stiffer_than_mineral("gassmann_substitute_fluid", name, value, k_mineral)

    zero_porosity = porosity == 0.0
    active = ~zero_porosity
    k_dry = _gassmann_dry_kernel(
        "gassmann_substitute_fluid",
        k_saturated,
        k_mineral,
        k_fluid_initial,
        porosity,
        active=active,
    )
    if bool(torch.any(active & ((k_dry <= 0.0) | (k_dry > k_mineral)))):
        raise RockNumericsError(
            "fluid substitution recovered an invalid dry frame",
            object_name="gassmann_substitute_fluid",
            field="k_dry",
            expected="0 < k_dry <= k_mineral in Pa",
            actual=_extrema(k_dry, "Pa"),
        )
    replaced = gassmann_bulk_modulus(k_dry, k_mineral, k_fluid_final, porosity)
    k_final = torch.where(zero_porosity, k_mineral, replaced)
    rho_final = rho_saturated + porosity * (rho_fluid_final - rho_fluid_initial)
    mu_final = mu_saturated + torch.zeros_like(k_final)
    vp, vs = velocities_from_moduli(k_final, mu_final, rho_final)
    return GassmannResult(k_sat=k_final, mu_sat=mu_final, rho=rho_final, vp=vp, vs=vs)


def gassmann_vp_from_dry_frame(
    k_dry: torch.Tensor,
    mu_dry: TensorInput,
    k_mineral: TensorInput,
    k_fluid: TensorInput,
    porosity: TensorInput,
    rho_mineral: TensorInput,
    rho_fluid: TensorInput,
) -> torch.Tensor:
    """Return only ``vp`` for an EarthModel link or other pure-function client.

    Args:
        k_dry / mu_dry: dry-frame moduli [Pa].
        k_mineral / k_fluid: mineral and pore-fluid bulk moduli [Pa].
        porosity: fractional porosity.
        rho_mineral / rho_fluid: densities [kg/m^3].
    """

    return gassmann_saturated_properties(
        k_dry,
        mu_dry,
        k_mineral,
        k_fluid,
        porosity,
        rho_mineral,
        rho_fluid,
    ).vp


def _brown_korringa_inputs(
    object_name: str,
    compliance: torch.Tensor,
    k_mineral: TensorInput,
    mu_mineral: TensorInput,
    k_fluid: TensorInput,
    porosity: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and broadcast one Brown--Korringa compliance call."""
    if not isinstance(compliance, torch.Tensor):
        raise RockContractError(
            "Brown-Korringa compliance must be a tensor",
            object_name=object_name,
            field="compliance",
            expected="torch.Tensor[..., 6, 6] in Pa^-1",
            actual=type(compliance).__qualname__,
        )
    if compliance.ndim < 2 or compliance.shape[-2:] != (6, 6):
        raise RockContractError(
            "Brown-Korringa compliance has the wrong Voigt shape",
            object_name=object_name,
            field="compliance",
            expected="(..., 6, 6)",
            actual=tuple(compliance.shape),
            code=ErrorCode.SHAPE_MISMATCH,
        )
    # This scaled solve also enforces finite, symmetric, positive-definite input.
    stiffness_from_compliance(compliance)
    batch_reference, k_mineral, mu_mineral, k_fluid, porosity = _validated_inputs(
        object_name,
        ("matrix_batch", compliance[..., 0, 0]),
        ("k_mineral", k_mineral),
        ("mu_mineral", mu_mineral),
        ("k_fluid", k_fluid),
        ("porosity", porosity),
    )
    del batch_reference
    batch_shape = torch.broadcast_shapes(
        compliance.shape[:-2],
        k_mineral.shape,
        mu_mineral.shape,
        k_fluid.shape,
        porosity.shape,
    )
    compliance = compliance.expand(batch_shape + (6, 6))
    k_mineral = k_mineral.expand(batch_shape)
    mu_mineral = mu_mineral.expand(batch_shape)
    k_fluid = k_fluid.expand(batch_shape)
    porosity = porosity.expand(batch_shape)
    _require_positive(object_name, "k_mineral", k_mineral, "Pa")
    _require_positive(object_name, "mu_mineral", mu_mineral, "Pa")
    _require_positive(object_name, "k_fluid", k_fluid, "Pa")
    _require_porosity(object_name, porosity)
    _require_not_stiffer_than_mineral(object_name, "k_fluid", k_fluid, k_mineral)
    return compliance, k_mineral, mu_mineral, k_fluid, porosity


def _brown_korringa_update(
    object_name: str,
    compliance: torch.Tensor,
    k_mineral: torch.Tensor,
    mu_mineral: torch.Tensor,
    k_fluid: torch.Tensor,
    porosity: torch.Tensor,
    *,
    inverse: bool,
) -> torch.Tensor:
    """Apply the dimensionless Brown--Korringa rank-one update."""
    mineral = isotropic_compliance(k_mineral, mu_mineral)
    scale = k_mineral[..., None, None]
    scaled = compliance * scale
    scaled_mineral = mineral * scale
    beta = scaled[..., :3, :3].sum(dim=(-2, -1))
    coupling = (scaled[..., :3, :] - scaled_mineral[..., :3, :]).sum(dim=-2)
    fluid_term = porosity * (k_mineral / k_fluid - 1.0)
    denominator = fluid_term - (beta - 1.0) if inverse else fluid_term + (beta - 1.0)
    coupling_scale = coupling.abs().amax(dim=-1)
    tolerance = 64.0 * torch.finfo(compliance.dtype).eps
    removable = (coupling_scale == 0.0) & (denominator == 0.0)
    singular = (~removable) & (
        denominator.abs() <= tolerance * (fluid_term.abs() + (beta - 1.0).abs())
    )
    if bool(torch.any(singular)):
        raise RockNumericsError(
            "Brown-Korringa compliance update is singular",
            object_name=object_name,
            field="substitution_denominator",
            expected="non-zero denominator relative to compressibility scale",
            actual={"minimum_absolute_denominator": denominator.abs().amin().item()},
        )
    safe_denominator = torch.where(removable, torch.ones_like(denominator), denominator)
    update = coupling[..., :, None] * coupling[..., None, :] / safe_denominator[..., None, None]
    candidate = scaled + update if inverse else scaled - update
    result = torch.where(removable[..., None, None], scaled, candidate) / scale
    result = 0.5 * (result + result.transpose(-1, -2))
    # Enforce the scientific domain on the returned compliance as well.
    stiffness_from_compliance(result)
    return result


def brown_korringa_saturated_compliance(
    dry_compliance: torch.Tensor,
    k_mineral: TensorInput,
    mu_mineral: TensorInput,
    k_fluid: TensorInput,
    porosity: TensorInput,
) -> torch.Tensor:
    """Return saturated anisotropic compliance in Pa⁻¹.

    The calculation is scaled by the caller-dtype mineral bulk modulus before
    the rank-one update, preserving float32 precision without changing units.
    """
    values = _brown_korringa_inputs(
        "brown_korringa_saturated_compliance",
        dry_compliance,
        k_mineral,
        mu_mineral,
        k_fluid,
        porosity,
    )
    return _brown_korringa_update("brown_korringa_saturated_compliance", *values, inverse=False)


def brown_korringa_dry_compliance(
    saturated_compliance: torch.Tensor,
    k_mineral: TensorInput,
    mu_mineral: TensorInput,
    k_fluid: TensorInput,
    porosity: TensorInput,
) -> torch.Tensor:
    """Recover dry anisotropic compliance in Pa⁻¹."""
    values = _brown_korringa_inputs(
        "brown_korringa_dry_compliance",
        saturated_compliance,
        k_mineral,
        mu_mineral,
        k_fluid,
        porosity,
    )
    return _brown_korringa_update("brown_korringa_dry_compliance", *values, inverse=True)


def brown_korringa_substitute_fluid(
    saturated_compliance: torch.Tensor,
    k_mineral: TensorInput,
    mu_mineral: TensorInput,
    k_fluid_initial: TensorInput,
    k_fluid_final: TensorInput,
    porosity: TensorInput,
) -> torch.Tensor:
    """Substitute anisotropic pore fluid through an explicit dry intermediate."""
    dry = brown_korringa_dry_compliance(
        saturated_compliance,
        k_mineral,
        mu_mineral,
        k_fluid_initial,
        porosity,
    )
    return brown_korringa_saturated_compliance(
        dry,
        k_mineral,
        mu_mineral,
        k_fluid_final,
        porosity,
    )


__all__ = [
    "GassmannResult",
    "brown_korringa_dry_compliance",
    "brown_korringa_saturated_compliance",
    "brown_korringa_substitute_fluid",
    "gassmann_bulk_modulus",
    "gassmann_dry_bulk_modulus",
    "gassmann_saturated_properties",
    "gassmann_substitute_fluid",
    "gassmann_vp_from_dry_frame",
]
