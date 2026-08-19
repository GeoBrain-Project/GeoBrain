"""Canonical SI granular-contact, sand, and cement constitutive kernels.

All public functions preserve the first tensor input's dtype and device,
broadcast compatible inputs, and reject non-physical domains instead of
clamping them. Moduli and pressure are expressed in Pa.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias, cast

import torch

from geobrain.core import ErrorCode

from .contracts import require_compatible_tensors
from .elastic import poisson_ratio_from_moduli
from .errors import RockContractError, RockNumericsError

TensorInput: TypeAlias = torch.Tensor | int | float


@dataclass(frozen=True, slots=True)
class GranularModuli:
    """One immutable pair of dry-frame bulk and shear moduli in Pa."""

    k_dry: torch.Tensor
    mu_dry: torch.Tensor


@dataclass(frozen=True, slots=True)
class ConstantCementResult:
    """Constant-cement dry moduli and their canonical contact endpoint."""

    k_dry: torch.Tensor
    mu_dry: torch.Tensor
    contact_endpoint: GranularModuli


def _validated_inputs(
    object_name: str,
    *fields: tuple[str, TensorInput],
) -> tuple[torch.Tensor, ...]:
    tensors = cast(
        tuple[torch.Tensor, ...],
        require_compatible_tensors(object_name, *fields),
    )
    for (field, _), tensor in zip(fields, tensors, strict=True):
        if tensor.layout is not torch.strided:
            raise RockContractError(
                "granular kernels require strided tensors",
                object_name=object_name,
                field=field,
                expected="torch.strided layout",
                actual=str(tensor.layout),
            )
        if tensor.device.type == "meta":
            raise RockContractError(
                "granular kernels require materialized tensors",
                object_name=object_name,
                field=field,
                expected="materialized CPU or accelerator tensor",
                actual=str(tensor.device),
                code=ErrorCode.DEVICE_UNAVAILABLE,
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RockContractError(
                "granular input must be finite",
                object_name=object_name,
                field=field,
                expected="finite values",
                actual="non-finite value(s)",
            )
    return tensors


def _extrema(value: torch.Tensor) -> dict[str, float]:
    return {"minimum": value.amin().item(), "maximum": value.amax().item()}


def _require_positive(
    object_name: str,
    field: str,
    value: torch.Tensor,
) -> None:
    if bool(torch.any(value <= 0.0)):
        raise RockContractError(
            "granular input must be positive",
            object_name=object_name,
            field=field,
            expected="> 0",
            actual=_extrema(value),
        )


def _require_nonnegative(
    object_name: str,
    field: str,
    value: torch.Tensor,
) -> None:
    if bool(torch.any(value < 0.0)):
        raise RockContractError(
            "granular input must be non-negative",
            object_name=object_name,
            field=field,
            expected=">= 0",
            actual=_extrema(value),
        )


def _require_open_porosity(
    object_name: str,
    field: str,
    value: torch.Tensor,
) -> None:
    if bool(torch.any((value <= 0.0) | (value >= 1.0))):
        raise RockContractError(
            "critical porosity must lie strictly between zero and one",
            object_name=object_name,
            field=field,
            expected="0 < value < 1",
            actual=_extrema(value),
        )


def _require_unit_interval(
    object_name: str,
    field: str,
    value: torch.Tensor,
) -> None:
    if bool(torch.any((value < 0.0) | (value > 1.0))):
        raise RockContractError(
            "granular fraction must lie in the unit interval",
            object_name=object_name,
            field=field,
            expected="0 <= value <= 1",
            actual=_extrema(value),
        )


def _require_porosity_not_above(
    object_name: str,
    porosity: torch.Tensor,
    upper: torch.Tensor,
    *,
    upper_name: str,
) -> None:
    if bool(torch.any((porosity < 0.0) | (porosity > upper))):
        raise RockContractError(
            "porosity is outside the model domain",
            object_name=object_name,
            field="porosity",
            expected=f"0 <= porosity <= {upper_name}",
            actual=_extrema(porosity),
        )


def _require_positive_result(object_name: str, result: GranularModuli) -> None:
    for field, value in (("k_dry", result.k_dry), ("mu_dry", result.mu_dry)):
        if not bool(torch.isfinite(value).all()) or bool(torch.any(value <= 0.0)):
            raise RockNumericsError(
                "granular equation produced non-positive or non-finite moduli",
                object_name=object_name,
                field=field,
                expected="finite values > 0",
                actual=_extrema(value),
            )


def hertz_mindlin_moduli(
    effective_pressure: torch.Tensor,
    k_mineral: TensorInput,
    mu_mineral: TensorInput,
    critical_porosity: TensorInput,
    coordination_number: TensorInput,
    friction_factor: TensorInput | None = None,
) -> GranularModuli:
    """Return Hertz-Mindlin moduli at critical porosity in Pa.

    Args:
        effective_pressure: effective stress [Pa].
        k_mineral: grain bulk modulus [Pa].
        mu_mineral: grain shear modulus [Pa].
        critical_porosity: random-pack critical porosity.
        coordination_number: mean grain contacts.
        friction_factor: fraction of frictional contacts (``None`` = 1).
    """

    (
        effective_pressure,
        k_mineral,
        mu_mineral,
        critical_porosity,
        coordination_number,
    ) = _validated_inputs(
        "hertz_mindlin_moduli",
        ("effective_pressure", effective_pressure),
        ("k_mineral", k_mineral),
        ("mu_mineral", mu_mineral),
        ("critical_porosity", critical_porosity),
        ("coordination_number", coordination_number),
    )
    _require_positive("hertz_mindlin_moduli", "effective_pressure", effective_pressure)
    _require_positive("hertz_mindlin_moduli", "k_mineral", k_mineral)
    _require_positive("hertz_mindlin_moduli", "mu_mineral", mu_mineral)
    _require_open_porosity("hertz_mindlin_moduli", "critical_porosity", critical_porosity)
    _require_positive("hertz_mindlin_moduli", "coordination_number", coordination_number)
    values = torch.broadcast_tensors(
        effective_pressure,
        k_mineral,
        mu_mineral,
        critical_porosity,
        coordination_number,
    )
    pressure, k_mineral, mu_mineral, phi_c, coordination = values
    poisson = poisson_ratio_from_moduli(k_mineral, mu_mineral)
    common = (
        coordination.square()
        * (1.0 - phi_c).square()
        * mu_mineral.square()
        / (math.pi**2 * (1.0 - poisson).square())
    )
    k_hm = (common * pressure / 18.0).pow(1.0 / 3.0)
    if friction_factor is None:
        mu_factor = (5.0 - 4.0 * poisson) / (5.0 * (2.0 - poisson))
    else:
        _, friction = _validated_inputs(
            "hertz_mindlin_moduli",
            ("effective_pressure", pressure),
            ("friction_factor", friction_factor),
        )
        if bool(torch.any((friction < 0.0) | (friction > 1.0))):
            raise RockContractError(
                "friction factor must lie in the unit interval",
                object_name="hertz_mindlin_moduli",
                field="friction_factor",
                expected="0 <= value <= 1",
                actual=_extrema(friction),
            )
        friction = torch.broadcast_to(friction, pressure.shape)
        mu_factor = (2.0 + 3.0 * friction - poisson * (1.0 + 3.0 * friction)) / (
            5.0 * (2.0 - poisson)
        )
    mu_hm = mu_factor * (1.5 * common * pressure).pow(1.0 / 3.0)
    result = GranularModuli(k_hm, mu_hm)
    _require_positive_result("hertz_mindlin_moduli", result)
    return result


def walton_moduli(
    effective_pressure: torch.Tensor,
    k_mineral: TensorInput,
    mu_mineral: TensorInput,
    critical_porosity: TensorInput,
    coordination_number: TensorInput,
) -> GranularModuli:
    """Return Walton frictionless-contact moduli in Pa."""

    hertz = hertz_mindlin_moduli(
        effective_pressure,
        k_mineral,
        mu_mineral,
        critical_porosity,
        coordination_number,
    )
    return GranularModuli(hertz.k_dry, 0.6 * hertz.k_dry)


def _sand_inputs(
    object_name: str,
    porosity: torch.Tensor,
    k_mineral: TensorInput,
    mu_mineral: TensorInput,
    k_contact: TensorInput,
    mu_contact: TensorInput,
    critical_porosity: TensorInput,
) -> tuple[torch.Tensor, ...]:
    values = _validated_inputs(
        object_name,
        ("porosity", porosity),
        ("k_mineral", k_mineral),
        ("mu_mineral", mu_mineral),
        ("k_contact", k_contact),
        ("mu_contact", mu_contact),
        ("critical_porosity", critical_porosity),
    )
    porosity, k_mineral, mu_mineral, k_contact, mu_contact, phi_c = values
    for field, value in (
        ("k_mineral", k_mineral),
        ("mu_mineral", mu_mineral),
        ("k_contact", k_contact),
        ("mu_contact", mu_contact),
    ):
        _require_positive(object_name, field, value)
    _require_open_porosity(object_name, "critical_porosity", phi_c)
    porosity, k_mineral, mu_mineral, k_contact, mu_contact, phi_c = torch.broadcast_tensors(*values)
    _require_porosity_not_above(
        object_name,
        porosity,
        phi_c,
        upper_name="critical_porosity",
    )
    return porosity, k_mineral, mu_mineral, k_contact, mu_contact, phi_c


def soft_sand_moduli(
    porosity: torch.Tensor,
    k_mineral: TensorInput,
    mu_mineral: TensorInput,
    k_contact: TensorInput,
    mu_contact: TensorInput,
    critical_porosity: TensorInput,
) -> GranularModuli:
    """Return the modified Hashin-Shtrikman lower (soft-sand) line."""

    porosity, k_mineral, mu_mineral, k_contact, mu_contact, phi_c = _sand_inputs(
        "soft_sand_moduli",
        porosity,
        k_mineral,
        mu_mineral,
        k_contact,
        mu_contact,
        critical_porosity,
    )
    fraction = porosity / phi_c
    zeta = mu_contact / 6.0 * (9.0 * k_contact + 8.0 * mu_contact) / (k_contact + 2.0 * mu_contact)
    bulk_offset = 4.0 * mu_contact / 3.0
    k_dry = (
        fraction / (k_contact + bulk_offset) + (1.0 - fraction) / (k_mineral + bulk_offset)
    ).reciprocal() - bulk_offset
    mu_dry = (
        fraction / (mu_contact + zeta) + (1.0 - fraction) / (mu_mineral + zeta)
    ).reciprocal() - zeta
    result = GranularModuli(k_dry, mu_dry)
    _require_positive_result("soft_sand_moduli", result)
    return result


def stiff_sand_moduli(
    porosity: torch.Tensor,
    k_mineral: TensorInput,
    mu_mineral: TensorInput,
    k_contact: TensorInput,
    mu_contact: TensorInput,
    critical_porosity: TensorInput,
) -> GranularModuli:
    """Return the modified Hashin-Shtrikman upper (stiff-sand) line."""

    porosity, k_mineral, mu_mineral, k_contact, mu_contact, phi_c = _sand_inputs(
        "stiff_sand_moduli",
        porosity,
        k_mineral,
        mu_mineral,
        k_contact,
        mu_contact,
        critical_porosity,
    )
    fraction = porosity / phi_c
    zeta = mu_mineral / 6.0 * (9.0 * k_mineral + 8.0 * mu_mineral) / (k_mineral + 2.0 * mu_mineral)
    bulk_offset = 4.0 * mu_mineral / 3.0
    k_dry = (
        fraction / (k_contact + bulk_offset) + (1.0 - fraction) / (k_mineral + bulk_offset)
    ).reciprocal() - bulk_offset
    mu_dry = (
        fraction / (mu_contact + zeta) + (1.0 - fraction) / (mu_mineral + zeta)
    ).reciprocal() - zeta
    result = GranularModuli(k_dry, mu_dry)
    _require_positive_result("stiff_sand_moduli", result)
    return result


def contact_radius(
    cement_fraction: torch.Tensor,
    critical_porosity: TensorInput,
    coordination_number: TensorInput,
    *,
    scheme: int,
) -> torch.Tensor:
    """Return the published Dvorkin-Nur dimensionless contact radius."""

    if type(scheme) is not int or scheme not in (1, 2):
        raise RockContractError(
            "unknown contact-cement deposition scheme",
            object_name="contact_radius",
            field="scheme",
            expected="exact integer 1 or 2",
            actual=scheme,
        )

    cement_fraction, critical_porosity, coordination_number = _validated_inputs(
        "contact_radius",
        ("cement_fraction", cement_fraction),
        ("critical_porosity", critical_porosity),
        ("coordination_number", coordination_number),
    )
    _require_nonnegative("contact_radius", "cement_fraction", cement_fraction)
    _require_open_porosity("contact_radius", "critical_porosity", critical_porosity)
    _require_positive("contact_radius", "coordination_number", coordination_number)
    cement_fraction, critical_porosity, coordination_number = torch.broadcast_tensors(
        cement_fraction,
        critical_porosity,
        coordination_number,
    )
    if bool(torch.any(cement_fraction >= critical_porosity)):
        raise RockContractError(
            "cement fraction must leave positive contact porosity",
            object_name="contact_radius",
            field="cement_fraction",
            expected="0 <= cement_fraction < critical_porosity",
            actual=_extrema(cement_fraction),
        )
    zero_cement = cement_fraction == 0.0
    safe_cement = torch.where(
        zero_cement,
        torch.ones_like(cement_fraction),
        cement_fraction,
    )
    if scheme == 1:
        denominator = 3.0 * coordination_number * (1.0 - critical_porosity)
        radius = 2.0 * (safe_cement / denominator).pow(0.25)
        return torch.where(zero_cement, torch.zeros_like(radius), radius)
    if scheme == 2:
        denominator = 3.0 * (1.0 - critical_porosity)
        radius = (2.0 * safe_cement / denominator).sqrt()
        return torch.where(zero_cement, torch.zeros_like(radius), radius)
    raise AssertionError("validated contact-cement scheme was not handled")


def contact_cement_moduli(
    k_mineral: torch.Tensor,
    mu_mineral: TensorInput,
    k_cement: TensorInput,
    mu_cement: TensorInput,
    cement_fraction: TensorInput,
    critical_porosity: TensorInput,
    coordination_number: TensorInput,
    *,
    scheme: int,
) -> GranularModuli:
    """Return the canonical full Dvorkin-Nur contact-cement endpoint."""

    values = _validated_inputs(
        "contact_cement_moduli",
        ("k_mineral", k_mineral),
        ("mu_mineral", mu_mineral),
        ("k_cement", k_cement),
        ("mu_cement", mu_cement),
        ("cement_fraction", cement_fraction),
        ("critical_porosity", critical_porosity),
        ("coordination_number", coordination_number),
    )
    (
        k_mineral,
        mu_mineral,
        k_cement,
        mu_cement,
        cement_fraction,
        critical_porosity,
        coordination_number,
    ) = values
    for field, value in (
        ("k_mineral", k_mineral),
        ("mu_mineral", mu_mineral),
        ("k_cement", k_cement),
        ("mu_cement", mu_cement),
    ):
        _require_positive("contact_cement_moduli", field, value)
    k_mineral, mu_mineral, k_cement, mu_cement, cement_fraction, phi_c, coordination = (
        torch.broadcast_tensors(*values)
    )
    alpha = contact_radius(
        cement_fraction,
        phi_c,
        coordination,
        scheme=scheme,
    )
    poisson_mineral = poisson_ratio_from_moduli(k_mineral, mu_mineral)
    poisson_cement = poisson_ratio_from_moduli(k_cement, mu_cement)
    lambda_n = (
        2.0
        * mu_cement
        * (1.0 - poisson_mineral)
        * (1.0 - poisson_cement)
        / (math.pi * mu_mineral * (1.0 - 2.0 * poisson_cement))
    )
    lambda_t = mu_cement / (math.pi * mu_mineral)
    n1 = -0.024153 * lambda_n.pow(-1.3646)
    n2 = 0.20405 * lambda_n.pow(-0.89008)
    n3 = 0.00024649 * lambda_n.pow(-1.9864)
    s_n = n1 * alpha.square() + n2 * alpha + n3
    t1 = (
        -0.01
        * (2.26 * poisson_mineral.square() + 2.07 * poisson_mineral + 2.3)
        * lambda_t.pow(0.079 * poisson_mineral.square() + 0.1754 * poisson_mineral - 1.342)
    )
    t2 = (0.0573 * poisson_mineral.square() + 0.0937 * poisson_mineral + 0.202) * lambda_t.pow(
        0.0274 * poisson_mineral.square() + 0.0529 * poisson_mineral - 0.8765
    )
    t3 = (
        0.0001
        * (9.654 * poisson_mineral.square() + 4.945 * poisson_mineral + 3.1)
        * lambda_t.pow(0.01867 * poisson_mineral.square() + 0.4011 * poisson_mineral - 1.8186)
    )
    s_t = t1 * alpha.square() + t2 * alpha + t3
    k_dry = coordination * (1.0 - phi_c) * (k_cement + 4.0 * mu_cement / 3.0) * s_n / 6.0
    mu_dry = 3.0 * k_dry / 5.0 + 3.0 * coordination * (1.0 - phi_c) * mu_cement * s_t / 20.0
    zero_cement = cement_fraction == 0.0
    result = GranularModuli(
        torch.where(zero_cement, torch.zeros_like(k_dry), k_dry),
        torch.where(zero_cement, torch.zeros_like(mu_dry), mu_dry),
    )
    for field, value in (("k_dry", result.k_dry), ("mu_dry", result.mu_dry)):
        invalid = (~torch.isfinite(value)) | (value < 0.0)
        if bool(torch.any(invalid)):
            raise RockNumericsError(
                "contact-cement equation produced invalid moduli",
                object_name="contact_cement_moduli",
                field=field,
                expected="finite values >= 0",
                actual=_extrema(value),
            )
    return result


def constant_cement_moduli(
    porosity: torch.Tensor,
    k_mineral: TensorInput,
    mu_mineral: TensorInput,
    k_cement: TensorInput,
    mu_cement: TensorInput,
    cement_fraction: TensorInput,
    critical_porosity: TensorInput,
    coordination_number: TensorInput,
    *,
    scheme: int,
) -> ConstantCementResult:
    """Return the constant-cement line from one canonical contact endpoint."""

    values = _validated_inputs(
        "constant_cement_moduli",
        ("porosity", porosity),
        ("k_mineral", k_mineral),
        ("mu_mineral", mu_mineral),
        ("k_cement", k_cement),
        ("mu_cement", mu_cement),
        ("cement_fraction", cement_fraction),
        ("critical_porosity", critical_porosity),
        ("coordination_number", coordination_number),
    )
    porosity, k_mineral, mu_mineral, k_cement, mu_cement, cement_fraction, phi_c, coordination = (
        torch.broadcast_tensors(*values)
    )
    endpoint = contact_cement_moduli(
        k_mineral,
        mu_mineral,
        k_cement,
        mu_cement,
        cement_fraction,
        phi_c,
        coordination,
        scheme=scheme,
    )
    endpoint_porosity = phi_c - cement_fraction
    _require_porosity_not_above(
        "constant_cement_moduli",
        porosity,
        endpoint_porosity,
        upper_name="critical_porosity - cement_fraction",
    )
    if bool(torch.any(endpoint_porosity <= 0.0)):
        raise RockContractError(
            "constant-cement endpoint porosity must be positive",
            object_name="constant_cement_moduli",
            field="cement_fraction",
            expected="cement_fraction < critical_porosity",
            actual=_extrema(cement_fraction),
        )
    fraction = porosity / endpoint_porosity
    zero_endpoint = (endpoint.k_dry == 0.0) & (endpoint.mu_dry == 0.0)
    safe_k_endpoint = torch.where(
        zero_endpoint,
        torch.ones_like(endpoint.k_dry),
        endpoint.k_dry,
    )
    safe_mu_endpoint = torch.where(
        zero_endpoint,
        torch.ones_like(endpoint.mu_dry),
        endpoint.mu_dry,
    )
    bulk_offset = 4.0 * safe_mu_endpoint / 3.0
    normal_k = (
        fraction / (safe_k_endpoint + bulk_offset) + (1.0 - fraction) / (k_mineral + bulk_offset)
    ).reciprocal() - bulk_offset
    k_dry = torch.where(
        zero_endpoint,
        torch.where(porosity == 0.0, k_mineral, torch.zeros_like(normal_k)),
        normal_k,
    )
    zeta = (
        safe_mu_endpoint
        / 6.0
        * (9.0 * safe_k_endpoint + 8.0 * safe_mu_endpoint)
        / (safe_k_endpoint + 2.0 * safe_mu_endpoint)
    )
    normal_mu = (
        fraction / (safe_mu_endpoint + zeta) + (1.0 - fraction) / (mu_mineral + zeta)
    ).reciprocal() - zeta
    mu_dry = torch.where(
        zero_endpoint,
        torch.where(porosity == 0.0, mu_mineral, torch.zeros_like(normal_mu)),
        normal_mu,
    )
    result = GranularModuli(k_dry, mu_dry)
    for field, value in (("k_dry", result.k_dry), ("mu_dry", result.mu_dry)):
        if not bool(torch.isfinite(value).all()) or bool(torch.any(value < 0.0)):
            raise RockNumericsError(
                "constant-cement equation produced invalid moduli",
                object_name="constant_cement_moduli",
                field=field,
                expected="finite values >= 0",
                actual=_extrema(value),
            )
    return ConstantCementResult(result.k_dry, result.mu_dry, endpoint)


def modified_upper_hashin_shtrikman(
    porosity: torch.Tensor,
    k_mineral: TensorInput,
    mu_mineral: TensorInput,
    k_cement: TensorInput,
    mu_cement: TensorInput,
    cement_fraction: TensorInput,
    critical_porosity: TensorInput,
    coordination_number: TensorInput,
    *,
    scheme: int,
) -> ConstantCementResult:
    """Return the increasing-cement MUHS line through the same endpoint."""

    values = _validated_inputs(
        "modified_upper_hashin_shtrikman",
        ("porosity", porosity),
        ("k_mineral", k_mineral),
        ("mu_mineral", mu_mineral),
        ("k_cement", k_cement),
        ("mu_cement", mu_cement),
        ("cement_fraction", cement_fraction),
        ("critical_porosity", critical_porosity),
        ("coordination_number", coordination_number),
    )
    porosity, k_mineral, mu_mineral, k_cement, mu_cement, cement_fraction, phi_c, coordination = (
        torch.broadcast_tensors(*values)
    )
    endpoint = contact_cement_moduli(
        k_mineral,
        mu_mineral,
        k_cement,
        mu_cement,
        cement_fraction,
        phi_c,
        coordination,
        scheme=scheme,
    )
    endpoint_porosity = phi_c - cement_fraction
    _require_porosity_not_above(
        "modified_upper_hashin_shtrikman",
        porosity,
        endpoint_porosity,
        upper_name="critical_porosity - cement_fraction",
    )
    fraction = porosity / endpoint_porosity
    bulk_offset = 4.0 * mu_mineral / 3.0
    k_dry = (
        fraction / (endpoint.k_dry + bulk_offset) + (1.0 - fraction) / (k_mineral + bulk_offset)
    ).reciprocal() - bulk_offset
    zeta = mu_mineral / 6.0 * (9.0 * k_mineral + 8.0 * mu_mineral) / (k_mineral + 2.0 * mu_mineral)
    mu_dry = (
        fraction / (endpoint.mu_dry + zeta) + (1.0 - fraction) / (mu_mineral + zeta)
    ).reciprocal() - zeta
    result = GranularModuli(k_dry, mu_dry)
    for field, value in (("k_dry", result.k_dry), ("mu_dry", result.mu_dry)):
        if not bool(torch.isfinite(value).all()) or bool(torch.any(value < 0.0)):
            raise RockNumericsError(
                "modified upper Hashin-Shtrikman equation produced invalid moduli",
                object_name="modified_upper_hashin_shtrikman",
                field=field,
                expected="finite values >= 0",
                actual=_extrema(value),
            )
    return ConstantCementResult(result.k_dry, result.mu_dry, endpoint)


def _patch_endpoints(
    k_uncemented: torch.Tensor,
    mu_uncemented: torch.Tensor,
    k_cemented: torch.Tensor,
    mu_cemented: torch.Tensor,
    cemented_patch_fraction: torch.Tensor,
) -> tuple[GranularModuli, GranularModuli]:
    """Return stiff and soft HS patch mixtures without reciprocal epsilons."""

    fraction = cemented_patch_fraction
    zero_cement = (k_cemented == 0.0) & (mu_cemented == 0.0)
    safe_k_cemented = torch.where(zero_cement, k_uncemented, k_cemented)
    safe_mu_cemented = torch.where(zero_cement, mu_uncemented, mu_cemented)
    bulk_delta = k_uncemented - safe_k_cemented
    shear_delta = mu_uncemented - safe_mu_cemented
    cemented_p = safe_k_cemented + 4.0 * safe_mu_cemented / 3.0
    uncemented_p = k_uncemented + 4.0 * mu_uncemented / 3.0
    k_stiff = safe_k_cemented + (
        (1.0 - fraction) * bulk_delta * cemented_p / (cemented_p + fraction * bulk_delta)
    )
    shear_factor_stiff = (safe_k_cemented + 2.0 * safe_mu_cemented) / (
        5.0 * safe_mu_cemented * cemented_p
    )
    mu_stiff = safe_mu_cemented + (
        (1.0 - fraction) * shear_delta / (1.0 + 2.0 * fraction * shear_factor_stiff * shear_delta)
    )
    k_soft = k_uncemented + (
        fraction * (-bulk_delta) * uncemented_p / (uncemented_p + (1.0 - fraction) * (-bulk_delta))
    )
    shear_factor_soft = (k_uncemented + 2.0 * mu_uncemented) / (5.0 * mu_uncemented * uncemented_p)
    mu_soft = mu_uncemented + (
        fraction
        * (-shear_delta)
        / (1.0 + 2.0 * (1.0 - fraction) * shear_factor_soft * (-shear_delta))
    )
    k_stiff = torch.where(zero_cement, k_uncemented, k_stiff)
    mu_stiff = torch.where(zero_cement, mu_uncemented, mu_stiff)
    k_soft = torch.where(zero_cement, k_uncemented, k_soft)
    mu_soft = torch.where(zero_cement, mu_uncemented, mu_soft)
    return GranularModuli(k_stiff, mu_stiff), GranularModuli(k_soft, mu_soft)


def patchy_cement_moduli(
    porosity: torch.Tensor,
    k_mineral: TensorInput,
    mu_mineral: TensorInput,
    k_cement: TensorInput,
    mu_cement: TensorInput,
    critical_porosity: TensorInput,
    cement_fraction: TensorInput,
    cemented_patch_fraction: TensorInput,
    coordination_number: TensorInput,
    effective_pressure: TensorInput,
    friction_factor: TensorInput = 0.5,
    *,
    scheme: int,
    mode: str,
) -> GranularModuli:
    """Return the PCM dry frame from canonical HM and cement endpoints."""

    values = _validated_inputs(
        "patchy_cement_moduli",
        ("porosity", porosity),
        ("k_mineral", k_mineral),
        ("mu_mineral", mu_mineral),
        ("k_cement", k_cement),
        ("mu_cement", mu_cement),
        ("critical_porosity", critical_porosity),
        ("cement_fraction", cement_fraction),
        ("cemented_patch_fraction", cemented_patch_fraction),
        ("coordination_number", coordination_number),
        ("effective_pressure", effective_pressure),
        ("friction_factor", friction_factor),
    )
    (
        porosity,
        k_mineral,
        mu_mineral,
        k_cement,
        mu_cement,
        phi_c,
        cement_fraction,
        patch_fraction,
        coordination,
        pressure,
        friction,
    ) = torch.broadcast_tensors(*values)
    _require_unit_interval("patchy_cement_moduli", "cemented_patch_fraction", patch_fraction)
    _require_unit_interval("patchy_cement_moduli", "friction_factor", friction)
    if mode not in ("stiff", "soft"):
        raise RockContractError(
            "unknown patchy-cement mixing mode",
            object_name="patchy_cement_moduli",
            field="mode",
            expected="stiff or soft",
            actual=mode,
        )
    uncemented = hertz_mindlin_moduli(
        pressure,
        k_mineral,
        mu_mineral,
        phi_c,
        coordination,
        friction,
    )
    cemented = contact_cement_moduli(
        k_mineral,
        mu_mineral,
        k_cement,
        mu_cement,
        cement_fraction,
        phi_c,
        coordination,
        scheme=scheme,
    )
    stiff, soft = _patch_endpoints(
        uncemented.k_dry,
        uncemented.mu_dry,
        cemented.k_dry,
        cemented.mu_dry,
        patch_fraction,
    )
    patch = stiff if mode == "stiff" else soft
    return soft_sand_moduli(
        porosity,
        k_mineral,
        mu_mineral,
        patch.k_dry,
        patch.mu_dry,
        phi_c,
    )


def varying_patchy_cement_moduli(
    dilution_fraction: torch.Tensor,
    porosity: TensorInput,
    k_mineral: TensorInput,
    mu_mineral: TensorInput,
    k_cement: TensorInput,
    mu_cement: TensorInput,
    critical_porosity: TensorInput,
    cement_fraction: TensorInput,
    cement_threshold: TensorInput,
    cemented_patch_fraction: TensorInput,
    coordination_number: TensorInput,
    effective_pressure: TensorInput,
    friction_factor: TensorInput = 0.5,
    *,
    scheme: int,
) -> GranularModuli:
    """Return VPCM by blending canonical stiff and soft patch endpoints."""

    values = _validated_inputs(
        "varying_patchy_cement_moduli",
        ("dilution_fraction", dilution_fraction),
        ("porosity", porosity),
        ("k_mineral", k_mineral),
        ("mu_mineral", mu_mineral),
        ("k_cement", k_cement),
        ("mu_cement", mu_cement),
        ("critical_porosity", critical_porosity),
        ("cement_fraction", cement_fraction),
        ("cement_threshold", cement_threshold),
        ("cemented_patch_fraction", cemented_patch_fraction),
        ("coordination_number", coordination_number),
        ("effective_pressure", effective_pressure),
        ("friction_factor", friction_factor),
    )
    (
        dilution,
        porosity,
        k_mineral,
        mu_mineral,
        k_cement,
        mu_cement,
        phi_c,
        cement_fraction,
        cement_threshold,
        patch_fraction,
        coordination,
        pressure,
        friction,
    ) = torch.broadcast_tensors(*values)
    _require_unit_interval("varying_patchy_cement_moduli", "dilution_fraction", dilution)
    _require_unit_interval(
        "varying_patchy_cement_moduli", "cemented_patch_fraction", patch_fraction
    )
    _require_open_porosity(
        "varying_patchy_cement_moduli",
        "critical_porosity",
        phi_c,
    )
    if bool(torch.any((cement_threshold < 0.0) | (cement_threshold >= phi_c))):
        raise RockContractError(
            "cement threshold is outside the VPCM calibrated domain",
            object_name="varying_patchy_cement_moduli",
            field="cement_threshold",
            expected="0 <= cement_threshold < critical_porosity",
            actual=_extrema(cement_threshold),
        )
    uncemented = hertz_mindlin_moduli(
        pressure,
        k_mineral,
        mu_mineral,
        phi_c,
        coordination,
        friction,
    )
    contact_endpoint = contact_cement_moduli(
        k_mineral,
        mu_mineral,
        k_cement,
        mu_cement,
        cement_fraction,
        phi_c,
        coordination,
        scheme=scheme,
    )
    use_increasing_cement = cement_fraction > cement_threshold
    increasing_porosity = torch.where(
        use_increasing_cement,
        phi_c - cement_fraction,
        phi_c - cement_threshold,
    )
    increasing_endpoint = modified_upper_hashin_shtrikman(
        increasing_porosity,
        k_mineral,
        mu_mineral,
        k_cement,
        mu_cement,
        cement_threshold,
        phi_c,
        coordination,
        scheme=scheme,
    )
    cemented = GranularModuli(
        torch.where(
            use_increasing_cement,
            increasing_endpoint.k_dry,
            contact_endpoint.k_dry,
        ),
        torch.where(
            use_increasing_cement,
            increasing_endpoint.mu_dry,
            contact_endpoint.mu_dry,
        ),
    )
    stiff, soft = _patch_endpoints(
        uncemented.k_dry,
        uncemented.mu_dry,
        cemented.k_dry,
        cemented.mu_dry,
        patch_fraction,
    )
    patch = GranularModuli(
        stiff.k_dry * (1.0 - dilution) + soft.k_dry * dilution,
        stiff.mu_dry * (1.0 - dilution) + soft.mu_dry * dilution,
    )
    return soft_sand_moduli(
        porosity,
        k_mineral,
        mu_mineral,
        patch.k_dry,
        patch.mu_dry,
        phi_c,
    )


__all__ = [
    "ConstantCementResult",
    "GranularModuli",
    "constant_cement_moduli",
    "contact_cement_moduli",
    "contact_radius",
    "hertz_mindlin_moduli",
    "modified_upper_hashin_shtrikman",
    "patchy_cement_moduli",
    "soft_sand_moduli",
    "stiff_sand_moduli",
    "varying_patchy_cement_moduli",
    "walton_moduli",
]
