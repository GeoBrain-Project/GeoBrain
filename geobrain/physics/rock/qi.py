# pyright: reportPrivateImportUsage=false
"""
Quantitative Interpretation (QI) helpers.

Canonical tensor functions for screening curves, cement estimation, and
Cartesian rock-physics-template grids.

QI helpers are **forward-only**. They preserve the input tensors' ``device``
and ``dtype`` and vectorize Cartesian template and cement-candidate axes.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from geobrain.physics.rock.contracts import require_compatible_tensors
from geobrain.physics.rock.errors import RockContractError
from geobrain.physics.rock.fluid_substitution import gassmann_saturated_properties
from geobrain.physics.rock.granular import (
    constant_cement_moduli,
    contact_cement_moduli,
    hertz_mindlin_moduli,
    modified_upper_hashin_shtrikman,
    soft_sand_moduli,
)
from geobrain.physics.rock.mixtures import vrh_average, wood_fluid_mix

EPS = 1.0e-12


# --- Private pure-math helpers ---------------------------------------------


@dataclass(frozen=True, slots=True)
class RockPhysicsTemplateResult:
    """One Cartesian rock-physics template in SI units."""

    impedance: Tensor
    vp_vs_ratio: Tensor
    vp: Tensor
    vs: Tensor
    density: Tensor
    fluid_bulk_modulus: Tensor


@dataclass(frozen=True, slots=True)
class QIResourceEstimate:
    """Deterministic tensor-storage estimate for one QI call."""

    output_elements: int
    workspace_elements: int
    bytes_per_element: int

    @property
    def total_bytes(self) -> int:
        return (self.output_elements + self.workspace_elements) * self.bytes_per_element


@dataclass(frozen=True, slots=True)
class ScreeningCurvesResult:
    """Canonical saturated QI screening curves in SI units."""

    porosity: Tensor
    soft_sand_vp: Tensor
    increasing_cement_vp: Tensor
    contact_cement_vp: Tensor
    soft_sand_vs: Tensor
    increasing_cement_vs: Tensor
    contact_cement_vs: Tensor


@dataclass(frozen=True, slots=True)
class CementEstimateResult:
    """Deterministic vectorized cement estimate and live diagnostics."""

    cement_fraction: Tensor
    misfit: Tensor
    candidate_velocities: Tensor
    tie_count: Tensor


def _cartesian_axes(
    porosity: Tensor,
    water_saturation: Tensor,
) -> tuple[Tensor, Tensor]:
    if porosity.ndim != 1 or water_saturation.ndim != 1:
        raise RockContractError(
            "RPT axes must be one-dimensional",
            object_name="rock_physics_template",
            field="axes",
            expected="two rank-1 tensors",
            actual=(tuple(porosity.shape), tuple(water_saturation.shape)),
        )
    if porosity.numel() == 0 or water_saturation.numel() == 0:
        raise RockContractError(
            "RPT axes must be non-empty",
            object_name="rock_physics_template",
            field="axes",
            expected="two non-empty rank-1 tensors",
            actual=(porosity.numel(), water_saturation.numel()),
        )
    porosity_axis, saturation_axis = require_compatible_tensors(
        "rock_physics_template",
        ("porosity", porosity[:, None]),
        ("water_saturation", water_saturation[None, :]),
    )
    return porosity_axis, saturation_axis


def estimate_rock_physics_template_resources(
    porosity: Tensor,
    water_saturation: Tensor,
) -> QIResourceEstimate:
    """Return a stable storage estimate before evaluating an RPT grid."""

    phi, saturation = _cartesian_axes(porosity, water_saturation)
    elements = phi.shape[0] * saturation.shape[1]
    return QIResourceEstimate(
        output_elements=6 * elements,
        workspace_elements=8 * elements,
        bytes_per_element=porosity.element_size(),
    )


def rock_physics_template(
    porosity: Tensor,
    water_saturation: Tensor,
    k_dry: Tensor | float,
    mu_dry: Tensor | float,
    k_mineral: Tensor | float,
    rho_mineral: Tensor | float,
    k_brine: Tensor | float,
    rho_brine: Tensor | float,
    k_hydrocarbon: Tensor | float,
    rho_hydrocarbon: Tensor | float,
) -> RockPhysicsTemplateResult:
    """Build a differentiable Cartesian RPT grid without per-cell loops.

    Args:
        porosity / water_saturation: fractional volume terms.
        k_dry / mu_dry: dry-frame moduli [Pa].
        k_mineral / rho_mineral: mineral modulus [Pa] and density [kg/m^3].
        k_brine / rho_brine: brine modulus [Pa] and density [kg/m^3].
        k_hydrocarbon / rho_hydrocarbon: hydrocarbon modulus [Pa] and
            density [kg/m^3].
    """

    phi, saturation = _cartesian_axes(porosity, water_saturation)
    (
        phi,
        saturation,
        k_dry,
        mu_dry,
        k_mineral,
        rho_mineral,
        k_brine,
        rho_brine,
        k_hydrocarbon,
        rho_hydrocarbon,
    ) = require_compatible_tensors(
        "rock_physics_template",
        ("porosity", phi),
        ("water_saturation", saturation),
        ("k_dry", k_dry),
        ("mu_dry", mu_dry),
        ("k_mineral", k_mineral),
        ("rho_mineral", rho_mineral),
        ("k_brine", k_brine),
        ("rho_brine", rho_brine),
        ("k_hydrocarbon", k_hydrocarbon),
        ("rho_hydrocarbon", rho_hydrocarbon),
    )
    k_fluid, rho_fluid = wood_fluid_mix(
        k_brine,
        k_hydrocarbon,
        rho_brine,
        rho_hydrocarbon,
        saturation,
    )
    saturated = gassmann_saturated_properties(
        k_dry,
        mu_dry,
        k_mineral,
        k_fluid,
        phi,
        rho_mineral,
        rho_fluid,
    )
    return RockPhysicsTemplateResult(
        impedance=saturated.vp * saturated.rho,
        vp_vs_ratio=saturated.vp / saturated.vs,
        vp=saturated.vp,
        vs=saturated.vs,
        density=saturated.rho,
        fluid_bulk_modulus=k_fluid + torch.zeros_like(saturated.rho),
    )


def screening_curves(
    porosity: Tensor,
    effective_pressure: Tensor | float,
    cement_boundary_porosity: Tensor | float,
    k_grain: Tensor | float,
    mu_grain: Tensor | float,
    rho_grain: Tensor | float,
    k_shale: Tensor | float,
    mu_shale: Tensor | float,
    rho_shale: Tensor | float,
    k_cement: Tensor | float,
    mu_cement: Tensor | float,
    rho_cement: Tensor | float,
    k_fluid: Tensor | float,
    rho_fluid: Tensor | float,
    *,
    shale_fraction: Tensor | float = 0.0,
    critical_porosity: Tensor | float = 0.4,
    coordination_number: Tensor | float = 8.6,
    friction_factor: Tensor | float = 0.5,
    scheme: int = 2,
) -> ScreeningCurvesResult:
    """Return soft-sand, increasing-cement and contact-cement curves."""

    if porosity.ndim != 1 or porosity.numel() == 0:
        raise RockContractError(
            "screening porosity axis must be non-empty and one-dimensional",
            object_name="screening_curves",
            field="porosity",
            expected="non-empty rank-1 tensor",
            actual=tuple(porosity.shape),
        )
    (
        porosity,
        pressure,
        boundary,
        k_grain,
        mu_grain,
        rho_grain,
        k_shale,
        mu_shale,
        rho_shale,
        k_cement,
        mu_cement,
        rho_cement,
        k_fluid,
        rho_fluid,
        shale,
        phi_c,
        coordination,
        friction,
    ) = require_compatible_tensors(
        "screening_curves",
        ("porosity", porosity),
        ("effective_pressure", effective_pressure),
        ("cement_boundary_porosity", cement_boundary_porosity),
        ("k_grain", k_grain),
        ("mu_grain", mu_grain),
        ("rho_grain", rho_grain),
        ("k_shale", k_shale),
        ("mu_shale", mu_shale),
        ("rho_shale", rho_shale),
        ("k_cement", k_cement),
        ("mu_cement", mu_cement),
        ("rho_cement", rho_cement),
        ("k_fluid", k_fluid),
        ("rho_fluid", rho_fluid),
        ("shale_fraction", shale_fraction),
        ("critical_porosity", critical_porosity),
        ("coordination_number", coordination_number),
        ("friction_factor", friction_factor),
    )
    if bool(torch.any((porosity <= 0.0) | (porosity >= phi_c))):
        raise RockContractError(
            "screening porosity lies outside the granular-frame domain",
            object_name="screening_curves",
            field="porosity",
            expected="0 < porosity < critical_porosity",
            actual={"minimum": porosity.amin().item(), "maximum": porosity.amax().item()},
        )
    if bool(torch.any((boundary <= 0.0) | (boundary >= phi_c))):
        raise RockContractError(
            "cement boundary lies outside the screening domain",
            object_name="screening_curves",
            field="cement_boundary_porosity",
            expected="0 < boundary < critical_porosity",
            actual={"minimum": boundary.amin().item(), "maximum": boundary.amax().item()},
        )
    k_mineral = vrh_average(k_shale, k_grain, shale)
    mu_mineral = vrh_average(mu_shale, mu_grain, shale)
    rho_mineral = shale * rho_shale + (1.0 - shale) * rho_grain
    contact = hertz_mindlin_moduli(
        pressure,
        k_mineral,
        mu_mineral,
        phi_c,
        coordination,
        friction,
    )
    soft = soft_sand_moduli(
        porosity, k_mineral, mu_mineral, contact.k_dry, contact.mu_dry, phi_c
    )
    boundary_cement = phi_c - boundary
    increasing_valid = porosity <= boundary
    safe_increasing_porosity = torch.where(
        increasing_valid, porosity, torch.zeros_like(porosity)
    )
    increasing = modified_upper_hashin_shtrikman(
        safe_increasing_porosity,
        k_mineral,
        mu_mineral,
        k_cement,
        mu_cement,
        boundary_cement,
        phi_c,
        coordination,
        scheme=scheme,
    )
    contact_curve = contact_cement_moduli(
        k_mineral,
        mu_mineral,
        k_cement,
        mu_cement,
        phi_c - porosity,
        phi_c,
        coordination,
        scheme=scheme,
    )

    increasing_cement_solid_fraction = boundary_cement / (1.0 - boundary)
    increasing_matrix_k = vrh_average(
        k_cement, k_mineral, increasing_cement_solid_fraction
    )
    increasing_matrix_rho = (
        (1.0 - phi_c) * rho_mineral + boundary_cement * rho_cement
    ) / (1.0 - boundary)
    contact_cement_fraction = phi_c - porosity
    contact_cement_solid_fraction = contact_cement_fraction / (1.0 - porosity)
    contact_matrix_k = vrh_average(
        k_cement, k_mineral, contact_cement_solid_fraction
    )
    contact_matrix_rho = (
        (1.0 - phi_c) * rho_mineral + contact_cement_fraction * rho_cement
    ) / (1.0 - porosity)

    def saturated(
        k_dry: Tensor,
        mu_dry: Tensor,
        matrix_k: Tensor,
        matrix_rho: Tensor,
    ) -> tuple[Tensor, Tensor]:
        result = gassmann_saturated_properties(
            k_dry,
            mu_dry,
            matrix_k,
            k_fluid,
            porosity,
            matrix_rho,
            rho_fluid,
        )
        return result.vp, result.vs

    soft_vp, soft_vs = saturated(
        soft.k_dry, soft.mu_dry, k_mineral, rho_mineral
    )
    increasing_vp, increasing_vs = saturated(
        torch.where(increasing_valid, increasing.k_dry, increasing_matrix_k),
        torch.where(increasing_valid, increasing.mu_dry, mu_mineral),
        increasing_matrix_k,
        increasing_matrix_rho,
    )
    contact_vp, contact_vs = saturated(
        contact_curve.k_dry,
        contact_curve.mu_dry,
        contact_matrix_k,
        contact_matrix_rho,
    )
    nan = torch.full_like(porosity, float("nan"))
    return ScreeningCurvesResult(
        porosity=porosity,
        soft_sand_vp=soft_vp,
        increasing_cement_vp=torch.where(increasing_valid, increasing_vp, nan),
        contact_cement_vp=contact_vp,
        soft_sand_vs=soft_vs,
        increasing_cement_vs=torch.where(increasing_valid, increasing_vs, nan),
        contact_cement_vs=contact_vs,
    )


def estimate_cement_fraction(
    observed_vp: Tensor,
    porosity: Tensor,
    cement_candidates: Tensor,
    k_mineral: Tensor | float,
    mu_mineral: Tensor | float,
    rho_mineral: Tensor | float,
    k_cement: Tensor | float,
    mu_cement: Tensor | float,
    k_fluid: Tensor | float,
    rho_fluid: Tensor | float,
    *,
    effective_pressure: Tensor | float = 20.0e6,
    critical_porosity: Tensor | float = 0.4,
    coordination_number: Tensor | float = 8.6,
    friction_factor: Tensor | float = 0.5,
    scheme: int = 2,
) -> CementEstimateResult:
    """Estimate cement volume by vectorized deterministic candidate search."""

    expected = "matching non-empty rank-1 observed_vp and porosity tensors"
    if observed_vp.ndim != 1 or observed_vp.numel() == 0:
        raise RockContractError(
            "observed velocity has an invalid sample axis",
            object_name="estimate_cement_fraction",
            field="observed_vp",
            expected=expected,
            actual=tuple(observed_vp.shape),
        )
    if not bool(torch.isfinite(observed_vp).all()):
        raise RockContractError(
            "observed velocity must contain finite SI values",
            object_name="estimate_cement_fraction",
            field="observed_vp",
            expected="finite values > 0 m/s",
            actual={"shape": list(observed_vp.shape), "value": "non-finite"},
        )
    if bool(torch.any(observed_vp <= 0.0)):
        raise RockContractError(
            "observed velocity must be positive",
            object_name="estimate_cement_fraction",
            field="observed_vp",
            expected="finite values > 0 m/s",
            actual={"minimum": observed_vp.amin().item(), "unit": "m/s"},
        )
    if porosity.ndim != 1 or porosity.shape != observed_vp.shape:
        raise RockContractError(
            "porosity has an invalid sample axis",
            object_name="estimate_cement_fraction",
            field="porosity",
            expected=expected,
            actual=tuple(porosity.shape),
        )
    if cement_candidates.ndim != 1 or cement_candidates.numel() == 0:
        raise RockContractError(
            "cement candidates must form a non-empty rank-1 axis",
            object_name="estimate_cement_fraction",
            field="cement_candidates",
            expected="strictly increasing values in 0 <= value < critical_porosity",
            actual=tuple(cement_candidates.shape),
        )
    sample = observed_vp[:, None]
    phi = porosity[:, None]
    candidates = cement_candidates[None, :]

    def candidate_aligned(value: Tensor | float) -> Tensor | float:
        if isinstance(value, Tensor) and value.ndim == 1 and value.shape == observed_vp.shape:
            return value[:, None]
        return value

    k_mineral = candidate_aligned(k_mineral)
    mu_mineral = candidate_aligned(mu_mineral)
    rho_mineral = candidate_aligned(rho_mineral)
    k_cement = candidate_aligned(k_cement)
    mu_cement = candidate_aligned(mu_cement)
    k_fluid = candidate_aligned(k_fluid)
    rho_fluid = candidate_aligned(rho_fluid)
    effective_pressure = candidate_aligned(effective_pressure)
    coordination_number = candidate_aligned(coordination_number)
    friction_factor = candidate_aligned(friction_factor)
    (
        sample,
        phi,
        candidates,
        k_mineral,
        mu_mineral,
        rho_mineral,
        k_cement,
        mu_cement,
        k_fluid,
        rho_fluid,
        pressure,
        phi_c,
        coordination,
        friction,
    ) = require_compatible_tensors(
        "estimate_cement_fraction",
        ("observed_vp", sample),
        ("porosity", phi),
        ("cement_candidates", candidates),
        ("k_mineral", k_mineral),
        ("mu_mineral", mu_mineral),
        ("rho_mineral", rho_mineral),
        ("k_cement", k_cement),
        ("mu_cement", mu_cement),
        ("k_fluid", k_fluid),
        ("rho_fluid", rho_fluid),
        ("effective_pressure", effective_pressure),
        ("critical_porosity", critical_porosity),
        ("coordination_number", coordination_number),
        ("friction_factor", friction_factor),
    )
    if not bool(torch.isfinite(candidates).all()) or bool(torch.any(candidates < 0.0)):
        raise RockContractError(
            "cement candidate axis contains invalid values",
            object_name="estimate_cement_fraction",
            field="cement_candidates",
            expected="strictly increasing finite values in 0 <= value < critical_porosity",
            actual="invalid candidate value(s)",
        )
    if cement_candidates.numel() > 1 and bool(torch.any(torch.diff(cement_candidates) <= 0.0)):
        raise RockContractError(
            "cement candidates are not strictly increasing",
            object_name="estimate_cement_fraction",
            field="cement_candidates",
            expected="strictly increasing values",
            actual="non-increasing candidate value(s)",
        )
    if bool(torch.any(candidates >= phi_c)):
        raise RockContractError(
            "cement candidate reaches or exceeds critical porosity",
            object_name="estimate_cement_fraction",
            field="cement_candidates",
            expected="candidate < critical_porosity",
            actual={"maximum": candidates.amax().item()},
        )
    valid = phi <= phi_c - candidates
    safe_phi = torch.where(valid, phi, torch.zeros_like(phi))
    contact = hertz_mindlin_moduli(
        pressure,
        k_mineral,
        mu_mineral,
        phi_c,
        coordination,
        friction,
    )
    uncemented = soft_sand_moduli(
        safe_phi,
        k_mineral,
        mu_mineral,
        contact.k_dry,
        contact.mu_dry,
        phi_c,
    )
    positive_replacement = phi_c * candidates.new_tensor(
        64.0 * torch.finfo(candidates.dtype).eps
    )
    safe_candidates = torch.where(candidates == 0.0, positive_replacement, candidates)
    cemented = constant_cement_moduli(
        safe_phi,
        k_mineral,
        mu_mineral,
        k_cement,
        mu_cement,
        safe_candidates,
        phi_c,
        coordination,
        scheme=scheme,
    )
    zero = candidates == 0.0
    k_dry = torch.where(zero, uncemented.k_dry, cemented.k_dry)
    mu_dry = torch.where(zero, uncemented.mu_dry, cemented.mu_dry)
    k_dry = torch.where(valid, k_dry, k_mineral + torch.zeros_like(k_dry))
    mu_dry = torch.where(valid, mu_dry, mu_mineral + torch.zeros_like(mu_dry))
    saturated = gassmann_saturated_properties(
        k_dry,
        mu_dry,
        k_mineral,
        k_fluid,
        safe_phi,
        rho_mineral,
        rho_fluid,
    )
    candidate_vp = torch.where(valid, saturated.vp, torch.full_like(saturated.vp, float("nan")))
    absolute_misfit = torch.where(
        valid,
        (candidate_vp - sample).abs(),
        torch.full_like(candidate_vp, float("inf")),
    )
    no_valid_candidate = ~valid.any(dim=1)
    if bool(torch.any(no_valid_candidate)):
        raise RockContractError(
            "no cement candidate is valid for one or more samples",
            object_name="estimate_cement_fraction",
            field="porosity",
            expected="at least one candidate with porosity <= critical_porosity - candidate",
            actual={"invalid_sample_count": int(no_valid_candidate.sum().item())},
        )
    adjacent_valid = valid[:, 1:] & valid[:, :-1]
    non_increasing = adjacent_valid & (candidate_vp[:, 1:] <= candidate_vp[:, :-1])
    if bool(torch.any(non_increasing)):
        raise RockContractError(
            "cement candidate velocity curves are not strictly increasing",
            object_name="estimate_cement_fraction",
            field="candidate_velocities",
            expected="strictly increasing velocity over valid cement candidates",
            actual={"non_increasing_pair_count": int(non_increasing.sum().item())},
        )
    upper_mask = valid & (candidate_vp >= sample)
    has_upper = upper_mask.any(dim=1)
    first_upper = upper_mask.to(torch.int64).argmax(dim=1)
    valid_count = valid.sum(dim=1)
    last_valid = valid_count - 1
    upper_index = torch.where(has_upper, first_upper, last_valid)
    lower_index = torch.where(upper_index > 0, upper_index - 1, upper_index)
    lower_velocity = candidate_vp.gather(1, lower_index[:, None]).squeeze(1)
    upper_velocity = candidate_vp.gather(1, upper_index[:, None]).squeeze(1)
    candidate_grid = candidates.expand_as(candidate_vp)
    lower_cement = candidate_grid.gather(1, lower_index[:, None]).squeeze(1)
    upper_cement = candidate_grid.gather(1, upper_index[:, None]).squeeze(1)
    denominator = upper_velocity - lower_velocity
    safe_denominator = torch.where(
        denominator == 0.0, torch.ones_like(denominator), denominator
    )
    interpolation = (sample.squeeze(1) - lower_velocity) / safe_denominator
    interior = has_upper & (upper_index > 0)
    selected = torch.where(
        interior,
        lower_cement + interpolation * (upper_cement - lower_cement),
        upper_cement,
    )
    predicted = torch.where(
        interior,
        lower_velocity + interpolation * denominator,
        upper_velocity,
    )
    selected_misfit = (predicted - sample.squeeze(1)).abs()
    nearest_misfit = absolute_misfit.amin(dim=1)
    tolerance = sample.squeeze(1).abs() * sample.new_tensor(
        8.0 * torch.finfo(sample.dtype).eps
    )
    ties = (absolute_misfit <= nearest_misfit[:, None] + tolerance[:, None]).sum(dim=1)
    return CementEstimateResult(
        cement_fraction=selected,
        misfit=selected_misfit,
        candidate_velocities=candidate_vp,
        tie_count=ties,
    )


def _matrix_density_pure(
    vsh: Tensor,
    phi_c: Tensor,
    phi_0: Tensor,
    rho_sh: Tensor,
    rho_grain: Tensor,
    rho_cem: Tensor,
) -> Tensor:
    """Return cementing-matrix density without changing dtype or device."""

    cement_fraction = phi_c - phi_0
    denominator = 1.0 - phi_0 + phi_0.new_tensor(EPS)
    shale_fraction = vsh / denominator
    grain_fraction = (1.0 - phi_c - vsh) / denominator
    cement_fraction = cement_fraction / denominator
    return shale_fraction * rho_sh + grain_fraction * rho_grain + cement_fraction * rho_cem


def _sample_aligned_tensor(
    reference: Tensor,
    field: str,
    value: Tensor,
) -> Tensor:
    """Broadcast one scalar or per-sample value onto the QI sample axis."""

    try:
        return torch.broadcast_to(value, reference.shape)
    except RuntimeError as error:
        raise RockContractError(
            "cement-estimator input is not aligned with the sample axis",
            object_name="CementEstimator",
            field=field,
            expected=f"scalar or shape {list(reference.shape)}",
            actual={"shape": list(value.shape)},
        ) from error


__all__ = [
    "CementEstimateResult",
    "QIResourceEstimate",
    "RockPhysicsTemplateResult",
    "ScreeningCurvesResult",
    "estimate_rock_physics_template_resources",
    "estimate_cement_fraction",
    "rock_physics_template",
    "screening_curves",
]
