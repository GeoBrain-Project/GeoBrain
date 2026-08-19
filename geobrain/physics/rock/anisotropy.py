"""Canonical SI anisotropic elasticity and crack kernels.

All stiffnesses are in Pa, compliances in Pa⁻¹, densities in kg/m³, and
velocities in m/s.  Voigt order is ``11, 22, 33, 23, 13, 12`` with
engineering shear components.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import TypeAlias, cast

import torch

from geobrain.core import ErrorCode

from .contracts import require_compatible_tensors
from .errors import RockContractError, RockNumericsError

TensorInput: TypeAlias = torch.Tensor | int | float


@dataclass(frozen=True, slots=True)
class AnisotropicPhaseResult:
    """Vertical phase properties derived from a VTI stiffness tensor."""

    stiffness: torch.Tensor
    vp0: torch.Tensor
    vs0: torch.Tensor
    epsilon: torch.Tensor
    delta: torch.Tensor
    gamma: torch.Tensor


@dataclass(frozen=True, slots=True)
class BackusResult:
    """Backus-equivalent VTI stiffness, density, and vertical properties."""

    stiffness: torch.Tensor
    rho: torch.Tensor
    vp0: torch.Tensor
    vs0: torch.Tensor
    epsilon: torch.Tensor
    delta: torch.Tensor
    gamma: torch.Tensor


def _require_materialized_strided_tensor(
    object_name: str,
    field: str,
    tensor: torch.Tensor,
) -> None:
    if tensor.layout is not torch.strided:
        raise RockContractError(
            "anisotropy kernels require strided tensors",
            object_name=object_name,
            field=field,
            expected="torch.strided layout",
            actual=str(tensor.layout),
        )
    if tensor.device.type == "meta":
        raise RockContractError(
            "anisotropy kernels require materialized tensors",
            object_name=object_name,
            field=field,
            expected="materialized CPU or accelerator tensor",
            actual=str(tensor.device),
            code=ErrorCode.DEVICE_UNAVAILABLE,
        )


def _validated_scalars(
    object_name: str,
    *fields: tuple[str, TensorInput],
) -> tuple[torch.Tensor, ...]:
    reference_name, reference = fields[0]
    if not isinstance(reference, torch.Tensor):
        raise RockContractError(
            "anisotropy call requires its documented reference tensor",
            object_name=object_name,
            field=reference_name,
            expected="torch.Tensor reference",
            actual=type(reference).__qualname__,
        )
    _require_materialized_strided_tensor(object_name, reference_name, reference)
    tensors = cast(
        tuple[torch.Tensor, ...],
        require_compatible_tensors(object_name, *fields),
    )
    for (name, _), tensor in zip(fields, tensors, strict=True):
        _require_materialized_strided_tensor(object_name, name, tensor)
        if not bool(torch.isfinite(tensor).all()):
            raise RockContractError(
                "anisotropy input must be finite",
                object_name=object_name,
                field=name,
                expected="finite SI values",
                actual="non-finite value(s)",
            )
    return cast(tuple[torch.Tensor, ...], torch.broadcast_tensors(*tensors))


def _legacy_degrees_to_radians(
    reference: torch.Tensor,
    angle_degrees: TensorInput,
    *,
    object_name: str,
    field: str,
) -> torch.Tensor:
    """Validate one legacy degree value before explicitly converting it."""

    return _validated_scalars(
        object_name,
        ("reference", reference),
        (field, angle_degrees),
    )[1] * (math.pi / 180.0)


def _extrema(value: torch.Tensor, unit: str) -> dict[str, object]:
    return {
        "minimum": value.amin().item(),
        "maximum": value.amax().item(),
        "unit": unit,
    }


def _require_positive(object_name: str, field: str, value: torch.Tensor, unit: str) -> None:
    if bool(torch.any(value <= 0.0)):
        raise RockContractError(
            "anisotropy input must be positive",
            object_name=object_name,
            field=field,
            expected=f"> 0 {unit}",
            actual=_extrema(value, unit),
        )


def _require_nonnegative(
    object_name: str,
    field: str,
    value: torch.Tensor,
    unit: str,
) -> None:
    if bool(torch.any(value < 0.0)):
        raise RockContractError(
            "anisotropy input must be non-negative",
            object_name=object_name,
            field=field,
            expected=f">= 0 {unit}",
            actual=_extrema(value, unit),
        )


def _matrix_rows(*rows: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.stack(tuple(torch.stack(row, dim=-1) for row in rows), dim=-2)


def _vti_matrix(
    c11: torch.Tensor,
    c33: torch.Tensor,
    c13: torch.Tensor,
    c44: torch.Tensor,
    c66: torch.Tensor,
    *,
    axis: int,
) -> torch.Tensor:
    zero = torch.zeros_like(c11)
    if axis == 3:
        c12 = c11 - 2.0 * c66
        return _matrix_rows(
            (c11, c12, c13, zero, zero, zero),
            (c12, c11, c13, zero, zero, zero),
            (c13, c13, c33, zero, zero, zero),
            (zero, zero, zero, c44, zero, zero),
            (zero, zero, zero, zero, c44, zero),
            (zero, zero, zero, zero, zero, c66),
        )
    if axis == 1:
        c23 = c11 - 2.0 * c66
        return _matrix_rows(
            (c33, c13, c13, zero, zero, zero),
            (c13, c11, c23, zero, zero, zero),
            (c13, c23, c11, zero, zero, zero),
            (zero, zero, zero, c66, zero, zero),
            (zero, zero, zero, zero, c44, zero),
            (zero, zero, zero, zero, zero, c44),
        )
    raise RockContractError(
        "Hudson symmetry axis is unsupported",
        object_name="hudson_stiffness",
        field="axis",
        expected="1 or 3",
        actual=axis,
    )


def _validate_matrix(
    matrix: torch.Tensor,
    *,
    object_name: str,
    field: str,
    positive_definite: bool,
) -> torch.Tensor:
    if not isinstance(matrix, torch.Tensor):
        raise RockContractError(
            "anisotropy matrix must be a tensor",
            object_name=object_name,
            field=field,
            expected="torch.Tensor[..., 6, 6]",
            actual=type(matrix).__qualname__,
        )
    if matrix.dtype not in (torch.float32, torch.float64):
        raise RockContractError(
            "anisotropy matrix dtype is unsupported",
            object_name=object_name,
            field=field,
            expected="torch.float32 or torch.float64",
            actual=str(matrix.dtype),
            code=ErrorCode.DTYPE_UNSUPPORTED,
        )
    if matrix.layout is not torch.strided or matrix.device.type == "meta":
        raise RockContractError(
            "anisotropy matrix must be materialized and strided",
            object_name=object_name,
            field=field,
            expected="materialized strided tensor",
            actual={"layout": str(matrix.layout), "device": str(matrix.device)},
        )
    if matrix.ndim < 2 or matrix.shape[-2:] != (6, 6):
        raise RockContractError(
            "anisotropy matrix has the wrong Voigt shape",
            object_name=object_name,
            field=field,
            expected="(..., 6, 6)",
            actual=tuple(matrix.shape),
            code=ErrorCode.SHAPE_MISMATCH,
        )
    if not bool(torch.isfinite(matrix).all()):
        raise RockContractError(
            "anisotropy matrix must be finite",
            object_name=object_name,
            field=field,
            expected="finite entries",
            actual="non-finite value(s)",
        )
    epsilon = torch.finfo(matrix.dtype).eps
    scale = matrix.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(matrix.dtype).tiny)
    asymmetry = (matrix - matrix.transpose(-1, -2)).abs().amax(dim=(-2, -1), keepdim=True)
    if bool(torch.any(asymmetry > (64.0 * epsilon) * scale)):
        raise RockContractError(
            "anisotropy matrix must satisfy major symmetry",
            object_name=object_name,
            field=field,
            expected="matrix == matrix.T within dtype tolerance",
            actual={"maximum_scaled_asymmetry": (asymmetry / scale).amax().item()},
        )
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    if positive_definite:
        _, info = torch.linalg.cholesky_ex(symmetric)
        if bool(torch.any(info != 0)):
            raise RockNumericsError(
                "anisotropy matrix is not positive definite",
                object_name=object_name,
                field=field,
                expected="positive-definite calibrated stiffness/compliance",
                actual={"failed_batches": int((info != 0).sum().item())},
            )
    return symmetric


def isotropic_stiffness(k_iso: torch.Tensor, mu_iso: TensorInput) -> torch.Tensor:
    """Return a broadcast isotropic ``(..., 6, 6)`` stiffness in Pa."""
    k_iso, mu_iso = _validated_scalars("isotropic_stiffness", ("k_iso", k_iso), ("mu_iso", mu_iso))
    _require_positive("isotropic_stiffness", "k_iso", k_iso, "Pa")
    _require_positive("isotropic_stiffness", "mu_iso", mu_iso, "Pa")
    lam = k_iso - 2.0 * mu_iso / 3.0
    c11 = lam + 2.0 * mu_iso
    return _vti_matrix(c11, c11, lam, mu_iso, mu_iso, axis=3)


def compliance_from_stiffness(stiffness: torch.Tensor) -> torch.Tensor:
    """Return compliance in Pa⁻¹ using a dtype-scaled linear solve."""
    stiffness = _validate_matrix(
        stiffness,
        object_name="compliance_from_stiffness",
        field="stiffness",
        positive_definite=True,
    )
    scale = stiffness.abs().amax(dim=(-2, -1), keepdim=True)
    identity = torch.eye(6, dtype=stiffness.dtype, device=stiffness.device)
    identity = identity.expand(stiffness.shape)
    return torch.linalg.solve(stiffness / scale, identity) / scale


def stiffness_from_compliance(compliance: torch.Tensor) -> torch.Tensor:
    """Return stiffness in Pa using a dtype-scaled linear solve."""
    compliance = _validate_matrix(
        compliance,
        object_name="stiffness_from_compliance",
        field="compliance",
        positive_definite=True,
    )
    scale = compliance.abs().amax(dim=(-2, -1), keepdim=True)
    identity = torch.eye(6, dtype=compliance.dtype, device=compliance.device)
    identity = identity.expand(compliance.shape)
    stiffness = torch.linalg.solve(compliance / scale, identity) / scale
    return 0.5 * (stiffness + stiffness.transpose(-1, -2))


def isotropic_compliance(k_iso: torch.Tensor, mu_iso: TensorInput) -> torch.Tensor:
    """Return isotropic compliance in Pa⁻¹."""
    return compliance_from_stiffness(isotropic_stiffness(k_iso, mu_iso))


def isotropic_moduli_from_compliance(
    compliance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return isotropic ``(K, mu)`` in Pa from an isotropic compliance."""
    stiffness = stiffness_from_compliance(compliance)
    k_iso = stiffness[..., :3, :3].sum(dim=(-2, -1)) / 9.0
    mu_iso = torch.diagonal(stiffness[..., 3:, 3:], dim1=-2, dim2=-1).mean(dim=-1)
    return k_iso, mu_iso


def _hudson_factors(
    k_iso: torch.Tensor,
    mu_iso: torch.Tensor,
    aspect_ratio: torch.Tensor,
    k_fluid: torch.Tensor,
    mu_fluid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lam = k_iso - 2.0 * mu_iso / 3.0
    p_wave = lam + 2.0 * mu_iso
    kappa = (
        (k_fluid + 4.0 * mu_fluid / 3.0)
        * p_wave
        / (math.pi * aspect_ratio * mu_iso * (lam + mu_iso))
    )
    shear_correction = (
        4.0 * mu_fluid * p_wave / (math.pi * aspect_ratio * mu_iso * (3.0 * lam + 4.0 * mu_iso))
    )
    u3 = 4.0 * p_wave / (3.0 * (lam + mu_iso) * (1.0 + kappa))
    u1 = 16.0 * p_wave / (3.0 * (3.0 * lam + 4.0 * mu_iso) * (1.0 + shear_correction))
    return lam, p_wave, u1, u3


def _hudson_components(
    k_iso: torch.Tensor,
    mu_iso: torch.Tensor,
    crack_density: torch.Tensor,
    aspect_ratio: torch.Tensor,
    k_fluid: torch.Tensor,
    mu_fluid: torch.Tensor,
    *,
    order: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lam, p_wave, u1, u3 = _hudson_factors(k_iso, mu_iso, aspect_ratio, k_fluid, mu_fluid)
    c11_delta = -(lam.square() / mu_iso) * crack_density * u3
    c13_delta = -(lam * p_wave / mu_iso) * crack_density * u3
    c33_delta = -(p_wave.square() / mu_iso) * crack_density * u3
    c44_delta = -mu_iso * crack_density * u1
    if order == 2:
        q_value = 15.0 * (lam / mu_iso).square() + 28.0 * lam / mu_iso + 28.0
        density_u3 = crack_density * u3
        density_u1 = crack_density * u1
        c11_delta = c11_delta + (q_value / 15.0 * lam.square() / p_wave * density_u3.square())
        c13_delta = c13_delta + q_value / 15.0 * lam * density_u3.square()
        c33_delta = c33_delta + q_value / 15.0 * p_wave * density_u3.square()
        c44_delta = c44_delta + (
            2.0 / 15.0 * mu_iso * (3.0 * lam + 8.0 * mu_iso) / p_wave * density_u1.square()
        )
    return (
        p_wave + c11_delta,
        p_wave + c33_delta,
        lam + c13_delta,
        mu_iso + c44_delta,
        mu_iso,
    )


def hudson_stiffness(
    k_iso: torch.Tensor,
    mu_iso: TensorInput,
    crack_density: TensorInput,
    aspect_ratio: TensorInput,
    *,
    k_fluid: TensorInput = 0.0,
    mu_fluid: TensorInput = 0.0,
    axis: int = 3,
    order: int = 1,
) -> torch.Tensor:
    """Return Hudson aligned-crack stiffness in Pa."""
    if not isinstance(axis, Integral) or isinstance(axis, bool) or int(axis) not in (1, 3):
        raise RockContractError(
            "Hudson symmetry axis is unsupported",
            object_name="hudson_stiffness",
            field="axis",
            expected="1 or 3",
            actual=axis,
        )
    if not isinstance(order, Integral) or isinstance(order, bool) or int(order) not in (1, 2):
        raise RockContractError(
            "Hudson perturbation order is unsupported",
            object_name="hudson_stiffness",
            field="order",
            expected="1 or 2",
            actual=order,
        )
    k_iso, mu_iso, density, alpha, k_fluid, mu_fluid = _validated_scalars(
        "hudson_stiffness",
        ("k_iso", k_iso),
        ("mu_iso", mu_iso),
        ("crack_density", crack_density),
        ("aspect_ratio", aspect_ratio),
        ("k_fluid", k_fluid),
        ("mu_fluid", mu_fluid),
    )
    _require_positive("hudson_stiffness", "k_iso", k_iso, "Pa")
    _require_positive("hudson_stiffness", "mu_iso", mu_iso, "Pa")
    _require_nonnegative("hudson_stiffness", "crack_density", density, "1")
    _require_positive("hudson_stiffness", "aspect_ratio", alpha, "1")
    if bool(torch.any(alpha > 1.0)):
        raise RockContractError(
            "Hudson penny-crack aspect ratio exceeds one",
            object_name="hudson_stiffness",
            field="aspect_ratio",
            expected="0 < aspect_ratio <= 1",
            actual=_extrema(alpha, "1"),
        )
    _require_nonnegative("hudson_stiffness", "k_fluid", k_fluid, "Pa")
    _require_nonnegative("hudson_stiffness", "mu_fluid", mu_fluid, "Pa")
    components = _hudson_components(
        k_iso,
        mu_iso,
        density,
        alpha,
        k_fluid,
        mu_fluid,
        order=int(order),
    )
    stiffness = _vti_matrix(*components, axis=int(axis))
    return _validate_matrix(
        stiffness,
        object_name="hudson_stiffness",
        field="stiffness",
        positive_definite=True,
    )


def thomsen_properties_from_stiffness(
    stiffness: torch.Tensor,
    rho: TensorInput,
) -> AnisotropicPhaseResult:
    """Return vertical velocities and Thomsen parameters from VTI stiffness."""
    stiffness = _validate_matrix(
        stiffness,
        object_name="thomsen_properties_from_stiffness",
        field="stiffness",
        positive_definite=True,
    )
    rho_tensor = _validated_scalars(
        "thomsen_properties_from_stiffness",
        ("matrix_batch", stiffness[..., 0, 0]),
        ("rho", rho),
    )[1]
    batch_shape = torch.broadcast_shapes(stiffness.shape[:-2], rho_tensor.shape)
    stiffness = stiffness.expand(batch_shape + (6, 6))
    rho_tensor = rho_tensor.expand(batch_shape)
    _require_positive("thomsen_properties_from_stiffness", "rho", rho_tensor, "kg/m^3")
    c11 = stiffness[..., 0, 0]
    c13 = stiffness[..., 0, 2]
    c33 = stiffness[..., 2, 2]
    c44 = stiffness[..., 3, 3]
    c66 = stiffness[..., 5, 5]
    denominator = 2.0 * c33 * (c33 - c44)
    if bool(torch.any(denominator == 0.0)):
        raise RockNumericsError(
            "Thomsen delta denominator is singular",
            object_name="thomsen_properties_from_stiffness",
            field="delta_denominator",
            expected="non-zero Pa^2",
            actual=_extrema(denominator, "Pa^2"),
        )
    return AnisotropicPhaseResult(
        stiffness=stiffness,
        vp0=torch.sqrt(c33 / rho_tensor),
        vs0=torch.sqrt(c44 / rho_tensor),
        epsilon=(c11 - c33) / (2.0 * c33),
        delta=((c13 + c44).square() - (c33 - c44).square()) / denominator,
        gamma=(c66 - c44) / (2.0 * c44),
    )


def vti_stiffness_from_thomsen(
    vp0: torch.Tensor,
    vs0: TensorInput,
    epsilon: TensorInput,
    delta: TensorInput,
    gamma: TensorInput,
    rho: TensorInput,
) -> torch.Tensor:
    """Return exact VTI stiffness from vertical velocities and Thomsen values."""
    vp0, vs0, epsilon, delta, gamma, rho = _validated_scalars(
        "vti_stiffness_from_thomsen",
        ("vp0", vp0),
        ("vs0", vs0),
        ("epsilon", epsilon),
        ("delta", delta),
        ("gamma", gamma),
        ("rho", rho),
    )
    _require_positive("vti_stiffness_from_thomsen", "vp0", vp0, "m/s")
    _require_positive("vti_stiffness_from_thomsen", "vs0", vs0, "m/s")
    _require_positive("vti_stiffness_from_thomsen", "rho", rho, "kg/m^3")
    c33 = rho * vp0.square()
    c44 = rho * vs0.square()
    c11 = c33 * (1.0 + 2.0 * epsilon)
    c66 = c44 * (1.0 + 2.0 * gamma)
    discriminant = 2.0 * delta * c33 * (c33 - c44) + (c33 - c44).square()
    if bool(torch.any(discriminant < 0.0)):
        raise RockContractError(
            "Thomsen parameters imply a negative stiffness discriminant",
            object_name="vti_stiffness_from_thomsen",
            field="delta",
            expected="non-negative C13 discriminant",
            actual=_extrema(discriminant, "Pa^2"),
        )
    c13 = torch.sqrt(discriminant) - c44
    stiffness = _vti_matrix(c11, c33, c13, c44, c66, axis=3)
    return _validate_matrix(
        stiffness,
        object_name="vti_stiffness_from_thomsen",
        field="stiffness",
        positive_definite=True,
    )


def thomsen_phase_velocities(
    vp0: torch.Tensor,
    vs0: TensorInput,
    epsilon: TensorInput,
    delta: TensorInput,
    gamma: TensorInput,
    angles_radians: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weak-anisotropy ``(vp, vsv, vsh)`` over a one-dimensional angle axis."""
    vp0, vs0, epsilon, delta, gamma = _validated_scalars(
        "thomsen_phase_velocities",
        ("vp0", vp0),
        ("vs0", vs0),
        ("epsilon", epsilon),
        ("delta", delta),
        ("gamma", gamma),
    )
    _require_positive("thomsen_phase_velocities", "vp0", vp0, "m/s")
    _require_positive("thomsen_phase_velocities", "vs0", vs0, "m/s")
    if not isinstance(angles_radians, torch.Tensor):
        raise RockContractError(
            "Thomsen angles must be a tensor",
            object_name="thomsen_phase_velocities",
            field="angles_radians",
            expected="one-dimensional tensor in radians",
            actual=type(angles_radians).__qualname__,
        )
    _require_materialized_strided_tensor(
        "thomsen_phase_velocities",
        "angles_radians",
        angles_radians,
    )
    if angles_radians.dtype != vp0.dtype or angles_radians.device != vp0.device:
        raise RockContractError(
            "Thomsen angle dtype/device must match the velocity reference",
            object_name="thomsen_phase_velocities",
            field="angles_radians",
            expected={"dtype": str(vp0.dtype), "device": str(vp0.device)},
            actual={"dtype": str(angles_radians.dtype), "device": str(angles_radians.device)},
        )
    if angles_radians.ndim != 1 or not bool(torch.isfinite(angles_radians).all()):
        raise RockContractError(
            "Thomsen angles must be a finite one-dimensional tensor",
            object_name="thomsen_phase_velocities",
            field="angles_radians",
            expected="finite shape (angles,)",
            actual={"shape": tuple(angles_radians.shape)},
        )
    sin2 = torch.sin(angles_radians).square()
    cos2 = torch.cos(angles_radians).square()
    expand = (None,) * vp0.ndim
    sin2 = sin2[expand]
    cos2 = cos2[expand]
    vp0 = vp0[..., None]
    vs0 = vs0[..., None]
    epsilon = epsilon[..., None]
    delta = delta[..., None]
    gamma = gamma[..., None]
    vp = vp0 * (1.0 + delta * sin2 * cos2 + epsilon * sin2.square())
    vsv = vs0 * (1.0 + (vp0 / vs0).square() * (epsilon - delta) * sin2 * cos2)
    vsh = vs0 * (1.0 + gamma * sin2)
    return vp, vsv, vsh


def hudson_phase_properties(
    k_iso: torch.Tensor,
    mu_iso: TensorInput,
    rho: TensorInput,
    crack_density: TensorInput,
    aspect_ratio: TensorInput,
    *,
    k_fluid: TensorInput = 0.0,
    mu_fluid: TensorInput = 0.0,
    axis: int = 3,
    order: int = 1,
) -> AnisotropicPhaseResult:
    """Derive Hudson phase properties from the canonical stiffness tensor.

    Args:
        k_iso: uncracked isotropic bulk modulus [Pa].
        mu_iso: uncracked shear modulus [Pa].
        rho: bulk density [kg/m^3].
        crack_density: dimensionless crack density.
        aspect_ratio: crack aspect ratio.
        k_fluid / mu_fluid: crack-fill moduli [Pa].
        axis: symmetry axis index (Voigt convention).
        order: 1 for first-order Hudson, 2 adds the second-order term.
    """
    stiffness = hudson_stiffness(
        k_iso,
        mu_iso,
        crack_density,
        aspect_ratio,
        k_fluid=k_fluid,
        mu_fluid=mu_fluid,
        axis=axis,
        order=order,
    )
    if axis != 3:
        raise RockContractError(
            "Thomsen phase view requires a VTI Hudson tensor",
            object_name="hudson_phase_properties",
            field="axis",
            expected="3",
            actual=axis,
        )
    return thomsen_properties_from_stiffness(stiffness, rho)


def hudson_random_moduli(
    k_iso: torch.Tensor,
    mu_iso: TensorInput,
    crack_density: TensorInput,
    aspect_ratio: TensorInput,
    *,
    k_fluid: TensorInput = 0.0,
    mu_fluid: TensorInput = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Hudson's first-order random-orientation isotropic moduli."""
    k_iso, mu_iso, density, alpha, k_fluid, mu_fluid = _validated_scalars(
        "hudson_random_moduli",
        ("k_iso", k_iso),
        ("mu_iso", mu_iso),
        ("crack_density", crack_density),
        ("aspect_ratio", aspect_ratio),
        ("k_fluid", k_fluid),
        ("mu_fluid", mu_fluid),
    )
    # Reuse aligned-kernel validation before applying the published average.
    hudson_stiffness(
        k_iso,
        mu_iso,
        density,
        alpha,
        k_fluid=k_fluid,
        mu_fluid=mu_fluid,
    )
    lam, _, u1, u3 = _hudson_factors(k_iso, mu_iso, alpha, k_fluid, mu_fluid)
    mu_delta = -(2.0 / 15.0) * mu_iso * density * (3.0 * u1 + 2.0 * u3)
    lam_delta = (
        -((3.0 * lam + 2.0 * mu_iso).square()) * density * u3 / (3.0 * mu_iso) - 2.0 * mu_delta
    ) / 3.0
    mu_eff = mu_iso + mu_delta
    k_eff = lam + lam_delta + 2.0 * mu_eff / 3.0
    if bool(torch.any((k_eff <= 0.0) | (mu_eff <= 0.0))):
        raise RockNumericsError(
            "Hudson random-orientation moduli left the positive domain",
            object_name="hudson_random_moduli",
            field="effective_moduli",
            expected="K_eff > 0 Pa and mu_eff > 0 Pa",
            actual={"k_eff": _extrema(k_eff, "Pa"), "mu_eff": _extrema(mu_eff, "Pa")},
        )
    return k_eff, mu_eff


def hudson_orthogonal_stiffness(
    k_iso: torch.Tensor,
    mu_iso: TensorInput,
    crack_density: torch.Tensor,
    aspect_ratio: torch.Tensor,
    *,
    k_fluid: TensorInput = 0.0,
    mu_fluid: TensorInput = 0.0,
) -> torch.Tensor:
    """Return stiffness for three orthogonal Hudson crack sets."""
    for name, value in (
        ("crack_density", crack_density),
        ("aspect_ratio", aspect_ratio),
    ):
        if not isinstance(value, torch.Tensor):
            raise RockContractError(
                "orthogonal Hudson crack-set input must be a tensor",
                object_name="hudson_orthogonal_stiffness",
                field=name,
                expected="torch.Tensor[..., 3]",
                actual=type(value).__qualname__,
            )
        _require_materialized_strided_tensor(
            "hudson_orthogonal_stiffness",
            name,
            value,
        )
    if crack_density.ndim < 1 or crack_density.shape[-1] != 3:
        raise RockContractError(
            "orthogonal Hudson crack density needs three crack sets",
            object_name="hudson_orthogonal_stiffness",
            field="crack_density",
            expected="shape (..., 3)",
            actual=tuple(crack_density.shape),
            code=ErrorCode.SHAPE_MISMATCH,
        )
    if aspect_ratio.ndim < 1 or aspect_ratio.shape[-1] != 3:
        raise RockContractError(
            "orthogonal Hudson aspect ratio needs three crack sets",
            object_name="hudson_orthogonal_stiffness",
            field="aspect_ratio",
            expected="shape (..., 3)",
            actual=tuple(aspect_ratio.shape),
            code=ErrorCode.SHAPE_MISMATCH,
        )
    k_iso, mu_iso, k_fluid, mu_fluid = _validated_scalars(
        "hudson_orthogonal_stiffness",
        ("k_iso", k_iso),
        ("mu_iso", mu_iso),
        ("k_fluid", k_fluid),
        ("mu_fluid", mu_fluid),
    )
    _require_positive("hudson_orthogonal_stiffness", "k_iso", k_iso, "Pa")
    _require_positive("hudson_orthogonal_stiffness", "mu_iso", mu_iso, "Pa")
    _require_nonnegative("hudson_orthogonal_stiffness", "k_fluid", k_fluid, "Pa")
    _require_nonnegative("hudson_orthogonal_stiffness", "mu_fluid", mu_fluid, "Pa")
    for name, value in (
        ("crack_density", crack_density),
        ("aspect_ratio", aspect_ratio),
    ):
        if value.dtype != k_iso.dtype or value.device != k_iso.device:
            raise RockContractError(
                "orthogonal Hudson crack-set dtype/device mismatch",
                object_name="hudson_orthogonal_stiffness",
                field=name,
                expected={"dtype": str(k_iso.dtype), "device": str(k_iso.device)},
                actual={"dtype": str(value.dtype), "device": str(value.device)},
            )
        if not bool(torch.isfinite(value).all()):
            raise RockContractError(
                "orthogonal Hudson crack-set input must be finite",
                object_name="hudson_orthogonal_stiffness",
                field=name,
                expected="finite entries",
                actual="non-finite value(s)",
            )
    try:
        base_shape = torch.broadcast_shapes(
            k_iso.shape,
            crack_density.shape[:-1],
            aspect_ratio.shape[:-1],
        )
    except RuntimeError as exc:
        raise RockContractError(
            "orthogonal Hudson batch axes are not broadcastable",
            object_name="hudson_orthogonal_stiffness",
            field="broadcast_shape",
            expected="broadcastable batch axes before the crack-set axis",
            actual={
                "k_iso": tuple(k_iso.shape),
                "crack_density": tuple(crack_density.shape),
                "aspect_ratio": tuple(aspect_ratio.shape),
            },
            code=ErrorCode.SHAPE_MISMATCH,
        ) from exc
    k_iso = k_iso.expand(base_shape)[..., None]
    mu_iso = mu_iso.expand(base_shape)[..., None]
    k_fluid = k_fluid.expand(base_shape)[..., None]
    mu_fluid = mu_fluid.expand(base_shape)[..., None]
    density = crack_density.expand(base_shape + (3,))
    alpha = aspect_ratio.expand(base_shape + (3,))
    _require_nonnegative("hudson_orthogonal_stiffness", "crack_density", density, "1")
    _require_positive("hudson_orthogonal_stiffness", "aspect_ratio", alpha, "1")
    if bool(torch.any(alpha > 1.0)):
        raise RockContractError(
            "orthogonal Hudson aspect ratio exceeds one",
            object_name="hudson_orthogonal_stiffness",
            field="aspect_ratio",
            expected="0 < aspect_ratio <= 1",
            actual=_extrema(alpha, "1"),
        )
    c11, c33, c13, c44, _ = _hudson_components(
        k_iso,
        mu_iso,
        density,
        alpha,
        k_fluid,
        mu_fluid,
        order=1,
    )
    lam = k_iso[..., 0] - 2.0 * mu_iso[..., 0] / 3.0
    mu = mu_iso[..., 0]
    d11 = c11 - (lam + 2.0 * mu)[..., None]
    d33 = c33 - (lam + 2.0 * mu)[..., None]
    d13 = c13 - lam[..., None]
    d44 = c44 - mu[..., None]
    c11_o = lam + 2.0 * mu + d33[..., 0] + d11[..., 1] + d11[..., 2]
    c22_o = lam + 2.0 * mu + d11[..., 0] + d33[..., 1] + d11[..., 2]
    c33_o = lam + 2.0 * mu + d11[..., 0] + d11[..., 1] + d33[..., 2]
    c12_o = lam + d13[..., 0] + d13[..., 1] + d11[..., 2]
    c13_o = lam + d13[..., 0] + d11[..., 1] + d13[..., 2]
    c23_o = lam + d11[..., 0] + d13[..., 1] + d13[..., 2]
    c44_o = mu + d44[..., 1] + d44[..., 2]
    c55_o = mu + d44[..., 0] + d44[..., 2]
    c66_o = mu + d44[..., 0] + d44[..., 1]
    zero = torch.zeros_like(c11_o)
    stiffness = _matrix_rows(
        (c11_o, c12_o, c13_o, zero, zero, zero),
        (c12_o, c22_o, c23_o, zero, zero, zero),
        (c13_o, c23_o, c33_o, zero, zero, zero),
        (zero, zero, zero, c44_o, zero, zero),
        (zero, zero, zero, zero, c55_o, zero),
        (zero, zero, zero, zero, zero, c66_o),
    )
    return _validate_matrix(
        stiffness,
        object_name="hudson_orthogonal_stiffness",
        field="stiffness",
        positive_definite=True,
    )


def hudson_cone_stiffness(
    k_iso: torch.Tensor,
    mu_iso: TensorInput,
    crack_density: TensorInput,
    aspect_ratio: TensorInput,
    *,
    cone_angle_radians: TensorInput,
    k_fluid: TensorInput = 0.0,
    mu_fluid: TensorInput = 0.0,
) -> torch.Tensor:
    """Return Hudson stiffness for a radian cone distribution around axis 3."""
    k_iso, mu_iso, density, alpha, theta, k_fluid, mu_fluid = _validated_scalars(
        "hudson_cone_stiffness",
        ("k_iso", k_iso),
        ("mu_iso", mu_iso),
        ("crack_density", crack_density),
        ("aspect_ratio", aspect_ratio),
        ("cone_angle_radians", cone_angle_radians),
        ("k_fluid", k_fluid),
        ("mu_fluid", mu_fluid),
    )
    hudson_stiffness(
        k_iso,
        mu_iso,
        density,
        alpha,
        k_fluid=k_fluid,
        mu_fluid=mu_fluid,
    )
    if bool(torch.any((theta < 0.0) | (theta > math.pi / 2.0))):
        raise RockContractError(
            "Hudson cone angle lies outside the orientation domain",
            object_name="hudson_cone_stiffness",
            field="cone_angle_radians",
            expected="0 <= angle <= pi/2 rad",
            actual=_extrema(theta, "rad"),
        )
    lam, p_wave, u1, u3 = _hudson_factors(k_iso, mu_iso, alpha, k_fluid, mu_fluid)
    sin2 = torch.sin(theta).square()
    cos2 = torch.cos(theta).square()
    sin4 = sin2.square()
    d11 = (
        -density
        / (2.0 * mu_iso)
        * (
            u3 * (2.0 * lam.square() + 4.0 * lam * mu_iso * sin2 + 3.0 * mu_iso.square() * sin4)
            + u1 * mu_iso.square() * sin2 * (4.0 - 3.0 * sin2)
        )
    )
    d33 = (
        -density
        / mu_iso
        * (u3 * (lam + 2.0 * mu_iso * cos2).square() + 4.0 * u1 * mu_iso.square() * cos2 * sin2)
    )
    d13 = (
        -density
        / mu_iso
        * (
            u3 * (lam + mu_iso * sin2) * (lam + 2.0 * mu_iso * cos2)
            - 2.0 * u1 * mu_iso.square() * sin2 * cos2
        )
    )
    d44 = (
        -density
        * mu_iso
        / 2.0
        * (4.0 * u3 * sin2 * cos2 + u1 * (sin2 + 2.0 * cos2 - 4.0 * sin2 * cos2))
    )
    d66 = -density * mu_iso / 2.0 * (u3 * sin4 + u1 * sin2 * (2.0 - sin2))
    stiffness = _vti_matrix(
        p_wave + d11,
        p_wave + d33,
        lam + d13,
        mu_iso + d44,
        mu_iso + d66,
        axis=3,
    )
    return _validate_matrix(
        stiffness,
        object_name="hudson_cone_stiffness",
        field="stiffness",
        positive_definite=True,
    )


def _bond_matrix(theta: torch.Tensor, axis: int) -> torch.Tensor:
    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)
    if axis == 3:
        b11, b22, b33 = cosine, cosine, one
        b12, b21 = sine, -sine
        b13 = b31 = b23 = b32 = zero
    elif axis == 1:
        b11 = one
        b22, b33 = cosine, cosine
        b23, b32 = -sine, sine
        b12 = b21 = b13 = b31 = zero
    elif axis == 2:
        b11, b33 = cosine, cosine
        b22 = one
        b13, b31 = sine, -sine
        b12 = b21 = b23 = b32 = zero
    else:
        raise RockContractError(
            "Bond rotation axis is unsupported",
            object_name="bond_rotate_stiffness",
            field="axis",
            expected="1, 2, or 3",
            actual=axis,
        )
    m1 = _matrix_rows(
        (b11.square(), b12.square(), b13.square()),
        (b21.square(), b22.square(), b23.square()),
        (b31.square(), b32.square(), b33.square()),
    )
    m2 = _matrix_rows(
        (2.0 * b12 * b13, 2.0 * b13 * b11, 2.0 * b11 * b12),
        (2.0 * b22 * b23, 2.0 * b23 * b21, 2.0 * b21 * b22),
        (2.0 * b32 * b33, 2.0 * b33 * b31, 2.0 * b31 * b32),
    )
    m3 = _matrix_rows(
        (b21 * b31, b22 * b32, b23 * b33),
        (b31 * b11, b32 * b12, b33 * b13),
        (b11 * b21, b12 * b22, b13 * b23),
    )
    m4 = _matrix_rows(
        (b22 * b33 + b23 * b32, b21 * b33 + b23 * b31, b22 * b31 + b21 * b32),
        (b12 * b33 + b13 * b32, b11 * b33 + b13 * b31, b11 * b32 + b12 * b31),
        (b22 * b13 + b12 * b23, b11 * b23 + b13 * b21, b22 * b11 + b12 * b21),
    )
    return torch.cat((torch.cat((m1, m2), dim=-1), torch.cat((m3, m4), dim=-1)), dim=-2)


def bond_rotate_stiffness(
    stiffness: torch.Tensor,
    angle_radians: TensorInput,
    *,
    axis: int = 3,
) -> torch.Tensor:
    """Rotate stiffness by a radian Bond transform around a principal axis."""
    if not isinstance(axis, Integral) or isinstance(axis, bool) or int(axis) not in (1, 2, 3):
        raise RockContractError(
            "Bond rotation axis is unsupported",
            object_name="bond_rotate_stiffness",
            field="axis",
            expected="integer 1, 2, or 3",
            actual=axis,
        )
    axis_value = int(axis)
    stiffness = _validate_matrix(
        stiffness,
        object_name="bond_rotate_stiffness",
        field="stiffness",
        positive_definite=True,
    )
    angle = _validated_scalars(
        "bond_rotate_stiffness",
        ("matrix_batch", stiffness[..., 0, 0]),
        ("angle_radians", angle_radians),
    )[1]
    batch_shape = torch.broadcast_shapes(stiffness.shape[:-2], angle.shape)
    stiffness = stiffness.expand(batch_shape + (6, 6))
    transform = _bond_matrix(angle.expand(batch_shape), axis_value)
    rotated = transform @ stiffness @ transform.transpose(-1, -2)
    return _validate_matrix(
        rotated,
        object_name="bond_rotate_stiffness",
        field="stiffness",
        positive_definite=True,
    )


def sayers_kachanov_stiffness(
    k_iso: torch.Tensor,
    mu_iso: TensorInput,
    normal_compliance: TensorInput,
    tangential_compliance: TensorInput,
) -> torch.Tensor:
    """Return VTI stiffness from the Sayers--Kachanov compliance law."""
    k_iso, mu_iso, normal, tangential = _validated_scalars(
        "sayers_kachanov_stiffness",
        ("k_iso", k_iso),
        ("mu_iso", mu_iso),
        ("normal_compliance", normal_compliance),
        ("tangential_compliance", tangential_compliance),
    )
    _require_positive("sayers_kachanov_stiffness", "k_iso", k_iso, "Pa")
    _require_positive("sayers_kachanov_stiffness", "mu_iso", mu_iso, "Pa")
    _require_nonnegative("sayers_kachanov_stiffness", "normal_compliance", normal, "Pa^-1")
    _require_nonnegative("sayers_kachanov_stiffness", "tangential_compliance", tangential, "Pa^-1")
    isotropic = isotropic_stiffness(k_iso, mu_iso)
    compliance = compliance_from_stiffness(isotropic).clone()
    compliance[..., 2, 2] = compliance[..., 2, 2] + normal
    compliance[..., 3, 3] = compliance[..., 3, 3] + tangential
    compliance[..., 4, 4] = compliance[..., 4, 4] + tangential
    return stiffness_from_compliance(compliance)


def sayers_kachanov_phase_properties(
    k_iso: torch.Tensor,
    mu_iso: TensorInput,
    rho: TensorInput,
    normal_compliance: TensorInput,
    tangential_compliance: TensorInput,
) -> AnisotropicPhaseResult:
    """Return Sayers--Kachanov phase properties from its stiffness tensor.

    Args:
        k_iso: background isotropic bulk modulus [Pa].
        mu_iso: background shear modulus [Pa].
        rho: bulk density [kg/m^3].
        normal_compliance: fracture normal compliance [1/Pa].
        tangential_compliance: fracture tangential compliance [1/Pa].
    """
    return thomsen_properties_from_stiffness(
        sayers_kachanov_stiffness(
            k_iso,
            mu_iso,
            normal_compliance,
            tangential_compliance,
        ),
        rho,
    )


def backus_average(
    vp: torch.Tensor,
    vs: TensorInput,
    rho: TensorInput,
    fractions: TensorInput,
) -> BackusResult:
    """Return the Backus average of one one-dimensional isotropic stack."""
    vp, vs, rho, fractions = _validated_scalars(
        "backus_average",
        ("vp", vp),
        ("vs", vs),
        ("rho", rho),
        ("fractions", fractions),
    )
    if vp.ndim != 1:
        raise RockContractError(
            "Backus inputs must be one-dimensional layer vectors",
            object_name="backus_average",
            field="vp",
            expected="shape (layers,)",
            actual=tuple(vp.shape),
            code=ErrorCode.SHAPE_MISMATCH,
        )
    _require_positive("backus_average", "vp", vp, "m/s")
    _require_positive("backus_average", "vs", vs, "m/s")
    _require_positive("backus_average", "rho", rho, "kg/m^3")
    _require_nonnegative("backus_average", "fractions", fractions, "1")
    total = fractions.sum()
    tolerance = fractions.new_tensor(64.0 * torch.finfo(fractions.dtype).eps)
    if not bool(torch.isclose(total, total.new_tensor(1.0), rtol=0.0, atol=tolerance)):
        raise RockContractError(
            "Backus layer fractions must sum to one",
            object_name="backus_average",
            field="fractions",
            expected={"sum": 1.0, "absolute_tolerance": tolerance.item()},
            actual={"sum": total.item()},
        )
    p_wave = rho * vp.square()
    mu = rho * vs.square()
    bulk_modulus = p_wave - 4.0 * mu / 3.0
    _require_positive("backus_average", "shear_modulus", mu, "Pa")
    _require_positive("backus_average", "bulk_modulus", bulk_modulus, "Pa")
    lam = p_wave - 2.0 * mu
    average_inverse_p = (fractions / p_wave).sum()
    average_inverse_mu = (fractions / mu).sum()
    average_mu = (fractions * mu).sum()
    average_lam_over_p = (fractions * lam / p_wave).sum()
    average_reduced_p = (fractions * (p_wave - lam.square() / p_wave)).sum()
    rho_eff = (fractions * rho).sum()
    c33 = average_inverse_p.reciprocal()
    c44 = average_inverse_mu.reciprocal()
    c66 = average_mu
    c13 = average_lam_over_p * c33
    c11 = average_reduced_p + average_lam_over_p.square() * c33
    stiffness = _vti_matrix(c11, c33, c13, c44, c66, axis=3)
    phase = thomsen_properties_from_stiffness(stiffness, rho_eff)
    return BackusResult(
        stiffness=phase.stiffness,
        rho=rho_eff,
        vp0=phase.vp0,
        vs0=phase.vs0,
        epsilon=phase.epsilon,
        delta=phase.delta,
        gamma=phase.gamma,
    )


__all__ = [
    "AnisotropicPhaseResult",
    "BackusResult",
    "backus_average",
    "bond_rotate_stiffness",
    "compliance_from_stiffness",
    "hudson_cone_stiffness",
    "hudson_orthogonal_stiffness",
    "hudson_phase_properties",
    "hudson_random_moduli",
    "hudson_stiffness",
    "isotropic_compliance",
    "isotropic_moduli_from_compliance",
    "isotropic_stiffness",
    "sayers_kachanov_phase_properties",
    "sayers_kachanov_stiffness",
    "stiffness_from_compliance",
    "thomsen_properties_from_stiffness",
    "thomsen_phase_velocities",
    "vti_stiffness_from_thomsen",
]
