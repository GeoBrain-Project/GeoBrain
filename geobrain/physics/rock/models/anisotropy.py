"""
VTI anisotropy ``ComponentModel`` implementations.

Pure-math layer. The ``*Operator`` wrappers are colocated in this module.

Models:
    Thomsen:           weak-anisotropy phase velocities vp(θ), vsv(θ), vsh(θ)
    Backus:            thin-layer averaging → effective VTI
    Hudson:            penny-shaped cracks → VTI perturbation
    BrownKorringa:     anisotropic Gassmann fluid substitution
    SayersKachanov:    fracture-compliance perturbation

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from ..anisotropy import (
    _legacy_degrees_to_radians,
    backus_average,
    bond_rotate_stiffness,
    compliance_from_stiffness,
    hudson_phase_properties,
    sayers_kachanov_phase_properties,
    stiffness_from_compliance,
    thomsen_phase_velocities,
    thomsen_properties_from_stiffness,
    vti_stiffness_from_thomsen,
)
from ..fluid_substitution import brown_korringa_saturated_compliance

from ....core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    GeoBrainError,
    ModelState,
    ForwardOperator,
    ForwardOutput,
    PropertyTransform,
)
from ._categories import AnisotropyModel
from ._registry import register
from ._types import EPS, PI


# --- Thomsen ---------------------------------------------------------------


@register("Thomsen", aliases=["thomsen"])
class Thomsen(AnisotropyModel):
    """
    Weak-anisotropy VTI phase velocities ``vp(θ), vsv(θ), vsh(θ)``.

    Takes ``angles_rad`` as a buffer-like tensor passed at forward time.
    Returns three tensors of shape ``(n_depth, n_angles)`` built by
    broadcasting ``(n_depth, 1)`` × ``(1, n_angles)``.
    """

    def forward(
        self,
        vp0: Tensor,
        vs0: Tensor,
        eps_: Tensor,
        delta_: Tensor,
        gamma_: Tensor,
        angles_rad: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return thomsen_phase_velocities(vp0, vs0, eps_, delta_, gamma_, angles_rad)


# --- Backus ----------------------------------------------------------------


@register("Backus", aliases=["backus"])
class Backus(AnisotropyModel):
    """
    Backus 1962 thin-layer averaging for isotropic layers → effective VTI.

    Inputs (forward): 1-D tensors ``vp, vs, rho, f`` of equal length;
    ``f`` must sum to one. Returns the 6-tuple
    ``(vp_eff, vs_eff, rho_eff, epsilon, delta, gamma)`` as scalars.
    """

    def forward(
        self,
        vp: Tensor,
        vs: Tensor,
        rho: Tensor,
        f: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        result = backus_average(vp, vs, rho, f)
        return (
            result.vp0,
            result.vs0,
            result.rho,
            result.epsilon,
            result.delta,
            result.gamma,
        )


# --- Hudson ----------------------------------------------------------------


@register("Hudson", aliases=["hudson"])
class Hudson(AnisotropyModel):
    """
    Hudson 1980 first-order penny-shaped crack perturbation to an isotropic background.

    Returns ``(vp0, vs0, epsilon, delta, gamma)`` of the perturbed VTI medium.
    """

    def __init__(self, *, K_fluid: float = 0.0, mu_fluid: float = 0.0) -> None:
        super().__init__()
        for name, value in (("K_fluid", K_fluid), ("mu_fluid", mu_fluid)):
            if value < 0:
                raise GeoBrainError(
                    f"Hudson {name} must be non-negative",
                    object_name="Hudson",
                    field=name,
                    expected=">= 0",
                    actual=value,
                )
        self.K_fluid = float(K_fluid)
        self.mu_fluid = float(mu_fluid)

    def forward(
        self,
        K_iso: Tensor,
        mu_iso: Tensor,
        rho: Tensor,
        eps_c: Tensor,
        alpha: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        result = hudson_phase_properties(
            K_iso,
            mu_iso,
            rho,
            eps_c,
            alpha,
            k_fluid=self.K_fluid,
            mu_fluid=self.mu_fluid,
        )
        return result.vp0, result.vs0, result.epsilon, result.delta, result.gamma


# --- Brown-Korringa --------------------------------------------------------


@register("BrownKorringa", aliases=["brown_korringa", "bk"])
class BrownKorringa(AnisotropyModel):
    """
    Brown-Korringa 1975 anisotropic Gassmann substitution.

    Returns saturated ``(vp0, vs0, epsilon, delta, gamma, rho_sat)``.
    Shear compliances (C44, C66) are BK-invariant, so γ_sat = γ_dry.

    Internally scales stiffness by ``K_mineral`` so all intermediate
    compliance values are O(1); without this scaling, float32 precision
    is insufficient through the 3×3 inverse (compliance ≈ 1e-11) and
    backward gradients turn into NaN.
    """

    def __init__(
        self,
        *,
        K_mineral: float = 37.0e9,
        mu_mineral: float = 44.0e9,
        K_fluid: float = 2.25e9,
        rho_fluid: float = 1000.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("K_mineral", K_mineral),
            ("mu_mineral", mu_mineral),
            ("K_fluid", K_fluid),
            ("rho_fluid", rho_fluid),
        ):
            if value <= 0:
                raise GeoBrainError(
                    f"BrownKorringa {name} must be positive",
                    object_name="BrownKorringa",
                    field=name,
                    expected="> 0",
                    actual=value,
                )
        self.K_mineral = float(K_mineral)
        self.mu_mineral = float(mu_mineral)
        self.K_fluid = float(K_fluid)
        self.rho_fluid = float(rho_fluid)

    def forward(
        self,
        vp0: Tensor,
        vs0: Tensor,
        eps_: Tensor,
        delta_: Tensor,
        gamma_: Tensor,
        rho_dry: Tensor,
        phi: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        dry_stiffness = vti_stiffness_from_thomsen(vp0, vs0, eps_, delta_, gamma_, rho_dry)
        dry_compliance = compliance_from_stiffness(dry_stiffness)
        saturated_compliance = brown_korringa_saturated_compliance(
            dry_compliance,
            self.K_mineral,
            self.mu_mineral,
            self.K_fluid,
            phi,
        )
        saturated_stiffness = stiffness_from_compliance(saturated_compliance)
        rho_sat = rho_dry + phi * self.rho_fluid
        phase = thomsen_properties_from_stiffness(saturated_stiffness, rho_sat)
        return phase.vp0, phase.vs0, phase.epsilon, phase.delta, phase.gamma, rho_sat


# --- Sayers-Kachanov --------------------------------------------------------


@register("SayersKachanov", aliases=["sayers_kachanov", "sk"])
class SayersKachanov(AnisotropyModel):
    """
    Sayers-Kachanov 1995 general fracture-compliance VTI perturbation.

    Adds VTI anisotropy to an isotropic background by direct compliance
    perturbation (Z_N, Z_T). Returns
    ``(vp0, vs0, epsilon, delta, gamma)``.
    """

    def forward(
        self,
        K_iso: Tensor,
        mu_iso: Tensor,
        rho: Tensor,
        Z_N: Tensor,
        Z_T: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        result = sayers_kachanov_phase_properties(K_iso, mu_iso, rho, Z_N, Z_T)
        return result.vp0, result.vs0, result.epsilon, result.delta, result.gamma


# --- Bond rotation ----------------------------------------------------------


@register("BondTransform", aliases=["Bond", "bond"])
class BondTransform(AnisotropyModel):
    """
    Bond rotation of a 6×6 stiffness matrix around a principal axis.

    ``forward(C, theta_deg, axis)`` returns the rotated stiffness.
    ``axis`` is 1, 2, or 3 (x, y, z).
    """

    def forward(self, C: Tensor, theta: Tensor | float, axis: int = 3) -> Tensor:
        if (
            not isinstance(C, Tensor)
            or C.layout is not torch.strided
            or C.device.type == "meta"
            or C.ndim < 2
        ):
            return bond_rotate_stiffness(C, 0.0, axis=axis)
        theta_radians = _legacy_degrees_to_radians(
            C[..., 0, 0],
            theta,
            object_name="BondTransform",
            field="theta_degrees",
        )
        return bond_rotate_stiffness(C, theta_radians, axis=axis)


# --- Azimuth-dependent velocities (HTI / VTI) -------------------------------


@register("VelocityAzimuthHTI", aliases=["VAz_HTI", "vaz_hti"])
class VelocityAzimuthHTI(AnisotropyModel):
    """
    Azimuth-dependent phase velocities for an HTI medium.

    Takes a 6×6 stiffness ``C``, density ``rho``, and azimuth (degrees).
    Returns ``(VP, VSH, VSV)`` in the same units as ``sqrt(C/rho)`` (so
    Pa/(kg/m³) → m/s).
    """

    def forward(
        self,
        C: Tensor,
        rho: Tensor,
        azimuth_deg: Tensor | float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        azimuth_rad = torch.as_tensor(
            azimuth_deg,
            dtype=C.dtype,
            device=C.device,
        ) * (PI / 180.0)
        C11 = C[0, 0]
        C12 = C[0, 1]
        C33 = C[2, 2]
        C44 = C[3, 3]
        C55 = C[4, 4]
        sin2 = torch.sin(azimuth_rad).pow(2)
        cos2 = torch.cos(azimuth_rad).pow(2)

        R = torch.sqrt(
            ((C33 - C55) * sin2 - (C11 - C55) * cos2) ** 2 + 4.0 * (C12 + C55) ** 2 * sin2 * cos2
        )
        Vp2 = 0.5 / rho * ((C33 + C55) * sin2 + (C11 + C55) * cos2 + R)
        Vsh2 = 0.5 / rho * ((C33 + C55) * sin2 + (C11 + C55) * cos2 - R)
        Vsv2 = 1.0 / rho * (C55 + (C44 - C55) * sin2)

        VP = torch.sqrt(torch.clamp(Vp2, min=EPS))
        VSH = torch.sqrt(torch.clamp(Vsh2, min=EPS))
        VSV = torch.sqrt(torch.clamp(Vsv2, min=EPS))
        return VP, VSH, VSV


@register("VelocityAzimuthVTI", aliases=["VAz_VTI", "vaz_vti"])
class VelocityAzimuthVTI(AnisotropyModel):
    """
    Angle-dependent phase velocities for a VTI medium (exact, not weak-anisotropy).

    Takes a 6×6 stiffness ``C``, density ``rho``, and polar angle θ (degrees
    from vertical). Returns ``(VP, VSH, VSV)``.
    """

    def forward(
        self,
        C: Tensor,
        rho: Tensor,
        theta_deg: Tensor | float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        theta_rad = torch.as_tensor(
            theta_deg,
            dtype=C.dtype,
            device=C.device,
        ) * (PI / 180.0)
        C11 = C[0, 0]
        C13 = C[0, 2]
        C33 = C[2, 2]
        C44 = C[3, 3]
        C66 = C[5, 5]
        sin2 = torch.sin(theta_rad).pow(2)
        cos2 = torch.cos(theta_rad).pow(2)

        R = torch.sqrt(
            ((C11 - C44) * sin2 - (C33 - C44) * cos2) ** 2
            + (C13 + C44) ** 2 * torch.sin(2.0 * theta_rad) ** 2
        )
        Vp2 = 0.5 / rho * (C11 * sin2 + C33 * cos2 + C44 + R)
        Vsv2 = 0.5 / rho * (C11 * sin2 + C33 * cos2 + C44 - R)
        Vsh2 = 1.0 / rho * (C66 * sin2 + C44 * cos2)

        VP = torch.sqrt(torch.clamp(Vp2, min=EPS))
        VSH = torch.sqrt(torch.clamp(Vsh2, min=EPS))
        VSV = torch.sqrt(torch.clamp(Vsv2, min=EPS))
        return VP, VSH, VSV


# --- Orthorhombic Thomsen-Tsvankin ------------------------------------------


@register("ThomsenTsvankin", aliases=["tt", "thomsen_tsvankin"])
class ThomsenTsvankin(AnisotropyModel):
    """
    Thomsen-Tsvankin 7-parameter orthorhombic anisotropy descriptor.

    Returns ``(ε₁, δ₁, γ₁, ε₂, δ₂, γ₂, δ₃)`` from the orthorhombic
    stiffness components ``(C11, C22, C33, C12, C13, C23, C44, C55, C66)``.
    """

    def forward(
        self,
        C11: Tensor,
        C22: Tensor,
        C33: Tensor,
        C12: Tensor,
        C13: Tensor,
        C23: Tensor,
        C44: Tensor,
        C55: Tensor,
        C66: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        eps1 = (C22 - C33) / (2.0 * C33 + EPS)
        delta1 = ((C23 + C44) ** 2 - (C33 - C44) ** 2) / (2.0 * C33 * (C33 - C44) + EPS)
        gamma1 = (C66 - C55) / (2.0 * C55 + EPS)

        eps2 = (C11 - C33) / (2.0 * C33 + EPS)
        delta2 = ((C13 + C55) ** 2 - (C33 - C55) ** 2) / (2.0 * C33 * (C33 - C55) + EPS)
        gamma2 = (C66 - C44) / (2.0 * C44 + EPS)

        delta3 = ((C12 + C66) ** 2 - (C11 - C66) ** 2) / (2.0 * C11 * (C11 - C66) + EPS)
        return eps1, delta1, gamma1, eps2, delta2, gamma2, delta3


__all__ = [
    "Backus",
    "BackusOperator",
    "BondTransform",
    "BrownKorringa",
    "BrownKorringaOperator",
    "Hudson",
    "HudsonOperator",
    "SayersKachanov",
    "SayersKachanovOperator",
    "Thomsen",
    "ThomsenOperator",
    "ThomsenTsvankin",
    "VelocityAzimuthHTI",
    "VelocityAzimuthVTI",
]


# ============================================================================
# Operator wrappers (PropertyTransform / ForwardOperator)
#
# Merged from rock/anisotropy.py.
# ============================================================================

# Private aliases
_ThomsenModel = Thomsen
_BackusModel = Backus
_HudsonModel = Hudson
_BKModel = BrownKorringa
_SKModel = SayersKachanov


class ThomsenOperator(ForwardOperator):
    """
    VTI weak-anisotropy phase velocities (ForwardOperator).

    Inputs (ModelState): ``vp0, vs0, epsilon, delta, gamma``.
    Output (ForwardOutput): ``vp_theta, vsv_theta, vsh_theta``, each
        shape ``(n_depth, n_angles)``.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp0", "vs0", "epsilon", "delta", "gamma"),
        output_keys=("vp_theta", "vsv_theta", "vsh_theta"),
    )

    def __init__(self, angles_deg: Sequence[float] | torch.Tensor) -> None:
        super().__init__()
        angles_t = torch.as_tensor(angles_deg, dtype=torch.float32)
        if angles_t.ndim != 1:
            raise GeoBrainError(
                "Thomsen angles must be a 1D sequence",
                object_name="Thomsen",
                field="angles_deg",
                expected="1D",
                actual=tuple(angles_t.shape),
            )
        if (angles_t < 0).any() or (angles_t >= 90).any():
            raise GeoBrainError(
                "Thomsen angles must be in [0, 90)",
                object_name="Thomsen",
                field="angles_deg",
                expected="[0, 90)",
                actual=angles_t.tolist(),
            )
        self.register_buffer(
            "angles_rad",
            angles_t * (math.pi / 180.0),
            persistent=False,
        )
        self._model = _ThomsenModel()

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        vp0, vs0, eps_, delta_, gamma_ = state.fetch(
            "vp0",
            "vs0",
            "epsilon",
            "delta",
            "gamma",
        )
        for name, t in zip(
            ("vs0", "epsilon", "delta", "gamma"),
            (vs0, eps_, delta_, gamma_),
        ):
            if t.shape != vp0.shape:
                raise GeoBrainError(
                    "Thomsen inputs must share shape",
                    object_name="Thomsen",
                    field=name,
                    expected=tuple(vp0.shape),
                    actual=tuple(t.shape),
                )
        theta = self.angles_rad.to(vp0.dtype).to(vp0.device)
        vp_theta, vsv_theta, vsh_theta = self._model(
            vp0,
            vs0,
            eps_,
            delta_,
            gamma_,
            theta,
        )
        return ForwardOutput(
            data={
                "vp_theta": vp_theta,
                "vsv_theta": vsv_theta,
                "vsh_theta": vsh_theta,
            },
            metadata={"angles_deg": (self.angles_rad * 180.0 / math.pi).tolist()},
        )


class BackusOperator(PropertyTransform):
    """Backus thin-layer averaging operator (1D layers → effective VTI scalars)."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp", "vs", "rho", "f"),
        output_keys=("vp_eff", "vs_eff", "rho_eff", "epsilon", "delta", "gamma"),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model = _BackusModel()

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        vp, vs, rho, f = state.fetch("vp", "vs", "rho", "f")
        for name, t in (("vs", vs), ("rho", rho), ("f", f)):
            if t.ndim != 1 or t.shape != vp.shape:
                raise GeoBrainError(
                    "Backus inputs must be matching 1D tensors",
                    object_name="Backus",
                    field=name,
                    expected=tuple(vp.shape),
                    actual=tuple(t.shape),
                )
        if (f < 0).any():
            raise GeoBrainError(
                "Backus volume fractions must be non-negative",
                object_name="Backus",
                field="f",
                expected=">= 0",
                actual=f.tolist(),
            )
        f_sum = f.sum()
        f_sum_scalar = float(f_sum.detach())
        if f_sum_scalar <= 0:
            raise GeoBrainError(
                "Backus volume fractions sum must be positive",
                object_name="Backus",
                field="f",
                expected="> 0",
                actual=f_sum_scalar,
            )
        vp_eff, vs_eff, rho_eff, epsilon, delta, gamma = self._model(vp, vs, rho, f)
        return state.with_tensors(
            vp_eff=vp_eff,
            vs_eff=vs_eff,
            rho_eff=rho_eff,
            epsilon=epsilon,
            delta=delta,
            gamma=gamma,
        )


class HudsonOperator(PropertyTransform):
    """Hudson penny-shaped cracks → VTI perturbation operator."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("K_iso", "mu_iso", "rho", "crack_density", "aspect_ratio"),
        output_keys=("vp0", "vs0", "epsilon", "delta", "gamma"),
    )

    def __init__(self, *, K_fluid: float = 0.0, mu_fluid: float = 0.0) -> None:
        super().__init__()
        self._model = _HudsonModel(K_fluid=K_fluid, mu_fluid=mu_fluid)
        self.K_fluid = self._model.K_fluid
        self.mu_fluid = self._model.mu_fluid

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        K_iso, mu_iso, rho, eps_c, alpha = state.fetch(
            "K_iso",
            "mu_iso",
            "rho",
            "crack_density",
            "aspect_ratio",
        )
        for name, t in (
            ("mu_iso", mu_iso),
            ("rho", rho),
            ("crack_density", eps_c),
            ("aspect_ratio", alpha),
        ):
            if t.shape != K_iso.shape:
                raise GeoBrainError(
                    "Hudson inputs must share shape",
                    object_name="Hudson",
                    field=name,
                    expected=tuple(K_iso.shape),
                    actual=tuple(t.shape),
                )
        if (eps_c < 0).any():
            raise GeoBrainError(
                "Hudson crack_density must be non-negative",
                object_name="Hudson",
                field="crack_density",
                expected=">= 0",
                actual=float(eps_c.min()),
            )
        if (alpha <= 0).any() or (alpha > 1).any():
            raise GeoBrainError(
                "Hudson aspect_ratio must lie in (0, 1]",
                object_name="Hudson",
                field="aspect_ratio",
                expected="(0, 1]",
                actual=(float(alpha.min()), float(alpha.max())),
            )
        vp0, vs0, epsilon, delta, gamma = self._model(K_iso, mu_iso, rho, eps_c, alpha)
        return state.with_tensors(
            vp0=vp0,
            vs0=vs0,
            epsilon=epsilon,
            delta=delta,
            gamma=gamma,
        )


class BrownKorringaOperator(PropertyTransform):
    """Brown-Korringa anisotropic Gassmann fluid-substitution operator."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp0", "vs0", "epsilon", "delta", "gamma", "rho", "phi"),
        output_keys=("vp0", "vs0", "epsilon", "delta", "gamma", "rho"),
    )

    def __init__(
        self,
        *,
        K_mineral: float = 37.0e9,
        mu_mineral: float = 44.0e9,
        K_fluid: float = 2.25e9,
        rho_fluid: float = 1000.0,
    ) -> None:
        super().__init__()
        self._model = _BKModel(
            K_mineral=K_mineral,
            mu_mineral=mu_mineral,
            K_fluid=K_fluid,
            rho_fluid=rho_fluid,
        )
        self.K_mineral = self._model.K_mineral
        self.mu_mineral = self._model.mu_mineral
        self.K_fluid = self._model.K_fluid
        self.rho_fluid = self._model.rho_fluid

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        vp0, vs0, eps_, delta_, gamma_, rho_dry, phi = state.fetch(
            "vp0",
            "vs0",
            "epsilon",
            "delta",
            "gamma",
            "rho",
            "phi",
        )
        ref_shape = vp0.shape
        for name, t in (
            ("vs0", vs0),
            ("epsilon", eps_),
            ("delta", delta_),
            ("gamma", gamma_),
            ("rho", rho_dry),
            ("phi", phi),
        ):
            if t.shape != ref_shape:
                raise GeoBrainError(
                    "BrownKorringa inputs must share shape",
                    object_name="BrownKorringa",
                    field=name,
                    expected=tuple(ref_shape),
                    actual=tuple(t.shape),
                )
        if (phi < 0).any() or (phi >= 1).any():
            raise GeoBrainError(
                "BrownKorringa porosity must lie in [0, 1)",
                object_name="BrownKorringa",
                field="phi",
                expected="[0, 1)",
                actual=(float(phi.min()), float(phi.max())),
            )
        vp0_s, vs0_s, eps_s, delta_s, gamma_s, rho_sat = self._model(
            vp0,
            vs0,
            eps_,
            delta_,
            gamma_,
            rho_dry,
            phi,
        )
        return state.with_tensors(
            vp0=vp0_s,
            vs0=vs0_s,
            epsilon=eps_s,
            delta=delta_s,
            gamma=gamma_s,
            rho=rho_sat,
        )


class SayersKachanovOperator(PropertyTransform):
    """Sayers-Kachanov fracture-compliance perturbation operator."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("K_iso", "mu_iso", "rho", "Z_N", "Z_T"),
        output_keys=("vp0", "vs0", "epsilon", "delta", "gamma"),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model = _SKModel()

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        K_iso, mu_iso, rho, Z_N, Z_T = state.fetch(
            "K_iso",
            "mu_iso",
            "rho",
            "Z_N",
            "Z_T",
        )
        for name, t in (
            ("mu_iso", mu_iso),
            ("rho", rho),
            ("Z_N", Z_N),
            ("Z_T", Z_T),
        ):
            if t.shape != K_iso.shape:
                raise GeoBrainError(
                    "SayersKachanov inputs must share shape",
                    object_name="SayersKachanov",
                    field=name,
                    expected=tuple(K_iso.shape),
                    actual=tuple(t.shape),
                )
        if (Z_N < 0).any() or (Z_T < 0).any():
            raise GeoBrainError(
                "SayersKachanov fracture compliances must be non-negative",
                object_name="SayersKachanov",
                field="Z_N/Z_T",
                expected=">= 0",
                actual=(float(Z_N.min()), float(Z_T.min())),
            )
        vp0, vs0, epsilon, delta, gamma = self._model(K_iso, mu_iso, rho, Z_N, Z_T)
        return state.with_tensors(
            vp0=vp0,
            vs0=vs0,
            epsilon=epsilon,
            delta=delta,
            gamma=gamma,
        )
