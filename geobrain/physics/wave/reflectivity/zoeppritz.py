"""
Exact Zoeppritz P-P reflection coefficient.

Handles post-critical angles by promoting to complex when ``p · v > 1`` (the
transmitted-wave ``arcsin`` argument leaves the real interval).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from torch import Tensor

from ....core import GeoBrainError
from .base import AVOModel, vectorize_angles
from .registry import register_avo


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    """Complex dtype matching a real dtype (float64 -> complex128, else complex64)."""
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def _complex_arcsin(x: Tensor) -> Tensor:
    """``arcsin`` that promotes to complex past the real branch (``|x| > 1``).

    The complex working dtype follows the input REAL precision; float64 stays in
    ``complex128``, float32 in ``complex64``. Hardcoding ``complex64`` here would
    silently truncate a double-precision computation in the post-critical
    ``arcsin``/``log`` stage (the loss was masked downstream because mixing the
    real float64 densities with the complex64 angles re-promotes the final
    reflectivity back to ``complex128``)."""
    cdtype = _complex_dtype(x.dtype)
    real_mask = torch.abs(x) <= 1.0
    x_clamped = torch.clamp(x, -1.0, 1.0)
    result = torch.arcsin(x_clamped)
    if not real_mask.all():
        result = result.to(cdtype)
        x_c = x.to(cdtype)
        i = torch.tensor(1j, dtype=cdtype, device=x.device)
        complex_result = -i * torch.log(i * x_c + torch.sqrt(1 - x_c ** 2))
        result = torch.where(real_mask, result, complex_result)
    return result


def _require_positive_finite_properties(
    named_properties: tuple[tuple[str, Tensor], ...],
) -> None:
    """Reject non-physical material values without perturbing valid inputs."""
    for name, value in named_properties:
        finite = torch.isfinite(value)
        if bool((~finite | (value <= 0)).any()):
            finite_values = value.detach()[finite]
            actual_minimum = (
                float(finite_values.min().cpu()) if finite_values.numel() else None
            )
            raise GeoBrainError(
                "Zoeppritz material properties must be finite and positive",
                object_name="ZoeppritzReflectivity",
                field=name,
                expected="finite values > 0",
                actual={
                    "all_finite": bool(finite.all()),
                    "minimum_finite_value": actual_minimum,
                },
            )


@register_avo(
    "ZoeppritzReflectivity",
    aliases=["zoeppritz", "exact", "Zoeppritz"],
    description="Full Zoeppritz P-P solution (handles post-critical angles)",
)
class ZoeppritzReflectivity(AVOModel):
    """
    Exact Zoeppritz P-P reflection coefficient (math nn.Module).

    Distinct name from the user-facing ForwardOperator wrapper
    :class:`geobrain.physics.wave.reflectivity.operators.Zoeppritz`. Outputs a real tensor
    when all angles are pre-critical; otherwise a complex tensor whose imaginary
    part is the phase shift induced by total reflection.
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "zoeppritz"
        self._valid_angles = (0, 90)

    @vectorize_angles
    def forward(
        self,
        vp1: Tensor, vs1: Tensor, rho1: Tensor,
        vp2: Tensor, vs2: Tensor, rho2: Tensor,
        theta: Tensor,
    ) -> Tensor:
        _require_positive_finite_properties(
            (
                ("vp1", vp1),
                ("vs1", vs1),
                ("rho1", rho1),
                ("vp2", vp2),
                ("vs2", vs2),
                ("rho2", rho2),
            )
        )

        p = torch.sin(theta) / vp1

        theta_p2 = _complex_arcsin(p * vp2)
        theta_s1 = _complex_arcsin(p * vs1)
        theta_s2 = _complex_arcsin(p * vs2)

        tensors = [theta, theta_p2, theta_s1, theta_s2, p]
        if any(t.is_complex() for t in tensors):
            # Promote every operand to the complex dtype matching the input real
            # precision (complex128 for float64) so the solve keeps full precision.
            cdtype = _complex_dtype(p.dtype)
            theta = theta.to(cdtype)
            theta_p2 = theta_p2.to(cdtype)
            theta_s1 = theta_s1.to(cdtype)
            theta_s2 = theta_s2.to(cdtype)
            p = p.to(cdtype)

        cos1 = torch.cos(theta)
        cos2 = torch.cos(theta_p2)
        cos_s1 = torch.cos(theta_s1)
        cos_s2 = torch.cos(theta_s2)
        sin_s1 = torch.sin(theta_s1)
        sin_s2 = torch.sin(theta_s2)

        a = rho2 * (1 - 2 * sin_s2 ** 2) - rho1 * (1 - 2 * sin_s1 ** 2)
        b = rho2 * (1 - 2 * sin_s2 ** 2) + 2 * rho1 * sin_s1 ** 2
        c = rho1 * (1 - 2 * sin_s1 ** 2) + 2 * rho2 * sin_s2 ** 2
        d = 2 * (rho2 * vs2 ** 2 - rho1 * vs1 ** 2)

        E = b * cos1 / vp1 + c * cos2 / vp2
        F = b * cos_s1 / vs1 + c * cos_s2 / vs2
        G = a - d * cos1 / vp1 * cos_s2 / vs2
        H = a - d * cos2 / vp2 * cos_s1 / vs1

        D = E * F + G * H * p ** 2
        finite_denominator = torch.isfinite(D)
        if bool((~finite_denominator | (torch.abs(D) == 0)).any()):
            finite_absolute_values = torch.abs(D.detach())[finite_denominator]
            minimum_absolute_value = (
                float(finite_absolute_values.min().cpu())
                if finite_absolute_values.numel()
                else None
            )
            raise GeoBrainError(
                "Zoeppritz denominator is singular or non-finite",
                object_name="ZoeppritzReflectivity",
                field="denominator",
                expected="finite non-zero values",
                actual={
                    "all_finite": bool(finite_denominator.all()),
                    "minimum_absolute_value": minimum_absolute_value,
                },
            )

        numerator = F * (b * cos1 / vp1 - c * cos2 / vp2) - H * p ** 2 * (
            a + d * cos1 / vp1 * cos_s2 / vs2
        )
        Rpp = numerator / D
        return Rpp


__all__ = ["ZoeppritzReflectivity"]
