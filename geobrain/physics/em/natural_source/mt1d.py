# pyright: reportPrivateImportUsage=false
"""Magnetotelluric 1D: Wait recursion through a layered earth.

This module also exposes :func:`mt1d_response_channels` here, a tiny utility that
maps a complex surface impedance ``Z(ω)`` to the conventional response
channels (apparent resistivity, phase) so callers writing MT-flavoured
inversions don't have to reinvent the conversion formulas.

The math is the Wait (1962) recursion in
tanh form, which is the standard MT1D formulation and is numerically
more stable than the equivalent ``(1 − R·e^(−2γh)) / (1 + R·e^(−2γh))``
form when layers are thick (avoids overflow when ``exp(−2γh)`` becomes
tiny).

Quasi-static, low-frequency limit (displacement current ignored):

    k_k     = √(i ω μ₀ σ_k)                          # vertical wavenumber
    Z_k_∞   = i ω μ₀ / k_k                           # intrinsic impedance

    Z_N     = Z_N_∞                                  # bottom half-space
    Z_k     = Z_k_∞ · (Z_{k+1} + Z_k_∞ tanh(k_k h_k))
                       / (Z_k_∞ + Z_{k+1} tanh(k_k h_k))   # upward recursion

Output: complex surface impedance ``Z(ω)``. Apparent resistivity is
``|Z|² / (ω μ₀)`` and phase is ``arg(Z)`` (= atan2(Im Z, Re Z)) in radians;
this operator returns ``Z`` directly and lets downstream code derive whichever
scalar response is wanted.

Time convention exp(+iωt); phase = atan2(Im Z, Re Z) in radians.

Inputs (ModelState):
    sigma, 1D tensor of layer conductivities, length ``n_layers``.
            The last entry is the half-space.

Outputs (ForwardOutput):
    ``data["impedance"]``: complex tensor of shape ``(n_frequencies,)``.

Differentiability: FULL_AUTOGRAD through the complex recursion.
Internally promotes to complex128 for adjoint precision, then casts
back to the working complex dtype on output.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from geobrain.physics.em.capabilities import EMOperatorDiscovery

from ....core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    GeoBrainError,
    ModelState,
    ForwardOperator,
    ForwardOutput,
)
from ..conventions import MU_0
from ..numerics.layered import mt1d_wait_impedance


MT1DOutput = Literal[
    "impedance_real_imag",
    "apparent_resistivity",
    "log_apparent_resistivity",
    "phase",
]


def mt1d_response_channels(
    impedance: torch.Tensor,
    frequencies: torch.Tensor,
    output: MT1DOutput,
) -> tuple[torch.Tensor, str, dict[str, torch.Tensor]]:
    """
    Convert complex MT impedance into the requested response channel.

    Adapted from ``natural_source/responses.py``; see the
    the rationale of carrying this helper alongside
    :class:`MT1D` rather than as a dedicated submodule.

    Args:
        impedance: complex tensor of shape ``(..., n_frequencies)``,
            ``data["impedance"]`` from :class:`MT1D` or any 1D-equivalent.
        frequencies: real 1-D tensor of frequencies in Hz, broadcast over
            the leading dimensions of ``impedance``.
        output: which derived channel to return. One of
            ``"impedance_real_imag"``, ``"apparent_resistivity"``,
            ``"log_apparent_resistivity"``, or ``"phase"``.

    Returns:
        A ``(data, units, extras)`` triple. ``data`` is the selected
        channel; ``units`` is the SI label; ``extras`` carries the four
        primary derived quantities so callers can stash them on a
        ``ForwardOutput.data`` without recomputing.
    """
    omega = (2.0 * math.pi) * frequencies
    apparent_resistivity = impedance.abs().pow(2) / (omega * MU0)
    phase = torch.atan2(impedance.imag, impedance.real)

    if output == "impedance_real_imag":
        data = torch.stack((impedance.real, impedance.imag), dim=-1)
        units = "ohm"
    elif output == "apparent_resistivity":
        data = apparent_resistivity
        units = "ohm m"
    elif output == "log_apparent_resistivity":
        data = torch.log(apparent_resistivity)
        units = "log ohm m"
    elif output == "phase":
        data = phase
        units = "rad"
    else:
        raise GeoBrainError(
            f"unsupported MT1D output {output!r}",
            object_name="mt1d_response_channels",
            field="output",
            expected="one of 'impedance_real_imag', 'apparent_resistivity', "
                     "'log_apparent_resistivity', 'phase'",
            actual=output,
        )

    return data, units, {
        "impedance_real": impedance.real,
        "impedance_imag": impedance.imag,
        "apparent_resistivity": apparent_resistivity,
        "phase": phase,
    }


MU0 = MU_0  # platform canon, single-sourced in core.constants


@dataclass(frozen=True)
class MT1DSurvey:
    """
    Magnetotelluric frequency list + fixed layer geometry.

    1-D: no lateral coordinates; ``layer_thickness`` is metres, ordered
    top→bottom above a terminal half-space.

    Attributes:
        frequencies: 1D tensor of frequencies in Hz.
        layer_thickness: 1D tensor of layer thicknesses (m); length =
            ``n_layers - 1`` because the bottom layer is a half-space.
    """

    frequencies: torch.Tensor
    layer_thickness: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("frequencies", "layer_thickness"):
            t = getattr(self, name)
            if not isinstance(t, torch.Tensor):
                raise GeoBrainError(
                    f"MT1DSurvey.{name} must be a torch.Tensor",
                    object_name="MT1DSurvey",
                    field=name,
                    expected=torch.Tensor,
                    actual=type(t),
                )
            if t.ndim != 1:
                raise GeoBrainError(
                    f"MT1DSurvey.{name} must be 1D",
                    object_name="MT1DSurvey",
                    field=name,
                    expected="1D Tensor",
                    actual=tuple(t.shape),
                )
        if torch.any(self.frequencies <= 0):
            raise GeoBrainError(
                "MT1DSurvey frequencies must be positive",
                object_name="MT1DSurvey",
                field="frequencies",
                expected="all > 0",
                actual=self.frequencies.tolist(),
            )
        if torch.any(self.layer_thickness <= 0):
            raise GeoBrainError(
                "MT1DSurvey layer_thickness must be positive",
                object_name="MT1DSurvey",
                field="layer_thickness",
                expected="all > 0",
                actual=self.layer_thickness.tolist(),
            )

    @property
    def n_frequencies(self) -> int:
        return int(self.frequencies.numel())

    @property
    def n_interior_layers(self) -> int:
        """Number of finite-thickness layers (excludes the bottom half-space)."""
        return int(self.layer_thickness.numel())


class MT1D(EMOperatorDiscovery, ForwardOperator):
    """
    1D MT surface impedance via Wait recursion.

    Inputs (ModelState):
        ``sigma``: conductivity per layer, shape
            ``(n_interior_layers + 1,)``. Last entry = half-space.

    Outputs (ForwardOutput):
        ``data["impedance"]``: complex tensor of shape ``(n_frequencies,)``.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("sigma",),
        output_keys=("impedance",),
    )

    def __init__(self, survey: MT1DSurvey) -> None:
        super().__init__()
        if not isinstance(survey, MT1DSurvey):
            raise GeoBrainError(
                "MT1D requires an MT1DSurvey",
                object_name="MT1D",
                field="survey",
                expected=MT1DSurvey,
                actual=type(survey),
            )
        self.survey = survey

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        (sigma,) = state.fetch("sigma")
        n_interior = self.survey.n_interior_layers
        expected_n = n_interior + 1
        if sigma.ndim != 1 or sigma.numel() != expected_n:
            raise GeoBrainError(
                "MT1D sigma must be 1D of length n_interior_layers + 1",
                object_name="MT1D",
                field="sigma",
                expected=(expected_n,),
                actual=tuple(sigma.shape),
            )

        device, real_dtype = sigma.device, sigma.dtype
        frequencies = self.survey.frequencies.to(device=device, dtype=real_dtype)
        thickness = self.survey.layer_thickness.to(device=device, dtype=real_dtype)

        Z = mt1d_wait_impedance(sigma, thickness, frequencies)

        # Cast back to the natural complex dtype for the input precision
        # (complex64 for float32 inputs, keep complex128 for float64).
        if real_dtype == torch.float32:
            Z = Z.to(torch.complex64)

        return ForwardOutput(
            data={"impedance": Z},
            metadata={
                "n_frequencies": self.survey.n_frequencies,
                "n_layers": int(sigma.numel()),
            },
        )
