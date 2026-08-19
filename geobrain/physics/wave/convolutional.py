"""
1D convolutional seismogram.

Given a 1D depth profile of vp and rho, compute impedance ``Z = vp * rho``,
reflectivity ``R[i] = (Z[i+1] - Z[i]) / (Z[i+1] + Z[i])``, then convolve with a
source wavelet to produce a normal-incidence trace.

- Inputs (:class:`ModelState`): ``vp``, ``rho``: 1D tensors of length ``nz``.
- Context (:class:`ForwardContext`): ``wavelet``: 1D tensor.
- Output (:class:`ForwardOutput`): ``trace``: 1D tensor of length ``nz``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
from torch.nn import functional as F

from ...core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    GeoBrainError,
    ModelState,
    ForwardOperator,
    ForwardOutput,
)
from .capabilities import WaveCapabilityReport
from .reflectivity._contracts import capability_report, input_schema
from .reflectivity.base import require_compatible_tensors

# The public ``ricker`` free function now lives in ``wave.wavelets`` (the single
# wavelet home); re-exported here for the historical ``wave.ricker`` import path.
from .wavelets import ricker  # noqa: F401


_CONVOLUTION_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)


def build_convolution_matrix(
    wavelet: torch.Tensor,
    sample_count: int,
    *,
    mode: Literal["full", "same"] = "full",
) -> torch.Tensor:
    """Build a differentiable Toeplitz matrix for true seismic convolution.

    ``same`` uses the same zero-padding crop and even-wavelet left-center
    convention as :func:`convolve_reflectivity`.
    """
    if not isinstance(wavelet, torch.Tensor) or wavelet.ndim != 1 or wavelet.numel() < 1:
        raise GeoBrainError(
            "wavelet must be a non-empty 1D Tensor",
            object_name="build_convolution_matrix",
            field="wavelet",
            expected="non-empty 1D Tensor",
            actual=tuple(getattr(wavelet, "shape", ())),
        )
    require_compatible_tensors(
        (("wavelet", wavelet),),
        owner="build_convolution_matrix",
        allowed_dtypes=_CONVOLUTION_DTYPES,
    )
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise GeoBrainError(
            "sample_count must be a positive integer",
            object_name="build_convolution_matrix",
            field="sample_count",
            expected="integer >= 1",
            actual=sample_count,
        )
    if mode not in {"full", "same"}:
        raise GeoBrainError(
            "mode must be 'full' or 'same'",
            object_name="build_convolution_matrix",
            field="mode",
            expected="'full' or 'same'",
            actual=mode,
        )

    full = torch.stack(
        tuple(
            F.pad(wavelet, (index, sample_count - index - 1))
            for index in range(sample_count)
        ),
        dim=1,
    )
    if mode == "full":
        return full
    crop_start = (wavelet.numel() - 1) // 2
    return full[crop_start : crop_start + sample_count]


def convolve_reflectivity(
    reflectivity: torch.Tensor,
    wavelet: torch.Tensor,
    *,
    sample_axis: int = -1,
) -> torch.Tensor:
    """True zero-padded ``same`` convolution along one seismic sample axis.

    All other axes are independent trace/component batches. Odd wavelets are
    centered on their middle sample; even wavelets use the existing GeoBrain
    left-center convention and the trailing extra sample is trimmed.
    """
    if not isinstance(reflectivity, torch.Tensor) or reflectivity.ndim < 1:
        raise GeoBrainError(
            "reflectivity must be a Tensor with a seismic sample axis",
            object_name="convolve_reflectivity",
            field="reflectivity",
            expected="Tensor with rank >= 1",
            actual=type(reflectivity),
        )
    if not isinstance(wavelet, torch.Tensor) or wavelet.ndim != 1 or wavelet.numel() < 1:
        raise GeoBrainError(
            "wavelet must be a non-empty 1D Tensor",
            object_name="convolve_reflectivity",
            field="wavelet",
            expected="non-empty 1D Tensor",
            actual=tuple(getattr(wavelet, "shape", ())),
        )
    require_compatible_tensors(
        (("reflectivity", reflectivity), ("wavelet", wavelet)),
        owner="convolve_reflectivity",
        allowed_dtypes=_CONVOLUTION_DTYPES,
    )
    axis = sample_axis if sample_axis >= 0 else reflectivity.ndim + sample_axis
    if not 0 <= axis < reflectivity.ndim:
        raise GeoBrainError(
            "sample_axis is outside the reflectivity rank",
            object_name="convolve_reflectivity",
            field="sample_axis",
            expected=f"[-{reflectivity.ndim}, {reflectivity.ndim - 1}]",
            actual=sample_axis,
        )

    samples_last = reflectivity.movedim(axis, -1)
    sample_count = samples_last.shape[-1]
    batch_shape = samples_last.shape[:-1]
    flattened = samples_last.reshape(-1, 1, sample_count)
    kernel = wavelet.flip(0).reshape(1, 1, -1)
    convolved = F.conv1d(flattened, kernel, padding=wavelet.numel() // 2)
    same = convolved[..., :sample_count].reshape(*batch_shape, sample_count)
    return same.movedim(-1, axis)


class Convolutional1D(ForwardOperator):  # type: ignore[misc]  # isolated strict import boundary
    """1D convolutional seismic operator."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp", "rho"),
        output_keys=("trace",),
    )

    @classmethod
    def capabilities(cls) -> WaveCapabilityReport:
        """Return the standard immutable convolutional capability report."""
        return capability_report(cls.__name__)

    @classmethod
    def input_schema(cls) -> Mapping[str, object]:
        """Return the immutable unit-aware Agent/UI schema."""
        return input_schema(cls.__name__)

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        vp, rho = state.fetch("vp", "rho")
        if vp.ndim != 1:
            raise GeoBrainError(
                "Convolutional1D expects 1D vp",
                object_name="Convolutional1D",
                field="vp",
                expected="1D tensor",
                actual=tuple(vp.shape),
            )
        if rho.shape != vp.shape:
            raise GeoBrainError(
                "rho and vp must have matching shape",
                object_name="Convolutional1D",
                field="rho",
                expected=tuple(vp.shape),
                actual=tuple(rho.shape),
            )
        wavelet = ctx.require_wavelet()
        if not isinstance(wavelet, torch.Tensor) or wavelet.ndim != 1:
            raise GeoBrainError(
                "wavelet must be a 1D Tensor in ForwardContext.source.wavelet",
                object_name="Convolutional1D",
                field="wavelet",
                expected="1D Tensor",
                actual=type(wavelet),
            )
        require_compatible_tensors(
            (("vp", vp), ("rho", rho), ("wavelet", wavelet)),
            owner="Convolutional1D",
            allowed_dtypes=_CONVOLUTION_DTYPES,
        )
        if vp.numel() < 2:
            raise GeoBrainError(
                "Convolutional1D needs at least 2 depth samples",
                object_name="Convolutional1D",
                field="vp",
                expected="length >= 2",
                actual=vp.numel(),
            )

        impedance = vp * rho
        reflectivity = (impedance[1:] - impedance[:-1]) / (impedance[1:] + impedance[:-1])
        # Pad reflectivity to nz (one zero at top corresponds to depth-0 interface).
        zero = torch.zeros(1, dtype=reflectivity.dtype, device=reflectivity.device)
        r_full = torch.cat([zero, reflectivity])

        trace = convolve_reflectivity(r_full, wavelet, sample_axis=0)

        return ForwardOutput(data={"trace": trace})


__all__ = [
    "Convolutional1D",
    "build_convolution_matrix",
    "convolve_reflectivity",
    "ricker",
]
