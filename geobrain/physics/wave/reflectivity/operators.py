"""
AVO reflection-amplitude operators (``ForwardOperator`` layer).

Wraps the pure-math ``nn.Module`` reflectivity classes in sibling modules
:mod:`.approximations` and :mod:`.zoeppritz` (suffix-tagged ``*Reflectivity`` to
disambiguate from these operator wrappers) with the GeoBrain
``ModelState`` / ``ForwardContext`` contract:

- Inputs (:class:`ModelState`): ``vp``, ``vs``, ``rho``: 1D vertical profiles of
  length ``nz``.
- Outputs (:class:`ForwardOutput`): reflectivity ``(nz - 1, n_angles)``: one row per
  interface, one column per angle.

Two linearised approximations (Aki-Richards, Shuey) plus the exact Zoeppritz
solution. The :class:`ConvolutionalAVO` operator chains Aki-Richards reflectivity
with a per-angle 1-D wavelet convolution to produce a pre-stack synthetic trace of
shape ``(nz, n_angles)``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import ClassVar, cast

import torch

from ....core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    GeoBrainError,
    ModelState,
    ForwardOperator,
    ForwardOutput,
)
from ..capabilities import WaveCapabilityReport
from .approximations import AkiRichardsReflectivity as _AVOAkiRichards
from .approximations import ShueyReflectivity as _AVOShuey
from .base import require_compatible_tensors
from ._contracts import (
    MAX_ABS_CONTRAST,
    MAX_ABS_ERROR,
    MAX_ANGLE_DEG,
    capability_report,
    input_schema,
)
from .zoeppritz import ZoeppritzReflectivity as _AVOZoeppritz


# --- shared validation helpers ---


def _check_angles_tensor(angles: torch.Tensor) -> None:
    if not isinstance(angles, torch.Tensor) or angles.ndim != 1:
        raise GeoBrainError(
            "angles must be a 1D torch.Tensor",
            object_name="AVO",
            field="angles",
            expected="1D Tensor",
            actual=tuple(getattr(angles, "shape", ())),
        )


def _validate_angles_in_range(angles_t: torch.Tensor, op_name: str) -> None:
    _check_angles_tensor(angles_t)
    if (angles_t < 0).any() or (angles_t >= 90).any():
        raise GeoBrainError(
            f"{op_name} angles must be in [0, 90)",
            object_name=op_name,
            field="angles_deg",
            expected="[0, 90)",
            actual=angles_t.tolist(),
        )


def _validate_profiles(
    vp: torch.Tensor, vs: torch.Tensor, rho: torch.Tensor, op_name: str,
) -> None:
    if not (vp.ndim == 1 and vp.shape == vs.shape == rho.shape):
        raise GeoBrainError(
            f"{op_name} expects matching 1D profiles for vp / vs / rho",
            object_name=op_name,
            field="vp/vs/rho",
            expected="matching 1D",
            actual=(tuple(vp.shape), tuple(vs.shape), tuple(rho.shape)),
        )
    require_compatible_tensors(
        (("vp", vp), ("vs", vs), ("rho", rho)), owner=op_name
    )


def _interface_pairs(
    vp: torch.Tensor, vs: torch.Tensor, rho: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """
    Split a 1-D depth profile into ``(vp1, vs1, rho1, vp2, vs2, rho2)`` interface
    pairs of length ``nz - 1``.
    """
    return vp[:-1], vs[:-1], rho[:-1], vp[1:], vs[1:], rho[1:]


def _angles_deg_from_rad(angles_rad: torch.Tensor) -> torch.Tensor:
    return angles_rad * (180.0 / math.pi)


def _prepare_angles(
    angles_deg: torch.Tensor | list[float], op_name: str
) -> tuple[torch.Tensor, bool]:
    """Preserve live Tensor metadata; normalize factory/list metadata."""
    if isinstance(angles_deg, torch.Tensor):
        angles_t = angles_deg.detach().clone()
        require_compatible_tensors((("angles_deg", angles_t),), owner=op_name)
        live_tensor = True
    else:
        angles_t = torch.as_tensor(angles_deg, dtype=torch.float32)
        live_tensor = False
    _validate_angles_in_range(angles_t, op_name)
    return angles_t * (math.pi / 180.0), live_tensor


def _runtime_angles_deg(
    angles_rad: torch.Tensor,
    *,
    live_tensor: bool,
    reference: torch.Tensor,
    op_name: str,
) -> torch.Tensor:
    """Reject mixed live angle metadata; normalize only factory metadata."""
    if live_tensor:
        require_compatible_tensors(
            (("model", reference), ("angles_deg", angles_rad)), owner=op_name
        )
        runtime = angles_rad
    else:
        runtime = angles_rad.to(dtype=reference.dtype, device=reference.device)
    return _angles_deg_from_rad(runtime)


def _approximation_validation(
    vp: torch.Tensor,
    vs: torch.Tensor,
    rho: torch.Tensor,
    angles_deg: torch.Tensor,
) -> dict[str, object]:
    """Report a bound only when the executed row is inside its measured matrix."""
    contrasts = torch.stack(
        tuple(
            ((field[1:] - field[:-1]) / ((field[1:] + field[:-1]) / 2)).abs().max()
            for field in (vp, vs, rho)
        )
    )
    measured_contrast = contrasts.detach().max().item()
    measured_angle = angles_deg.detach().max().item()
    within = bool(
        measured_contrast <= MAX_ABS_CONTRAST + 1e-7
        and measured_angle <= MAX_ANGLE_DEG + 1e-5
    )
    return {
        "within_validated_domain": within,
        "max_abs_contrast": MAX_ABS_CONTRAST,
        "max_angle_deg": MAX_ANGLE_DEG,
        "max_abs_error": MAX_ABS_ERROR if within else None,
    }


class _ReflectivityDiscovery:
    """Standard immutable discovery surface shared by public operators."""

    _discovery_name: ClassVar[str]

    @classmethod
    def capabilities(cls) -> WaveCapabilityReport:
        return capability_report(cls._discovery_name)

    @classmethod
    def input_schema(cls) -> Mapping[str, object]:
        return input_schema(cls._discovery_name)


# --- operators ---


class AkiRichards(_ReflectivityDiscovery, ForwardOperator):  # type: ignore[misc]  # isolated strict import boundary
    """
    3-term Aki-Richards approximation. Inputs vp, vs, rho; output ``reflectivity``.

    Delegates the per-interface formula to
    :class:`geobrain.physics.wave.reflectivity.AkiRichards`.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp", "vs", "rho"),
        output_keys=("reflectivity",),
    )
    _discovery_name = "AkiRichards"

    def __init__(self, angles_deg: torch.Tensor | list[float]) -> None:
        super().__init__()
        angles_rad, self._angles_are_live_tensor = _prepare_angles(
            angles_deg, "AkiRichards"
        )
        self.register_buffer(
            "angles_rad", angles_rad, persistent=False,
        )
        self._model = _AVOAkiRichards()

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        vp, vs, rho = state.fetch("vp", "vs", "rho")
        _validate_profiles(vp, vs, rho, "AkiRichards")

        vp1, vs1, rho1, vp2, vs2, rho2 = _interface_pairs(vp, vs, rho)
        angles_deg = _runtime_angles_deg(
            cast(torch.Tensor, self.angles_rad),
            live_tensor=self._angles_are_live_tensor,
            reference=vp,
            op_name="AkiRichards",
        )
        # wave/reflectivity Shuey/AkiRichards/Zoeppritz return shape (n_angles, n_iface);
        # operator contract is (n_iface, n_angles), so transpose.
        R = self._model(vp1, vs1, rho1, vp2, vs2, rho2, angles_deg).T
        return ForwardOutput(
            data={"reflectivity": R},
            metadata={
                "angles_deg": angles_deg.tolist(),
                "approximation_validation": _approximation_validation(
                    vp, vs, rho, angles_deg
                ),
            },
        )


class Shuey(_ReflectivityDiscovery, ForwardOperator):  # type: ignore[misc]  # isolated strict import boundary
    """
    3-term (or 2-term) Shuey approximation. Inputs vp, vs, rho.

    Delegates to :class:`geobrain.physics.wave.reflectivity.Shuey` for the 3-term
    case; the 2-term variant uses the wave model's ``avo_attributes``
    helper to get the ``(R0, G, F)`` decomposition and drops the
    ``F·(tan²θ − sin²θ)`` curvature term.

    Args:
        angles_deg: incidence angles [deg].
        terms: 2-term (intercept+gradient) or 3-term Shuey form.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp", "vs", "rho"),
        output_keys=("reflectivity",),
    )
    _discovery_name = "Shuey"

    def __init__(
        self, angles_deg: torch.Tensor | list[float], *, terms: int = 3,
    ) -> None:
        super().__init__()
        if terms not in (2, 3):
            raise GeoBrainError(
                "Shuey terms must be 2 or 3",
                object_name="Shuey",
                field="terms",
                expected="2 or 3",
                actual=terms,
            )
        angles_rad, self._angles_are_live_tensor = _prepare_angles(
            angles_deg, "Shuey"
        )
        self.register_buffer(
            "angles_rad", angles_rad, persistent=False,
        )
        self.terms = terms
        self._model = _AVOShuey()

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        vp, vs, rho = state.fetch("vp", "vs", "rho")
        _validate_profiles(vp, vs, rho, "Shuey")

        vp1, vs1, rho1, vp2, vs2, rho2 = _interface_pairs(vp, vs, rho)
        angles_deg = _runtime_angles_deg(
            cast(torch.Tensor, self.angles_rad),
            live_tensor=self._angles_are_live_tensor,
            reference=vp,
            op_name="Shuey",
        )

        if self.terms == 3:
            R = self._model(vp1, vs1, rho1, vp2, vs2, rho2, angles_deg).T
        else:
            R0, G, _F = self._model.avo_attributes(vp1, vs1, rho1, vp2, vs2, rho2)
            sin2 = torch.sin(angles_deg * (math.pi / 180.0)).pow(2)
            # R0, G: (n_iface,); sin2: (n_angles,) → broadcast to (n_iface, n_angles).
            R = R0.unsqueeze(-1) + G.unsqueeze(-1) * sin2.unsqueeze(0)
        return ForwardOutput(
            data={"reflectivity": R},
            metadata={
                "angles_deg": angles_deg.tolist(),
                "terms": self.terms,
                "approximation_validation": _approximation_validation(
                    vp, vs, rho, angles_deg
                ),
            },
        )


class Zoeppritz(_ReflectivityDiscovery, ForwardOperator):  # type: ignore[misc]  # isolated strict import boundary
    """
    Exact Zoeppritz P-P reflection coefficients.

    Delegates to :class:`geobrain.physics.wave.reflectivity.Zoeppritz`, which uses
    the closed-form determinant expression and promotes to complex in
    the post-critical regime. For pre-critical angles the output is real
    and matches the linearised approximations to first order; the test
    suite documents typical agreement.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp", "vs", "rho"),
        output_keys=("reflectivity",),
    )
    _discovery_name = "Zoeppritz"

    def __init__(self, angles_deg: torch.Tensor | list[float]) -> None:
        super().__init__()
        angles_rad, self._angles_are_live_tensor = _prepare_angles(
            angles_deg, "Zoeppritz"
        )
        self.register_buffer(
            "angles_rad", angles_rad, persistent=False,
        )
        self._model = _AVOZoeppritz()

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        vp, vs, rho = state.fetch("vp", "vs", "rho")
        _validate_profiles(vp, vs, rho, "Zoeppritz")

        vp1, vs1, rho1, vp2, vs2, rho2 = _interface_pairs(vp, vs, rho)
        angles_deg = _runtime_angles_deg(
            cast(torch.Tensor, self.angles_rad),
            live_tensor=self._angles_are_live_tensor,
            reference=vp,
            op_name="Zoeppritz",
        )
        R = self._model(vp1, vs1, rho1, vp2, vs2, rho2, angles_deg).T
        return ForwardOutput(
            data={"reflectivity": R},
            metadata={"angles_deg": angles_deg.tolist()},
        )


class ConvolutionalAVO(_ReflectivityDiscovery, ForwardOperator):  # type: ignore[misc]  # isolated strict import boundary
    """
    Aki-Richards reflectivity convolved with a source wavelet, per angle.

    Forward chain (internal):
        vp, vs, rho  →  Aki-Richards R(θ) via wave/reflectivity  →
        zero-pad to ``nz``  →  per-angle 1-D wavelet conv  →
        trace of shape ``(nz, n_angles)``.

    Inputs (ModelState): ``vp``, ``vs``, ``rho``, 1D profiles of length ``nz``.
    Output (ForwardOutput):  ``data["trace"]`` of shape ``(nz, n_angles)``.

    The reflectivity step delegates to
    :class:`geobrain.physics.wave.reflectivity.AkiRichards`; only the
    wavelet-convolution chain lives here.

    Args:
        angles_deg: incidence angles of the AVO gather [deg].
        wavelet: 1-D wavelet convolved onto the reflectivity series.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp", "vs", "rho"),
        output_keys=("trace",),
    )
    _discovery_name = "ConvolutionalAVO"

    def __init__(
        self,
        angles_deg: torch.Tensor | list[float],
        wavelet: torch.Tensor,
    ) -> None:
        super().__init__()
        angles_rad, self._angles_are_live_tensor = _prepare_angles(
            angles_deg, "ConvolutionalAVO"
        )
        if not isinstance(wavelet, torch.Tensor) or wavelet.ndim != 1:
            raise GeoBrainError(
                "wavelet must be a 1D torch.Tensor",
                object_name="ConvolutionalAVO",
                field="wavelet",
                expected="1D Tensor",
                actual=tuple(getattr(wavelet, "shape", ())),
            )
        self.register_buffer(
            "angles_rad", angles_rad, persistent=False,
        )
        self.register_buffer("wavelet", wavelet.detach().clone(), persistent=False)
        self._reflectivity = _AVOAkiRichards()

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        vp, vs, rho = state.fetch("vp", "vs", "rho")
        _validate_profiles(vp, vs, rho, "ConvolutionalAVO")
        wavelet = cast(torch.Tensor, self.wavelet)
        require_compatible_tensors(
            (("vp", vp), ("vs", vs), ("rho", rho), ("wavelet", wavelet)),
            owner="ConvolutionalAVO",
        )
        if vp.numel() < 2:
            raise GeoBrainError(
                "ConvolutionalAVO needs at least 2 depth samples",
                object_name="ConvolutionalAVO",
                field="vp",
                expected="length >= 2",
                actual=vp.numel(),
            )

        vp1, vs1, rho1, vp2, vs2, rho2 = _interface_pairs(vp, vs, rho)
        angles_deg = _runtime_angles_deg(
            cast(torch.Tensor, self.angles_rad),
            live_tensor=self._angles_are_live_tensor,
            reference=vp,
            op_name="ConvolutionalAVO",
        )
        # Reflectivity: wave/reflectivity returns (n_angles, n_iface). Transpose so
        # the time/depth axis is first, matching the per-angle conv that
        # follows.
        R = self._reflectivity(vp1, vs1, rho1, vp2, vs2, rho2, angles_deg).T  # (n_iface, n_angles)

        # Pad with a leading zero so reflectivity aligns at the top interface.
        n_angles = R.shape[1]
        zero = torch.zeros(1, n_angles, dtype=R.dtype, device=R.device)
        R_full = torch.cat([zero, R], dim=0)  # (nz, n_angles)

        # Import locally to avoid package-initialization coupling: the generic
        # convolutional module imports reflectivity discovery contracts.
        from ..convolutional import convolve_reflectivity

        trace = convolve_reflectivity(R_full, wavelet, sample_axis=0)

        return ForwardOutput(
            data={"trace": trace},
            metadata={
                "angles_deg": angles_deg.tolist(),
                "approximation_validation": _approximation_validation(
                    vp, vs, rho, angles_deg
                ),
            },
        )
