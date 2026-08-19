# pyright: reportPrivateImportUsage=false
"""
Marine controlled-source EM (CSEM) 1D forward.

VMD source over a 1D layered earth. Observation: ``bz`` (vertical
magnetic flux density) at receiver positions across a frequency sweep.
Differentiability: FULL_AUTOGRAD. Time convention: ``e^{+iωt}``, the shared
layered kernel uses ``+iωμ₀σ`` (verified: MT1D half-space impedance phase = +45°),
so CSEM1D / MT1D / MT3D / TEM1D's frequency kernel and FDEM3D all agree. None
conjugates its public output; comparisons that require ``e^{-iωt}`` use an
explicit convention adapter.

Source: the ``frequency_domain/csem1d/`` 7-file subpackage, operator
+ kernel in one file.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import cast

import torch
from geobrain.physics.em.capabilities import EMOperatorDiscovery

from geobrain.core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ErrorCode,
    ForwardContext,
    ModelState,
    ForwardOperator,
    ForwardOutput,
)
from geobrain.physics.em.results import FieldComponent
from geobrain.physics.em.surveys import (
    EMReceiver,
    EMSource,
    FrequencyDomainSurvey,
)
from geobrain.physics.em.errors import (
    EMCapabilityError,
    EMContractError,
    EMNumericsError,
)
from geobrain.physics.em.numerics.hankel.dlf import (
    _direct_hankel_j0_with_error,
    axial_hankel_j0,
    dlf_hankel,
)
from geobrain.physics.em.frequency_domain._csem1d_kernels import (  # noqa: F401  re-export: split section
    _dimensionless_induction_log_shift,
    _dimensionless_integrand_bz,
    _dimensionless_te_reflection,
    _finite_complex_polar_from_log,
    _integrand_bz,
    _needs_scale_safe_mixed_hessian,
    _scale_dimensionless_secondary,
    _scale_safe_tanh_thickness_jet,
    _vmd_primary_bz,
)


_DIRECT_RELATIVE_TOLERANCE = 1.0e-7
_DLF_DIRECT_AGREEMENT_TOLERANCE = 1.0e-6
_DIRECT_COMPARISON_MAX_AIR_RATIO = 32.0


def _is_finite_real(value: object) -> bool:
    """Return whether ``value`` is a non-Boolean finite real scalar."""
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _canonical_position(value: object, *, object_name: str) -> tuple[float, float, float]:
    """Validate and freeze one public Cartesian position in metres."""
    coordinates = value if isinstance(value, (tuple, list)) else None
    if (
        coordinates is None
        or len(coordinates) != 3
        or any(not _is_finite_real(coordinate) for coordinate in coordinates)
    ):
        actual = (
            [str(coordinate) for coordinate in coordinates]
            if coordinates is not None
            else str(value)
        )
        raise EMContractError(
            "CSEM coordinates must be finite Cartesian metres",
            details={"field": "position", "actual": actual},
            object_name=object_name,
            field="position",
            expected="three finite coordinates in metres",
            actual=actual,
        )
    return cast(
        tuple[float, float, float],
        tuple(float(coordinate) for coordinate in coordinates),
    )


@dataclass(frozen=True, slots=True)
class VMDSource(EMSource):  # type: ignore[misc,unused-ignore]  # skip-mode base is Any
    """
    Vertical Magnetic Dipole source.

    Extends ``EMSource`` (``position``) with the dipole moment. It
    ships only the ``z``-orientation; if a future update adds
    tilted VMDs, an ``orientation`` field gets added here.

    Attributes:
        position: vertical magnetic dipole location [m].
        magnetic_moment_am2: dipole moment [A*m^2].
    """

    magnetic_moment_am2: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position",
            _canonical_position(self.position, object_name="VMDSource"),
        )
        if not _is_finite_real(self.magnetic_moment_am2) or float(self.magnetic_moment_am2) <= 0.0:
            raise EMContractError(
                "VMD magnetic moment must be finite and positive",
                details={
                    "field": "magnetic_moment_am2",
                    "actual": str(self.magnetic_moment_am2),
                },
                object_name="VMDSource",
                field="magnetic_moment_am2",
                expected="finite value > 0 A m^2",
                actual=str(self.magnetic_moment_am2),
            )
        object.__setattr__(self, "magnetic_moment_am2", float(self.magnetic_moment_am2))


@dataclass(frozen=True, slots=True)
class CSEMReceiver(EMReceiver):  # type: ignore[misc,unused-ignore]  # skip-mode base is Any
    """CSEM receiver: measured field component on top of ``EMReceiver``.

    Defaults to ``BZ`` (the canonical observable for a VMD source).

    Attributes:
        position: receiver location [m].
        component: recorded field component (``'ex'``...).
    """

    component: FieldComponent = FieldComponent.BZ

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position",
            _canonical_position(self.position, object_name="CSEMReceiver"),
        )


@dataclass(frozen=True, slots=True)
class MarineCSEM1DSurvey(
    FrequencyDomainSurvey,  # type: ignore[misc,unused-ignore]  # skip-mode base is Any
):
    """
    Marine CSEM 1D survey.

    Sources must be ``VMDSource``; receivers must be ``CSEMReceiver``
    (validated at operator construction).

    Attributes:
        sources / receivers: acquisition tables.
        frequencies: transmit frequencies [Hz].
        surface_z_m: sea-surface elevation [m].
    """

    surface_z_m: float = 0.0

    def __post_init__(self) -> None:
        FrequencyDomainSurvey.__post_init__(self)
        if not isinstance(self.sources, (tuple, list)):
            raise EMContractError(
                "CSEM1D sources must be a source sequence",
                details={"field": "sources", "actual": type(self.sources).__name__},
                object_name="MarineCSEM1DSurvey",
                field="sources",
                expected="non-empty tuple or list of VMDSource",
                actual=type(self.sources).__name__,
            )
        if not self.sources:
            raise EMContractError(
                "CSEM1D survey requires at least one source",
                details={"field": "sources", "count": 0},
                object_name="MarineCSEM1DSurvey",
                field="sources",
                expected="one or more sources",
                actual=0,
            )
        if not isinstance(self.receivers, (tuple, list)):
            raise EMContractError(
                "CSEM1D receivers must be a receiver sequence",
                details={"field": "receivers", "actual": type(self.receivers).__name__},
                object_name="MarineCSEM1DSurvey",
                field="receivers",
                expected="non-empty tuple or list of CSEMReceiver",
                actual=type(self.receivers).__name__,
            )
        if not self.receivers:
            raise EMContractError(
                "CSEM1D survey requires at least one receiver",
                details={"field": "receivers", "count": 0},
                object_name="MarineCSEM1DSurvey",
                field="receivers",
                expected="one or more receivers",
                actual=0,
            )
        for index, source in enumerate(self.sources):
            if not isinstance(source, VMDSource):
                raise EMContractError(
                    "CSEM1D sources must contain only VMDSource records",
                    details={"index": index, "actual": type(source).__name__},
                    object_name="MarineCSEM1DSurvey",
                    field="sources",
                    expected="VMDSource records",
                    actual=type(source).__name__,
                )
        for index, receiver in enumerate(self.receivers):
            if not isinstance(receiver, CSEMReceiver):
                raise EMContractError(
                    "CSEM1D receivers must contain only CSEMReceiver records",
                    details={"index": index, "actual": type(receiver).__name__},
                    object_name="MarineCSEM1DSurvey",
                    field="receivers",
                    expected="CSEMReceiver records",
                    actual=type(receiver).__name__,
                )
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "receivers", tuple(self.receivers))
        frequencies = self.frequencies
        frequencies_are_sequence = isinstance(frequencies, (tuple, list))
        valid_frequencies = (
            frequencies_are_sequence
            and bool(frequencies)
            and all(_is_finite_real(value) for value in frequencies)
        )
        numeric = tuple(float(value) for value in frequencies) if valid_frequencies else ()
        valid_frequencies = (
            valid_frequencies
            and all(value > 0.0 for value in numeric)
            and all(later > earlier for earlier, later in zip(numeric, numeric[1:]))
        )
        if not valid_frequencies:
            raise EMContractError(
                "CSEM1D frequencies must be finite, positive, and strictly increasing",
                details={
                    "field": "frequencies",
                    "actual": (
                        [str(value) for value in frequencies]
                        if frequencies_are_sequence
                        else str(frequencies)
                    ),
                },
                object_name="MarineCSEM1DSurvey",
                field="frequencies",
                expected="non-empty finite positive strictly increasing sequence",
                actual=(
                    tuple(str(value) for value in frequencies)
                    if frequencies_are_sequence
                    else str(frequencies)
                ),
            )
        object.__setattr__(self, "frequencies", numeric)
        if not _is_finite_real(self.surface_z_m):
            raise EMContractError(
                "CSEM1D surface elevation must be finite",
                details={"field": "surface_z_m", "actual": str(self.surface_z_m)},
                object_name="MarineCSEM1DSurvey",
                field="surface_z_m",
                expected="finite coordinate in metres",
                actual=str(self.surface_z_m),
            )
        object.__setattr__(self, "surface_z_m", float(self.surface_z_m))
        survey_records: list[tuple[str, int, EMSource | EMReceiver]] = []
        survey_records.extend(
            ("source", index, source) for index, source in enumerate(self.sources)
        )
        survey_records.extend(
            ("receiver", index, receiver) for index, receiver in enumerate(self.receivers)
        )
        for kind, index, record in survey_records:
            position = record.position
            position_z_m = float(position[2])
            surface_z_m = float(self.surface_z_m)
            if position_z_m > surface_z_m:
                raise EMContractError(
                    f"CSEM1D {kind} must be in air or on the surface",
                    details={
                        "field": f"{kind}s[{index}].position[2]",
                        "z_m": position_z_m,
                        "surface_z_m": surface_z_m,
                        "legal_domain": "z <= surface_z_m",
                    },
                    object_name="MarineCSEM1DSurvey",
                    field=f"{kind}s[{index}].position[2]",
                    expected=f"z <= {surface_z_m}",
                    actual=position_z_m,
                )
        for source_index, source in enumerate(self.sources):
            for receiver_index, receiver in enumerate(self.receivers):
                if source.position == receiver.position:
                    raise EMContractError(
                        "CSEM1D source and receiver cannot coincide",
                        details={
                            "source_index": source_index,
                            "receiver_index": receiver_index,
                            "position": [float(value) for value in source.position],
                        },
                        object_name="MarineCSEM1DSurvey",
                        field="receivers",
                        expected="non-coincident source/receiver pairs",
                        actual=[float(value) for value in source.position],
                    )


# ---------------------------------------------------------------------------
# Private kernel (ported from the frequency_domain/csem1d/kernel.py)
# # ``e^{+iωt}`` time convention: wavenumber uses ``+1j ω μ₀ σ`` (same as MT1D and
# the shared ``layered_te_reflection`` kernel; no output conjugation). All kernels operate
# in complex128 for the Hankel-domain spectrum and broadcast over
# ``(n_freq, n_lambda)``; the layered earth axis is collapsed by the Wait
# recursion before returning to the caller.
# ---------------------------------------------------------------------------




















class CSEM1D(EMOperatorDiscovery, ForwardOperator):  # type: ignore[misc,unused-ignore]  # skip-mode base is Any
    """Marine CSEM 1D forward: VMD source over a layered earth.

    Maps a :class:`~geobrain.core.ModelState` containing layered-earth
    conductivity (``sigma``) and layer thicknesses (``thickness``,
    ``n_layers - 1`` entries; empty for a half-space) to a
    :class:`~geobrain.core.ForwardOutput` containing the vertical magnetic
    flux density ``B_z`` at every (receiver, frequency) pair.

    Storage convention: complex output is emitted as a single native
    complex tensor under ``data["bz"]`` of shape
    ``(n_receivers, n_frequencies)`` (the platform-wide complex-data
    contract, one key per complex channel, not split real/imag).

    Prefactor convention::

        B_z = B_z^primary  +  μ₀ · (m / 4π) · ∫₀^∞ f_sec(λ) J_0(λ r) dλ

    where ``f_sec`` is the secondary (reflected) integrand returned by
    :func:`_integrand_bz` and ``B_z^primary`` is the free-space VMD
    closed form returned by :func:`_vmd_primary_bz`. Rolling the primary
    into the integrand as ``exp(-λ|Δz|)`` and handing it to the DLF
    would impose a ~1e-3 accuracy floor on
    primary-dominated configurations (HCP airborne, HEM). Splitting
    the integral drops the floor to ~1e-7 (machine precision on the
    primary part; DLF accuracy on the secondary).

    Construction-time validation: every source must be a :class:`VMDSource`
    and every receiver must be a :class:`CSEMReceiver`. Receivers must
    request the ``BZ`` component (the only component implemented).
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("sigma", "thickness"),
        output_keys=("bz",),
    )

    def __init__(self, survey: MarineCSEM1DSurvey) -> None:
        super().__init__()
        if not isinstance(survey, MarineCSEM1DSurvey):
            raise EMContractError(
                f"CSEM1D requires a MarineCSEM1DSurvey, got {type(survey).__name__}",
                details={"survey_type": type(survey).__name__},
                object_name="CSEM1D",
                field="survey",
                expected="MarineCSEM1DSurvey",
                actual=type(survey).__name__,
            )
        for i, src in enumerate(survey.sources):
            if not isinstance(src, VMDSource):
                raise EMContractError(
                    f"CSEM1D sources must be VMDSource; sources[{i}] is {type(src).__name__}",
                    details={
                        "source_index": i,
                        "source_type": type(src).__name__,
                    },
                    object_name="CSEM1D",
                    field=f"sources[{i}]",
                    expected="VMDSource",
                    actual=type(src).__name__,
                )
        for i, rcv in enumerate(survey.receivers):
            if not isinstance(rcv, CSEMReceiver):
                raise EMContractError(
                    f"CSEM1D receivers must be CSEMReceiver; "
                    f"receivers[{i}] is {type(rcv).__name__}",
                    details={
                        "receiver_index": i,
                        "receiver_type": type(rcv).__name__,
                    },
                    object_name="CSEM1D",
                    field=f"receivers[{i}]",
                    expected="CSEMReceiver",
                    actual=type(rcv).__name__,
                )
            if rcv.component != FieldComponent.BZ:
                raise EMContractError(
                    f"CSEM1D only supports the BZ component; "
                    f"receivers[{i}].component = {rcv.component}",
                    details={
                        "receiver_index": i,
                        "component": str(rcv.component),
                        "supported": ["bz"],
                    },
                    object_name="CSEM1D",
                    field=f"receivers[{i}].component",
                    expected="bz",
                    actual=str(rcv.component),
                )
        self.survey = survey
        self._sources = tuple(cast(VMDSource, source) for source in survey.sources)
        self._receivers = tuple(cast(CSEMReceiver, receiver) for receiver in survey.receivers)

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        sigma, thickness = state.fetch("sigma", "thickness")

        for field_name, tensor in (("sigma", sigma), ("thickness", thickness)):
            if tensor.is_nested or tensor.layout != torch.strided:
                raise EMContractError(
                    "CSEM1D model tensors must be dense and strided",
                    details={
                        "field": field_name,
                        "is_nested": tensor.is_nested,
                        "layout": str(tensor.layout),
                    },
                    object_name="CSEM1D",
                    field=field_name,
                    expected="dense strided torch.Tensor",
                    actual="nested tensor" if tensor.is_nested else str(tensor.layout),
                )
        if sigma.device != thickness.device:
            raise EMCapabilityError(
                "CSEM1D model tensors must share one device",
                details={
                    "sigma_device": str(sigma.device),
                    "thickness_device": str(thickness.device),
                },
                object_name="CSEM1D",
                field="state.device",
                expected="matching tensor devices",
                actual=[str(sigma.device), str(thickness.device)],
                code=ErrorCode.DEVICE_UNAVAILABLE,
            )
        if sigma.device.type not in {"cpu", "cuda"}:
            raise EMCapabilityError(
                "CSEM1D execution supports only CPU and CUDA tensors",
                details={
                    "operator": "CSEM1D",
                    "device": str(sigma.device),
                    "supported": ["cpu", "cuda"],
                },
                object_name="CSEM1D",
                field="state.device",
                expected=["cpu", "cuda"],
                actual=str(sigma.device),
                code=ErrorCode.DEVICE_UNAVAILABLE,
            )
        if sigma.dtype != torch.float64 or thickness.dtype != torch.float64:
            raise EMCapabilityError(
                "CSEM1D requires float64 model tensors",
                details={
                    "operator": "CSEM1D",
                    "sigma_dtype": str(sigma.dtype),
                    "thickness_dtype": str(thickness.dtype),
                    "supported": ["torch.float64"],
                },
                object_name="CSEM1D",
                field="state.dtype",
                expected="matching torch.float64 tensors",
                actual=[str(sigma.dtype), str(thickness.dtype)],
                code=ErrorCode.DTYPE_UNSUPPORTED,
            )
        if sigma.ndim != 1 or sigma.numel() == 0:
            raise EMContractError(
                "CSEM1D conductivity must be a non-empty 1-D tensor",
                details={"field": "sigma", "shape": list(sigma.shape)},
                object_name="CSEM1D",
                field="sigma",
                expected="non-empty 1-D tensor",
                actual=list(sigma.shape),
            )
        expected_thickness = int(sigma.numel()) - 1
        if thickness.ndim != 1 or thickness.numel() != expected_thickness:
            raise EMContractError(
                "CSEM1D thickness must contain one value per finite layer",
                details={
                    "field": "thickness",
                    "shape": list(thickness.shape),
                    "expected_count": expected_thickness,
                },
                object_name="CSEM1D",
                field="thickness",
                expected=f"1-D tensor with {expected_thickness} elements",
                actual=list(thickness.shape),
            )
        if not bool(torch.isfinite(sigma).all()) or not bool((sigma > 0.0).all()):
            raise EMContractError(
                "CSEM1D conductivity values must be finite and positive",
                details={"field": "sigma"},
                object_name="CSEM1D",
                field="sigma",
                expected="finite values > 0 S/m",
                actual="invalid tensor values",
            )
        if not bool(torch.isfinite(thickness).all()) or not bool((thickness > 0.0).all()):
            raise EMContractError(
                "CSEM1D thickness values must be finite and positive",
                details={"field": "thickness"},
                object_name="CSEM1D",
                field="thickness",
                expected="finite values > 0 m",
                actual="invalid tensor values",
            )

        # Build omega as a tensor on sigma's device/dtype for consistent dispatch.
        device = sigma.device
        omega = (
            2.0
            * math.pi
            * torch.tensor(self.survey.frequencies, dtype=torch.float64, device=device)
        )  # (n_freq,)
        if not bool(torch.isfinite(omega).all()) or not bool((omega > 0.0).all()):
            raise EMNumericsError(
                "CSEM1D angular-frequency conversion is not representable",
                details={
                    "operator": "CSEM1D",
                    "stage": "angular_frequency",
                    "frequencies_hz": [str(value) for value in self.survey.frequencies],
                },
                object_name="CSEM1D",
                field="frequencies",
            )
        n_freq = int(omega.numel())
        n_rcv = len(self._receivers)

        # Ships single-source goldens; the loop supports multi-source
        # so we fold contributions linearly across sources for forward-compat.
        per_rcv: list[torch.Tensor] = []
        transform_methods: set[str] = set()
        used_filter_assets = False
        for rcv in self._receivers:
            rx_x, rx_y, rx_z = rcv.position

            field = torch.zeros(n_freq, dtype=torch.complex128, device=device)
            for src in self._sources:
                sx, sy, sz = src.position
                dx = float(rx_x) - float(sx)
                dy = float(rx_y) - float(sy)
                if not math.isfinite(dx) or not math.isfinite(dy):
                    raise EMNumericsError(
                        "CSEM1D horizontal coordinate difference is not representable",
                        details={
                            "operator": "CSEM1D",
                            "stage": "coordinate_difference",
                            "difference": [str(dx), str(dy)],
                        },
                        object_name="CSEM1D",
                        field="survey",
                    )
                r = math.hypot(dx, dy)
                if not math.isfinite(r):
                    raise EMNumericsError(
                        "CSEM1D source-receiver offset is not representable",
                        details={
                            "operator": "CSEM1D",
                            "stage": "source_receiver_offset",
                            "offset_m": str(r),
                        },
                        object_name="CSEM1D",
                        field="survey",
                    )

                # Use the air propagation length as the wavenumber scale. Both
                # direct quadrature and DLF then integrate in q=lambda*L, so
                # neither a very large geometry nor a compensating source moment
                # can disappear through a common physical-lambda underflow.
                vertical_scale = max(
                    abs(float(rx_z) - float(sz)),
                    abs(float(self.survey.surface_z_m) - float(sz)),
                    abs(float(self.survey.surface_z_m) - float(rx_z)),
                )
                air_path_length = 2.0 * float(self.survey.surface_z_m) - float(sz) - float(rx_z)
                if air_path_length < 0.0 or not math.isfinite(air_path_length):
                    raise EMNumericsError(
                        "CSEM1D air propagation length is not representable",
                        details={
                            "operator": "CSEM1D",
                            "stage": "dimensionless_geometry",
                            "air_path_length_m": str(air_path_length),
                        },
                        object_name="CSEM1D",
                        field="survey",
                    )
                axial_limit = torch.finfo(torch.float64).eps ** 0.25 * vertical_scale
                axial_ratio = r / axial_limit if axial_limit > 0.0 else math.inf

                length_scale = air_path_length if air_path_length > 0.0 else r
                air_path_ratio = -air_path_length / length_scale
                induction_log_shift = _dimensionless_induction_log_shift(
                    omega,
                    sigma,
                    length_scale_m=length_scale,
                )

                def dimensionless_integrand(q: torch.Tensor) -> torch.Tensor:
                    return _dimensionless_integrand_bz(
                        q,
                        omega,
                        sigma,
                        thickness,
                        length_scale_m=length_scale,
                        air_path_ratio=air_path_ratio,
                        log_induction_shift=induction_log_shift,
                    )

                dimensionless_secondary: torch.Tensor | None = None
                secondary_field: torch.Tensor | None = None
                if r == 0.0:
                    if _needs_scale_safe_mixed_hessian(omega, sigma, thickness):

                        def scaled_axial_secondary(
                            conductivity: torch.Tensor,
                            layer_thickness: torch.Tensor,
                        ) -> torch.Tensor:
                            local_shift = _dimensionless_induction_log_shift(
                                omega,
                                conductivity,
                                length_scale_m=length_scale,
                            )

                            def local_integrand(q: torch.Tensor) -> torch.Tensor:
                                return _dimensionless_integrand_bz(
                                    q,
                                    omega,
                                    conductivity,
                                    layer_thickness,
                                    length_scale_m=length_scale,
                                    air_path_ratio=air_path_ratio,
                                    log_induction_shift=local_shift,
                                )

                            local_secondary = axial_hankel_j0(
                                local_integrand,
                                dtype=torch.float64,
                                device=device,
                                relative_tolerance=_DIRECT_RELATIVE_TOLERANCE,
                                max_refinements=8,
                            )
                            return _scale_dimensionless_secondary(
                                local_secondary,
                                length_scale_m=length_scale,
                                magnetic_moment_am2=src.magnetic_moment_am2,
                                log_amplitude_adjustment=-local_shift.reshape(-1),
                            )

                        # Separate the pure-sigma and pure-thickness graphs and
                        # restore every mixed edge from a forward thickness JVP
                        # evaluated after Hankel integration and SI scaling.  A
                        # zero-primal carrier makes both reverse Hessian orders
                        # traverse that same representable coefficient while
                        # retaining the exact ordinary forward bits.
                        thickness_anchor = thickness.detach()
                        thickness_graph = scaled_axial_secondary(
                            sigma.detach(),
                            thickness,
                        )
                        sigma_graph: torch.Tensor | None = None
                        tangent_graphs: list[torch.Tensor] = []
                        for thickness_index in range(int(thickness.numel())):
                            tangent = torch.zeros_like(thickness_anchor)
                            tangent[thickness_index] = 1.0
                            current_sigma_graph, tangent_graph = (
                                torch.autograd.functional.jvp(
                                    lambda layer_thickness: scaled_axial_secondary(
                                        sigma,
                                        layer_thickness,
                                    ),
                                    (thickness_anchor,),
                                    (tangent,),
                                    create_graph=True,
                                    strict=True,
                                )
                            )
                            if sigma_graph is None:
                                sigma_graph = current_sigma_graph
                            tangent_graphs.append(tangent_graph)
                        assert sigma_graph is not None
                        pure_secondary = sigma_graph.detach() + (
                            thickness_graph - thickness_graph.detach()
                        ) + (sigma_graph - sigma_graph.detach())
                        preliminary_secondary = pure_secondary
                        for thickness_index, tangent_graph in enumerate(
                            tangent_graphs
                        ):
                            preliminary_secondary = preliminary_secondary + (
                                tangent_graph - tangent_graph.detach()
                            ) * (
                                thickness[thickness_index]
                                - thickness_anchor[thickness_index]
                            )

                        # Take the mixed coefficients through the thickness-
                        # first graph, whose SI-scaled backward path preserves
                        # the sign when the exact result rounds to zero.  The
                        # returned field then uses those detached coefficients
                        # in a local bilinear carrier, so the transposed Hessian
                        # traverses the identical nearest-binary64 values.
                        mixed_real: list[list[torch.Tensor]] = [
                            [] for _ in tangent_graphs
                        ]
                        mixed_imag: list[list[torch.Tensor]] = [
                            [] for _ in tangent_graphs
                        ]
                        for frequency_index in range(
                            int(preliminary_secondary.numel())
                        ):
                            real_thickness_gradient = torch.autograd.grad(
                                preliminary_secondary.real[frequency_index],
                                thickness,
                                create_graph=True,
                                retain_graph=True,
                            )[0]
                            imag_thickness_gradient = torch.autograd.grad(
                                preliminary_secondary.imag[frequency_index],
                                thickness,
                                create_graph=True,
                                retain_graph=True,
                            )[0]
                            for thickness_index in range(int(thickness.numel())):
                                mixed_real[thickness_index].append(
                                    torch.autograd.grad(
                                        real_thickness_gradient[thickness_index],
                                        sigma,
                                        retain_graph=True,
                                    )[0].detach()
                                )
                                mixed_imag[thickness_index].append(
                                    torch.autograd.grad(
                                        imag_thickness_gradient[thickness_index],
                                        sigma,
                                        retain_graph=True,
                                    )[0].detach()
                                )

                        secondary_field = pure_secondary
                        sigma_delta = sigma - sigma.detach()
                        for thickness_index in range(int(thickness.numel())):
                            thickness_delta = (
                                thickness[thickness_index]
                                - thickness_anchor[thickness_index]
                            )

                            def mixed_channel_carrier(
                                coefficient_rows: list[torch.Tensor],
                            ) -> torch.Tensor:
                                raw_rows: list[torch.Tensor] = []
                                row_scales: list[float] = []
                                for coefficient_row in coefficient_rows:
                                    if bool((coefficient_row == 0.0).all()):
                                        # Carry signed zero as one minimum
                                        # subnormal until after the frequency
                                        # vector is assembled.  The final 1/2
                                        # then rounds it back to the oracle zero
                                        # without an intervening vector VJP
                                        # canonicalising its sign.
                                        coefficient_row = torch.copysign(
                                            torch.full_like(
                                                coefficient_row,
                                                math.ulp(0.0),
                                            ),
                                            coefficient_row,
                                        )
                                        row_scales.append(0.5)
                                    else:
                                        row_scales.append(1.0)
                                    raw_rows.append(
                                        torch.sum(coefficient_row * sigma_delta)
                                        * thickness_delta
                                    )
                                return torch.stack(raw_rows) * torch.tensor(
                                    row_scales,
                                    dtype=torch.float64,
                                    device=device,
                                )

                            mixed_real_carrier = mixed_channel_carrier(
                                mixed_real[thickness_index]
                            )
                            mixed_imag_carrier = mixed_channel_carrier(
                                mixed_imag[thickness_index]
                            )
                            mixed_carrier = torch.complex(
                                mixed_real_carrier,
                                mixed_imag_carrier,
                            )
                            secondary_field = secondary_field + mixed_carrier
                    else:
                        dimensionless_secondary = axial_hankel_j0(
                            dimensionless_integrand,
                            dtype=torch.float64,
                            device=device,
                            relative_tolerance=_DIRECT_RELATIVE_TOLERANCE,
                            max_refinements=8,
                        )
                    transform_methods.add("axial_log_trapezoid")
                elif air_path_length == 0.0:
                    radius_ratio_t = torch.tensor(1.0, dtype=torch.float64, device=device)
                    dimensionless_secondary = dlf_hankel(
                        dimensionless_integrand,
                        radius_ratio_t,
                        kind="j0",
                    )
                    used_filter_assets = True
                    transform_methods.add("key_201_2012_dlf")
                else:
                    air_ratio = r / air_path_length
                    comparison_ratio = min(
                        air_ratio,
                        _DIRECT_COMPARISON_MAX_AIR_RATIO,
                    )
                    comparison_ratio_t = torch.tensor(
                        comparison_ratio,
                        dtype=torch.float64,
                        device=device,
                    )
                    direct_secondary, direct_relative_error = _direct_hankel_j0_with_error(
                        dimensionless_integrand,
                        comparison_ratio_t,
                        dtype=torch.float64,
                        device=device,
                        relative_tolerance=_DIRECT_RELATIVE_TOLERANCE,
                        max_refinements=8,
                    )
                    if air_ratio > _DIRECT_COMPARISON_MAX_AIR_RATIO:
                        anchor_dlf = dlf_hankel(
                            dimensionless_integrand,
                            comparison_ratio_t,
                            kind="j0",
                        )
                        anchor_scale = torch.maximum(
                            torch.abs(direct_secondary),
                            torch.abs(anchor_dlf),
                        ).clamp_min(torch.finfo(torch.float64).tiny)
                        anchor_error = (
                            torch.abs(anchor_dlf - direct_secondary) / anchor_scale
                            + direct_relative_error
                        )
                        if bool((anchor_error > _DLF_DIRECT_AGREEMENT_TOLERANCE).any()):
                            raise EMNumericsError(
                                "CSEM1D DLF failed its dimensionless anchor error budget",
                                details={
                                    "operator": "CSEM1D",
                                    "stage": "dlf_anchor_validation",
                                    "air_ratio": str(air_ratio),
                                    "anchor_ratio": str(_DIRECT_COMPARISON_MAX_AIR_RATIO),
                                    "tolerance": _DLF_DIRECT_AGREEMENT_TOLERANCE,
                                    "maximum_estimated_error": float(
                                        torch.max(anchor_error).detach().cpu()
                                    ),
                                },
                                object_name="CSEM1D",
                                field="bz",
                            )

                        length_scale = r
                        air_path_ratio = -air_path_length / length_scale
                        induction_log_shift = _dimensionless_induction_log_shift(
                            omega,
                            sigma,
                            length_scale_m=length_scale,
                        )

                        def far_integrand(q: torch.Tensor) -> torch.Tensor:
                            return _dimensionless_integrand_bz(
                                q,
                                omega,
                                sigma,
                                thickness,
                                length_scale_m=length_scale,
                                air_path_ratio=air_path_ratio,
                                log_induction_shift=induction_log_shift,
                            )

                        radius_ratio_t = torch.tensor(
                            1.0,
                            dtype=torch.float64,
                            device=device,
                        )
                        dimensionless_secondary = dlf_hankel(
                            far_integrand,
                            radius_ratio_t,
                            kind="j0",
                        )
                        used_filter_assets = True
                        transform_methods.add("key_201_2012_dlf")
                    elif axial_ratio <= 0.5:
                        dimensionless_secondary = direct_secondary
                        transform_methods.add("direct_j0_log_trapezoid")
                    else:
                        dlf_secondary = dlf_hankel(
                            dimensionless_integrand,
                            comparison_ratio_t,
                            kind="j0",
                        )
                        used_filter_assets = True
                        agreement_scale = torch.maximum(
                            torch.abs(direct_secondary),
                            torch.abs(dlf_secondary),
                        ).clamp_min(torch.finfo(torch.float64).tiny)
                        relative_disagreement = (
                            torch.abs(dlf_secondary - direct_secondary) / agreement_scale
                        )
                        estimated_relative_error = relative_disagreement + direct_relative_error
                        agreement_coordinate = torch.clamp(
                            torch.log(
                                _DLF_DIRECT_AGREEMENT_TOLERANCE
                                / estimated_relative_error.clamp_min(
                                    torch.finfo(torch.float64).tiny
                                )
                            )
                            / math.log(4.0),
                            min=0.0,
                            max=1.0,
                        )
                        agreement_weight = (
                            agreement_coordinate
                            * agreement_coordinate
                            * (3.0 - 2.0 * agreement_coordinate)
                        )
                        if axial_ratio >= 2.0:
                            geometry_weight = 1.0
                        else:
                            geometry_coordinate = math.log(axial_ratio / 0.5) / math.log(4.0)
                            geometry_weight = (
                                geometry_coordinate
                                * geometry_coordinate
                                * (3.0 - 2.0 * geometry_coordinate)
                            )
                        dlf_weight = agreement_weight * geometry_weight
                        dimensionless_secondary = (1.0 - dlf_weight) * direct_secondary + (
                            dlf_weight * dlf_secondary
                        )
                        if bool((dlf_weight == 0.0).all()):
                            transform_methods.add("direct_j0_log_trapezoid")
                        elif bool((dlf_weight == 1.0).all()):
                            transform_methods.add("key_201_2012_dlf")
                        else:
                            transform_methods.add("direct_dlf_blend")
                if secondary_field is None:
                    assert dimensionless_secondary is not None
                    if not bool(torch.isfinite(dimensionless_secondary).all()):
                        raise EMNumericsError(
                            "CSEM1D Hankel transform produced a non-finite field",
                            details={
                                "operator": "CSEM1D",
                                "stage": "hankel_transform",
                                "offset_m": str(r),
                            },
                            object_name="CSEM1D",
                            field="bz",
                        )
                    secondary_field = _scale_dimensionless_secondary(
                        dimensionless_secondary,
                        length_scale_m=length_scale,
                        magnetic_moment_am2=src.magnetic_moment_am2,
                        log_amplitude_adjustment=-induction_log_shift.reshape(-1),
                    )
                elif not bool(torch.isfinite(secondary_field).all()):
                    raise EMNumericsError(
                        "CSEM1D Hankel transform produced a non-finite field",
                        details={
                            "operator": "CSEM1D",
                            "stage": "hankel_transform",
                            "offset_m": str(r),
                        },
                        object_name="CSEM1D",
                        field="bz",
                    )
                field = field + secondary_field

                # --- Primary (free-space VMD): closed form ---------------
                # Frequency-independent in the quasi-static air regime; we
                # broadcast a single scalar across the frequency sweep.
                bz_primary = _vmd_primary_bz(
                    source_pos=(float(sx), float(sy), float(sz)),
                    receiver_pos=(float(rx_x), float(rx_y), float(rx_z)),
                    magnetic_moment_am2=src.magnetic_moment_am2,
                )
                field = field + bz_primary

            per_rcv.append(field)

        # Stack to (n_rcv, n_freq) complex128: one native complex channel.
        data = torch.stack(per_rcv, dim=0)
        if not bool(torch.isfinite(data).all()):
            raise EMNumericsError(
                "CSEM1D forward produced a non-finite field",
                details={"operator": "CSEM1D", "stage": "forward_output"},
                object_name="CSEM1D",
                field="bz",
            )
        ordered_methods = sorted(transform_methods)
        hankel_transform = ordered_methods[0] if len(ordered_methods) == 1 else "mixed"
        return ForwardOutput(
            data={
                "bz": data.contiguous(),
            },
            metadata={
                "n_frequencies": n_freq,
                "n_receivers": n_rcv,
                "n_sources": len(self._sources),
                "component": "bz",
                "hankel_transform": hankel_transform,
                "filter_names": ["key_201_2012"] if used_filter_assets else [],
                "surface_z_m": self.survey.surface_z_m,
            },
        )
