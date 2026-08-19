"""Shared immutable public facade over the internal packed Wave engine.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import copy
import math
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from typing import ClassVar, Literal, cast

import torch

from geobrain.core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    ForwardOperator,
    ForwardOutput,
    ModelState,
)
from geobrain.mesh.capabilities import UniformMesh

from ._engine.backends import EagerWaveBackend, NativeWaveBackend
from ._engine.contracts import CompiledAcquisition, PropagationRequest, WaveEquationProtocol
from ._engine.memory import create_memory_strategy
from ._engine.resources import (
    allocate_with_budget,
    autograd_resource_estimate_supported,
    runtime_calibration_remediation,
)
from ._engine.results import assemble_forward_output
from .acquisition import Seismic2DSurvey, Seismic3DSurvey
from .capabilities import WaveCapabilityReport, WaveUnsupportedCombination
from .config import WaveSimulationConfig
from .errors import WaveCapabilityError, WaveContractError, WaveNumericsError


_MODEL_UNITS = {
    "vp": "m/s",
    "vs": "m/s",
    "rho": "kg/m^3",
    "epsilon": "1",
    "delta": "1",
    "gamma": "1",
    "theta": "rad",
    "Q": "1",
    "Qp": "1",
    "Qs": "1",
}
_COMPONENT_UNITS = {
    "pressure": "Pa",
    "vx": "m/s",
    "vy": "m/s",
    "vz": "m/s",
    "sxx": "Pa",
    "syy": "Pa",
    "szz": "Pa",
    "sxy": "Pa",
    "sxz": "Pa",
    "syz": "Pa",
    "r": "Pa/s",
    "r_xx": "Pa/s",
    "r_zz": "Pa/s",
    "r_xz": "Pa/s",
}


def _contract_error(
    object_name: str,
    field: str,
    expected: object,
    actual: object,
    *,
    hint: str | None = None,
) -> WaveContractError:
    """Build a public facade contract error with actionable attribution."""
    return WaveContractError(
        "invalid Wave facade input",
        object_name=object_name,
        field=field,
        expected=expected,
        actual=actual,
        hint=hint,
    )


def _numerics_error(
    object_name: str,
    field: str,
    expected: object,
    actual: object,
    *,
    hint: str,
) -> WaveNumericsError:
    """Build a pre-execution constitutive or quality error."""
    return WaveNumericsError(
        "invalid Wave numerical input",
        object_name=object_name,
        field=field,
        expected=expected,
        actual=actual,
        hint=hint,
    )


def _minimum(tensor: torch.Tensor) -> float:
    """Read one validation scalar without altering the live tensor."""
    with torch.no_grad():
        return float(tensor.min())


def _upper_half_power_frequency(wavelets: torch.Tensor, dt: float) -> float | None:
    """Return the strictest valid per-source upper half-power frequency."""
    nt = int(wavelets.shape[-1])
    if nt < 8:
        return None
    with torch.no_grad():
        centred = wavelets - wavelets.mean(dim=-1, keepdim=True)
        amplitude = torch.fft.rfft(centred, dim=-1).abs()
        if amplitude.shape[-1] <= 3:
            return None
        amplitude[:, 0] = 0
        peaks, peak_indices = amplitude.max(dim=-1)
        valid = (peaks > 0) & (peaks >= 2.0 * amplitude.mean(dim=-1)) & (peak_indices >= 3)
        if not bool(valid.any()):
            return None
        bins = torch.arange(amplitude.shape[-1], device=amplitude.device)
        in_half_power_band = amplitude >= peaks[:, None] / math.sqrt(2.0)
        upper_indices = (
            torch.where(in_half_power_band & valid[:, None], bins[None, :], 0).max(dim=-1).values
        )
        strictest_index = int(upper_indices[valid].max())
    return strictest_index / (nt * dt)


def _tensor_shape_schema(
    *,
    rank: int,
    axes: list[str],
    dtypes: list[str],
    shape: list[str | int],
    unit: str | None = None,
    cpu_only: bool = False,
) -> dict[str, object]:
    """Describe a JSON shape specification for one runtime tensor."""
    schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["shape", "dtype", "device"],
        "properties": {
            "shape": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "minItems": rank,
                "maxItems": rank,
            },
            "dtype": {"enum": dtypes},
            "device": (
                {"const": "cpu"}
                if cpu_only
                else {"type": "string", "pattern": r"^(cpu|cuda(?::[0-9]+)?)$"}
            ),
        },
        "x-geobrain-runtime-type": "tensor",
        "axes": axes,
        "x-geobrain-shape": shape,
    }
    if unit is not None:
        schema["unit"] = unit
    return schema


def _configuration_defs(
    components: tuple[str, ...],
    *,
    backends: tuple[str, ...],
) -> dict[str, object]:
    """Return closed schema definitions for the accepted production surface."""
    return {
        "WaveDiscretizationConfig": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "fd_order": {"type": "integer", "minimum": 2, "multipleOf": 2},
                "strict_cfl": {"type": "boolean"},
                "min_points_per_wavelength": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                },
                "quality_policy": {"enum": ["error", "degraded"]},
            },
        },
        "WaveBoundaryConfig": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"enum": ["cpml", "none"]},
                "layers": {"type": "integer", "minimum": 0},
                "free_surface": {"type": "boolean"},
                "target_reflection": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                },
                "profile_order": {"type": "number", "exclusiveMinimum": 0},
                "kappa_max": {"type": "number", "minimum": 1},
                "alpha_max": {"type": "number", "minimum": 0},
            },
            "allOf": [
                {
                    "if": {
                        "required": ["kind"],
                        "properties": {"kind": {"const": "none"}},
                    },
                    "else": {"properties": {"layers": {"minimum": 1}}},
                }
            ],
        },
        "WaveMemoryConfig": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "strategy": {"enum": ["full", "checkpoint", "recursive", "boundary"]},
                "checkpoint_segments": {"type": "integer", "minimum": 1},
                "recursive_leaf_steps": {"type": "integer", "minimum": 1},
                "budget_bytes": {"type": ["integer", "null"], "minimum": 1},
            },
        },
        "WaveBackendConfig": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"enum": list(backends)},
                "deterministic": {"type": "boolean"},
            },
        },
        "WaveOutputConfig": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "components": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "enum": list(components),
                    },
                    "minItems": 1,
                    "uniqueItems": True,
                },
                "snapshot_policy": {"enum": ["none", "final", "selected", "energy"]},
                "snapshot_indices": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "uniqueItems": True,
                },
                "retain_field_gradients": {"type": "boolean"},
                "illumination": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                },
            },
            "allOf": [
                {
                    "if": {
                        "required": ["snapshot_policy"],
                        "properties": {"snapshot_policy": {"const": "selected"}},
                    },
                    "then": {
                        "required": ["snapshot_indices"],
                        "properties": {"snapshot_indices": {"minItems": 1}},
                    },
                    "else": {"properties": {"snapshot_indices": {"maxItems": 0}}},
                }
            ],
        },
        "WaveSimulationConfig": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "discretization": {"$ref": "#/$defs/WaveDiscretizationConfig"},
                "boundary": {"$ref": "#/$defs/WaveBoundaryConfig"},
                "memory": {"$ref": "#/$defs/WaveMemoryConfig"},
                "backend": {"$ref": "#/$defs/WaveBackendConfig"},
                "output": {"$ref": "#/$defs/WaveOutputConfig"},
            },
        },
    }


class _TimeDomainFacade(ForwardOperator):  # type: ignore[misc]
    """One validation, compilation, execution, and assembly path for all facades."""

    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (UniformMesh,)
    _dimension: ClassVar[int]
    _physics: ClassVar[str]
    _model_fields: ClassVar[tuple[str, ...]]
    _survey_type: ClassVar[type[Seismic2DSurvey] | type[Seismic3DSurvey]]
    _supports_native: ClassVar[bool] = False
    _supports_free_surface: ClassVar[bool] = False
    _native_component_sets: ClassVar[tuple[tuple[str, ...], ...]] = ()
    _native_unsupported_reason: ClassVar[str] = (
        "no exact native handler is implemented for this equation"
    )
    _native_unsupported_remediation: ClassVar[str] = "select the eager backend"
    _supports_boundary: ClassVar[bool] = True
    _boundary_unsupported_reason: ClassVar[str] = "boundary-memory reconstruction is unavailable"
    _boundary_unsupported_remediation: ClassVar[str] = "select checkpoint or recursive memory"
    _maturity: ClassVar[Literal["production", "experimental"]] = "experimental"

    def __init__(self, survey, wavelets, *, config=None):  # type: ignore[no-untyped-def]
        super().__init__()
        object_name = type(self).__name__
        if not isinstance(survey, self._survey_type):
            raise _contract_error(
                object_name,
                "survey",
                self._survey_type.__name__,
                type(survey).__name__,
            )
        if not isinstance(wavelets, torch.Tensor):
            raise _contract_error(object_name, "wavelets", "torch.Tensor", type(wavelets).__name__)
        expected_shape = (survey.n_source, survey.nt)
        if tuple(wavelets.shape) != expected_shape:
            raise _contract_error(
                object_name,
                "wavelets",
                f"exact shape {expected_shape}; use shared_wavelet explicitly",
                tuple(wavelets.shape),
            )
        if config is None:
            config = WaveSimulationConfig()
        if not isinstance(config, WaveSimulationConfig):
            raise _contract_error(
                object_name,
                "config",
                "WaveSimulationConfig or None",
                type(config).__name__,
            )
        self.survey = survey
        self.config = config
        dynamic_custom_vjp = (config.memory.strategy == "boundary" and self._supports_boundary) or (
            config.backend.name == "native" and self._supports_native
        )
        if dynamic_custom_vjp:
            base_spec = cast(
                DifferentiabilitySpec,
                getattr(type(self), "differentiability"),
            )
            self.differentiability = replace(
                base_spec,
                level=DifferentiabilityLevel.CUSTOM_VJP,
            )
        self.register_buffer("wavelets", wavelets, persistent=False)

    @classmethod
    @abstractmethod
    def _new_equation(cls, config: WaveSimulationConfig) -> WaveEquationProtocol:
        raise NotImplementedError

    @classmethod
    def capabilities(cls) -> WaveCapabilityReport:
        """Return a deterministic immutable report for the local runtime."""
        equation = cls._new_equation(WaveSimulationConfig())
        declaration = equation.declaration
        # Capability reports and Agent/UI schemas describe only the accepted
        # production selection space. Native/CUDA remain explicit runtime
        # experiments until the separate device-matrix calibration campaign is accepted.
        backends = ("eager",)
        memory: tuple[str, ...] = ("full", "checkpoint", "recursive")
        if cls._supports_boundary:
            memory = (*memory, "boundary")
        cpu_resource_estimate_supported = autograd_resource_estimate_supported(device_type="cpu")
        direct_estimate_remediation = runtime_calibration_remediation(budget_enforcement=False)
        budget_remediation = runtime_calibration_remediation(budget_enforcement=True)
        unsupported: list[WaveUnsupportedCombination] = []
        if not cpu_resource_estimate_supported:
            unsupported.append(
                WaveUnsupportedCombination(
                    selection=(
                        ("resource.estimate", "autograd"),
                        ("model.device", "cpu"),
                    ),
                    reason=(
                        "direct autograd resource estimation on CPU has no verified "
                        "runtime calibration for the current OS, architecture, and "
                        "Torch version; device-matrix calibration evidence is pending"
                    ),
                    remediation=direct_estimate_remediation,
                )
            )
        unsupported.append(
            WaveUnsupportedCombination(
                selection=(
                    ("resource.estimate", "autograd"),
                    ("model.device", "cuda"),
                ),
                reason=(
                    "direct autograd resource estimation on CUDA has no verified "
                    "runtime calibration; device-matrix calibration evidence is pending"
                ),
                remediation=direct_estimate_remediation,
            )
        )
        if not cpu_resource_estimate_supported:
            unsupported.append(
                WaveUnsupportedCombination(
                    selection=(
                        ("memory.budget_bytes", "non-null"),
                        ("model.device", "cpu"),
                    ),
                    reason=(
                        "CPU budget enforcement has no verified runtime calibration "
                        "for the current OS, architecture, and Torch version; "
                        "device-matrix calibration evidence is pending"
                    ),
                    remediation=budget_remediation,
                )
            )
        unsupported.append(
            WaveUnsupportedCombination(
                selection=(
                    ("memory.budget_bytes", "non-null"),
                    ("model.device", "cuda"),
                ),
                reason=(
                    "CUDA budget enforcement has no verified runtime calibration; "
                    "device-matrix calibration evidence is pending"
                ),
                remediation=budget_remediation,
            )
        )
        unsupported.append(
            WaveUnsupportedCombination(
                selection=(("model.device", "cuda"),),
                reason=(
                    "CUDA execution has no accepted numerical device-matrix "
                    "evidence in GeoBrain 0.2.0"
                ),
                remediation="select CPU eager execution for production use",
            )
        )
        if not cls._supports_free_surface:
            unsupported.append(
                WaveUnsupportedCombination(
                    selection=(("boundary.free_surface", "true"),),
                    reason=("the selected equation has no validated free-surface boundary update"),
                    remediation="disable the free surface",
                )
            )
        if not cls._supports_native:
            unsupported.append(
                WaveUnsupportedCombination(
                    selection=(("backend.name", "native"),),
                    reason=cls._native_unsupported_reason,
                    remediation=cls._native_unsupported_remediation,
                )
            )
        else:
            unsupported.append(
                WaveUnsupportedCombination(
                    selection=(("backend.name", "native"),),
                    reason=(
                        "the native backend is an explicit experimental path until "
                        "CUDA numerical acceptance is complete"
                    ),
                    remediation=(
                        "select eager for production use; request native directly "
                        "only while collecting device-matrix evidence"
                    ),
                )
            )
            unsupported.extend(
                (
                    WaveUnsupportedCombination(
                        selection=(
                            ("backend.name", "native"),
                            ("model.device", "cpu"),
                        ),
                        reason="the native backend requires CUDA-resident live tensors",
                        remediation="select the eager backend or move the model and wavelets to CUDA",
                    ),
                    WaveUnsupportedCombination(
                        selection=(
                            ("backend.name", "native"),
                            ("discretization.fd_order", ">16"),
                        ),
                        reason="native finite-difference kernels support even orders no greater than 16",
                        remediation="select an even fd_order from 2 through 16 or use eager",
                    ),
                    WaveUnsupportedCombination(
                        selection=(
                            ("backend.name", "native"),
                            ("output.snapshot_policy", "selected|energy"),
                        ),
                        reason="native output retention supports only none or final",
                        remediation="select none/final snapshots or use eager",
                    ),
                    WaveUnsupportedCombination(
                        selection=(
                            ("backend.name", "native"),
                            ("output.illumination", "non-empty"),
                        ),
                        reason="native kernels do not accumulate illumination fields",
                        remediation="request no illumination or use eager",
                    ),
                    WaveUnsupportedCombination(
                        selection=(
                            ("backend.name", "native"),
                            ("output.retain_field_gradients", "true"),
                        ),
                        reason="native final wavefields are diagnostic and non-differentiable",
                        remediation="disable retained field gradients or use eager",
                    ),
                    WaveUnsupportedCombination(
                        selection=(
                            ("backend.name", "native"),
                            (
                                "survey.source_layout",
                                "not one-source-per-shot/shared-receivers",
                            ),
                        ),
                        reason="native kernels require one source per shot and identical receiver geometry",
                        remediation="regularize the packed survey layout or use eager",
                    ),
                )
            )
            component_selection = (
                "other-than-pressure"
                if cls._dimension == 3
                else "other-than-pressure-or-pressure+vx+vz"
            )
            unsupported.append(
                WaveUnsupportedCombination(
                    selection=(
                        ("backend.name", "native"),
                        ("output.components", component_selection),
                    ),
                    reason="the selected receiver component tuple has no complete native implementation",
                    remediation="select the reported native component tuple or use eager",
                )
            )
            if cls._physics == "acoustic" and cls._dimension == 2:
                unsupported.append(
                    WaveUnsupportedCombination(
                        selection=(
                            ("backend.name", "native"),
                            ("boundary.free_surface", "true"),
                        ),
                        reason="acoustic 2-D native CUDA does not implement a free surface",
                        remediation="disable the free surface or use eager",
                    )
                )
            for strategy in ("checkpoint", "recursive", "boundary"):
                unsupported.append(
                    WaveUnsupportedCombination(
                        selection=(
                            ("backend.name", "native"),
                            ("memory.strategy", strategy),
                        ),
                        reason="the native backend owns its internal adjoint storage policy",
                        remediation="select full memory or the eager backend",
                    )
                )
        if cls._supports_boundary:
            unsupported.extend(
                (
                    WaveUnsupportedCombination(
                        selection=(
                            ("memory.strategy", "boundary"),
                            ("boundary.kind", "none"),
                        ),
                        reason="boundary-saving reconstruction requires a CPML exterior band",
                        remediation="select a CPML boundary or another memory strategy",
                    ),
                    WaveUnsupportedCombination(
                        selection=(
                            ("memory.strategy", "boundary"),
                            ("boundary.free_surface", "true"),
                        ),
                        reason="boundary-saving reconstruction does not restore a free surface",
                        remediation="disable the free surface or select another memory strategy",
                    ),
                    WaveUnsupportedCombination(
                        selection=(
                            ("memory.strategy", "boundary"),
                            ("boundary.layers", "< equation halo width"),
                        ),
                        reason="the saved exterior band must cover the finite-difference halo",
                        remediation="increase CPML layers to at least the equation halo width",
                    ),
                    WaveUnsupportedCombination(
                        selection=(
                            ("memory.strategy", "boundary"),
                            ("output.components", "other-than-pressure"),
                        ),
                        reason="boundary-saving receiver reconstruction is validated only for pressure",
                        remediation="request pressure only or select another memory strategy",
                    ),
                    WaveUnsupportedCombination(
                        selection=(
                            ("memory.strategy", "boundary"),
                            ("output.snapshot_policy", "selected|energy"),
                        ),
                        reason="boundary-saving output retention supports only none or final",
                        remediation="select none/final snapshots or another memory strategy",
                    ),
                    WaveUnsupportedCombination(
                        selection=(
                            ("memory.strategy", "boundary"),
                            ("output.illumination", "non-empty"),
                        ),
                        reason="boundary-saving reconstruction does not retain illumination accumulators",
                        remediation="request no illumination or select another memory strategy",
                    ),
                )
            )
        else:
            unsupported.append(
                WaveUnsupportedCombination(
                    selection=(("memory.strategy", "boundary"),),
                    reason=cls._boundary_unsupported_reason,
                    remediation=cls._boundary_unsupported_remediation,
                )
            )
        return WaveCapabilityReport(
            physics=cls._physics,
            equation=declaration.identifier,
            dimension=cls._dimension,
            maturity=cls._maturity,
            required_model_fields=tuple((name, _MODEL_UNITS[name]) for name in cls._model_fields),
            components=declaration.declared_components,
            dtypes=("float32", "float64"),
            devices=("cpu",),
            backends=backends,
            boundaries=("cpml", "none"),
            memory_strategies=memory,
            differentiable_model_fields=cls._model_fields,
            differentiable_wavelets=True,
            mesh_capabilities=("UniformMesh",),
            resource_estimate_supported=cpu_resource_estimate_supported,
            unsupported=tuple(unsupported),
        )

    @classmethod
    def input_schema(cls) -> Mapping[str, object]:
        """Return a fresh JSON-ready schema for agents and future UI clients."""
        report = cls.capabilities()
        dimension = cls._dimension
        coordinate_names = ["x", "z"] if dimension == 2 else ["x", "y", "z"]
        model_axes = ["z", "x"] if dimension == 2 else ["z", "x", "y"]
        schema: dict[str, object] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "schema_version": "geobrain.wave.input/1.0",
            "title": cls.__name__,
            "type": "object",
            "additionalProperties": False,
            "required": ["survey", "wavelets", "model"],
            "properties": {
                "survey": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "source_positions",
                        "source_shot_index",
                        "receiver_positions",
                        "receiver_shot_index",
                        "nt",
                        "dt",
                        "t0",
                    ],
                    "properties": {
                        "source_positions": _tensor_shape_schema(
                            rank=2,
                            axes=["source", "coordinate"],
                            dtypes=["float64"],
                            shape=["n_source", dimension],
                            unit="m",
                            cpu_only=True,
                        ),
                        "source_shot_index": _tensor_shape_schema(
                            rank=1,
                            axes=["source"],
                            dtypes=["int64"],
                            shape=["n_source"],
                            cpu_only=True,
                        ),
                        "receiver_positions": _tensor_shape_schema(
                            rank=2,
                            axes=["trace", "coordinate"],
                            dtypes=["float64"],
                            shape=["n_trace", dimension],
                            unit="m",
                            cpu_only=True,
                        ),
                        "receiver_shot_index": _tensor_shape_schema(
                            rank=1,
                            axes=["trace"],
                            dtypes=["int64"],
                            shape=["n_trace"],
                            cpu_only=True,
                        ),
                        "nt": {"type": "integer", "minimum": 1},
                        "dt": {"type": "number", "exclusiveMinimum": 0},
                        "t0": {"type": "number"},
                    },
                    "packed": True,
                    "coordinate_order": coordinate_names,
                    "axes": {
                        "source_positions": ["source", "coordinate"],
                        "source_shot_index": ["source"],
                        "receiver_positions": ["trace", "coordinate"],
                        "receiver_shot_index": ["trace"],
                    },
                    "units": {"positions": "m", "dt": "s", "t0": "s"},
                },
                "wavelets": _tensor_shape_schema(
                    rank=2,
                    axes=["source", "time"],
                    dtypes=["float32", "float64"],
                    shape=["n_source", "nt"],
                    cpu_only=True,
                ),
                "model": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(cls._model_fields),
                    "properties": {
                        name: _tensor_shape_schema(
                            rank=dimension,
                            axes=model_axes,
                            dtypes=["float32", "float64"],
                            shape=["n_space"] * dimension,
                            unit=_MODEL_UNITS[name],
                            cpu_only=True,
                        )
                        for name in cls._model_fields
                    },
                    "fields": [
                        {
                            "name": name,
                            "unit": _MODEL_UNITS[name],
                            "axes": model_axes,
                        }
                        for name in cls._model_fields
                    ],
                },
                "config": {"$ref": "#/$defs/WaveSimulationConfig"},
            },
            "output": {
                "seismic": {
                    "axes": ["trace", "time", "component"],
                    "components": list(report.components),
                    "units": {name: _COMPONENT_UNITS.get(name, "Pa") for name in report.components},
                }
            },
            "$defs": _configuration_defs(report.components, backends=report.backends),
        }
        return copy.deepcopy(schema)

    def _validate_backend_device(self, reference: torch.Tensor) -> None:
        """Reject device/backend selections before backend construction or use."""
        if self.config.backend.name == "native" and reference.device.type != "cuda":
            raise WaveCapabilityError(
                "requested Wave backend/device combination is unavailable",
                object_name=type(self).__name__,
                field="model.device",
                expected="CUDA model and wavelets with native backend",
                actual=reference.device.type,
                hint="select the eager backend or move the model and wavelets to CUDA",
            )

    def _validate_capabilities(self, equation: WaveEquationProtocol) -> None:
        config = self.config
        if config.boundary.free_surface and not self._supports_free_surface:
            raise WaveCapabilityError(
                "selected Wave equation does not implement a free surface",
                object_name=type(self).__name__,
                field="boundary.free_surface",
                expected=False,
                actual=True,
                hint="disable the free surface",
            )
        if config.backend.name == "native" and not self._supports_native:
            raise WaveCapabilityError(
                self._native_unsupported_reason,
                object_name=type(self).__name__,
                field="backend.name",
                expected="eager",
                actual="native",
                hint=self._native_unsupported_remediation,
            )
        if config.memory.strategy == "boundary" and not self._supports_boundary:
            raise WaveCapabilityError(
                self._boundary_unsupported_reason,
                object_name=type(self).__name__,
                field="memory.strategy",
                expected="full, checkpoint, or recursive",
                actual="boundary",
                hint=self._boundary_unsupported_remediation,
            )
        declared = equation.declaration.declared_components
        unsupported_components = tuple(
            name for name in config.output.components if name not in declared
        )
        if unsupported_components:
            raise WaveCapabilityError(
                "requested Wave receiver component is unavailable",
                object_name=type(self).__name__,
                field="output.components",
                expected=declared,
                actual=unsupported_components,
                hint="select only reported capability components",
            )
        if config.backend.name != "native":
            return
        if config.memory.strategy != "full":
            raise WaveCapabilityError(
                "requested Wave backend/memory combination is unavailable",
                object_name=type(self).__name__,
                field="memory.strategy",
                expected="full with native backend",
                actual=config.memory.strategy,
                hint="select full memory or the eager backend",
            )
        if config.discretization.fd_order > 16:
            raise WaveCapabilityError(
                "native finite-difference order is unavailable",
                object_name=type(self).__name__,
                field="discretization.fd_order",
                expected="even order <= 16 with native backend",
                actual=config.discretization.fd_order,
                hint="select an even fd_order from 2 through 16 or use eager",
            )
        if config.output.components not in self._native_component_sets:
            raise WaveCapabilityError(
                "receiver component tuple has no complete native implementation",
                object_name=type(self).__name__,
                field="output.components",
                expected=self._native_component_sets,
                actual=config.output.components,
                hint="select a reported native component tuple or use eager",
            )
        if config.output.snapshot_policy not in ("none", "final"):
            raise WaveCapabilityError(
                "native output retention supports only none or final",
                object_name=type(self).__name__,
                field="output.snapshot_policy",
                expected=("none", "final"),
                actual=config.output.snapshot_policy,
                hint="select none/final snapshots or use eager",
            )
        if config.output.illumination:
            raise WaveCapabilityError(
                "native kernels do not accumulate illumination fields",
                object_name=type(self).__name__,
                field="output.illumination",
                expected=(),
                actual=config.output.illumination,
                hint="request no illumination or use eager",
            )
        if config.output.retain_field_gradients:
            raise WaveCapabilityError(
                "native final wavefields are diagnostic and non-differentiable",
                object_name=type(self).__name__,
                field="output.retain_field_gradients",
                expected=False,
                actual=True,
                hint="disable retained field gradients or use eager",
            )
        if self._physics == "acoustic" and self._dimension == 2 and config.boundary.free_surface:
            raise WaveCapabilityError(
                "acoustic 2-D native CUDA does not implement a free surface",
                object_name=type(self).__name__,
                field="boundary.free_surface",
                expected=False,
                actual=True,
                hint="disable the free surface or use eager",
            )

    def _validate_native_source_layout(self, acquisition: CompiledAcquisition) -> None:
        """Reject native layouts that cannot map to its dense shot geometry."""
        if self.config.backend.name != "native":
            return
        source_ids = acquisition.source_shot_index
        receiver_ids = acquisition.receiver_shot_index
        supported = (
            tuple(source_ids.shape) == (acquisition.n_shot,)
            and torch.equal(
                source_ids,
                torch.arange(
                    acquisition.n_shot,
                    dtype=torch.int64,
                    device=source_ids.device,
                ),
            )
            and acquisition.n_trace % acquisition.n_shot == 0
        )
        receivers_per_shot = acquisition.n_trace // acquisition.n_shot
        if supported:
            expected_receiver_ids = torch.arange(
                acquisition.n_shot,
                dtype=torch.int64,
                device=receiver_ids.device,
            ).repeat_interleave(receivers_per_shot)
            supported = torch.equal(receiver_ids, expected_receiver_ids)
        if supported:
            receivers = acquisition.receiver_indices.reshape(
                acquisition.n_shot,
                receivers_per_shot,
                acquisition.receiver_indices.shape[1],
            )
            supported = all(
                torch.equal(receivers[shot], receivers[0]) for shot in range(1, acquisition.n_shot)
            )
        if not supported:
            raise WaveCapabilityError(
                "native CUDA requires one source per shot and shared receivers",
                object_name=type(self).__name__,
                field="survey.source_layout",
                expected="one source per shot and shared receiver geometry",
                actual="packed irregular acquisition",
                hint="regularize the packed survey layout or use eager",
            )

    def _validate_live_model(
        self, state: ModelState, mesh_shape: tuple[int, ...]
    ) -> dict[str, torch.Tensor]:
        model = dict(zip(self._model_fields, state.fetch(*self._model_fields)))
        reference = model[self._model_fields[0]]
        expected = f"shape={mesh_shape}, dtype/device exactly matching {self._model_fields[0]}"
        if reference.dtype not in (torch.float32, torch.float64):
            raise _contract_error(
                type(self).__name__,
                self._model_fields[0],
                "torch.float32 or torch.float64",
                reference.dtype,
            )
        for name, tensor in model.items():
            if (
                tuple(tensor.shape) != mesh_shape
                or tensor.dtype is not reference.dtype
                or tensor.device != reference.device
            ):
                raise _contract_error(
                    type(self).__name__,
                    name,
                    expected,
                    f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}",
                )
            with torch.no_grad():
                finite = bool(torch.isfinite(tensor).all())
            if not finite:
                raise WaveNumericsError(
                    "Wave model contains non-finite values",
                    object_name=type(self).__name__,
                    field=name,
                    expected="finite live tensor",
                    actual="non-finite values",
                    hint="replace NaN or infinite model values before execution",
                )
        if self.wavelets.dtype is not reference.dtype or self.wavelets.device != reference.device:
            raise _contract_error(
                type(self).__name__,
                "wavelets",
                f"dtype={reference.dtype}, device={reference.device}",
                f"dtype={self.wavelets.dtype}, device={self.wavelets.device}",
            )
        with torch.no_grad():
            finite_wavelets = bool(torch.isfinite(self.wavelets).all())
        if not finite_wavelets:
            raise WaveNumericsError(
                "Wave source contains non-finite values",
                object_name=type(self).__name__,
                field="wavelets",
                expected="finite live tensor",
                actual="non-finite values",
                hint="replace NaN or infinite source samples before execution",
            )
        self._validate_constitutive(model)
        return model

    def _validate_constitutive(self, model: Mapping[str, torch.Tensor]) -> None:
        if _minimum(model["vp"]) <= 0.0:
            raise _numerics_error(
                type(self).__name__,
                "vp",
                "> 0 m/s",
                _minimum(model["vp"]),
                hint="provide positive P-wave velocity",
            )
        if _minimum(model["rho"]) <= 0.0:
            raise _numerics_error(
                type(self).__name__,
                "rho",
                "> 0 kg/m^3",
                _minimum(model["rho"]),
                hint="provide positive density",
            )
        if "vs" in model and _minimum(model["vs"]) < 0.0:
            raise _numerics_error(
                type(self).__name__,
                "vs",
                ">= 0 m/s",
                _minimum(model["vs"]),
                hint="provide non-negative S-wave velocity",
            )
        for name in ("Q", "Qp", "Qs"):
            if name in model and _minimum(model[name]) <= 1.0:
                raise _numerics_error(
                    type(self).__name__,
                    name,
                    "> 1",
                    _minimum(model[name]),
                    hint="provide a finite physical quality factor above one",
                )
        if "epsilon" in model:
            vp, vs, rho = model["vp"], model["vs"], model["rho"]
            c33 = rho * vp.square()
            c44 = rho * vs.square()
            difference = c33 - c44
            radicand = difference.square() + 2.0 * model["delta"] * c33 * difference
            with torch.no_grad():
                minimum_radicand = float(radicand.min())
                minimum_horizontal = float((1.0 + 2.0 * model["epsilon"]).min())
            if minimum_radicand < 0.0:
                raise _numerics_error(
                    type(self).__name__,
                    "delta",
                    "non-negative Thomsen stiffness radicand",
                    minimum_radicand,
                    hint="adjust delta so the anisotropic stiffness is real",
                )
            if minimum_horizontal <= 0.0:
                raise _numerics_error(
                    type(self).__name__,
                    "epsilon",
                    "1 + 2*epsilon > 0",
                    minimum_horizontal,
                    hint="provide an admissible horizontal P-wave stiffness",
                )
            if "gamma" in model:
                with torch.no_grad():
                    minimum_shear = float((1.0 + 2.0 * model["gamma"]).min())
                if minimum_shear <= 0.0:
                    raise _numerics_error(
                        type(self).__name__,
                        "gamma",
                        "1 + 2*gamma > 0",
                        minimum_shear,
                        hint="provide an admissible horizontal shear stiffness",
                    )

    def _quality_assessment(
        self,
        model: Mapping[str, torch.Tensor],
        spacing: tuple[float, ...],
    ) -> tuple[str, float | None]:
        """Return the PPW policy status and its measured value when assessable."""
        frequency = _upper_half_power_frequency(self.wavelets, self.survey.dt)
        if frequency is None:
            return "not_assessed", None
        minimum_velocity = _minimum(model["vp"])
        if "vs" in model:
            positive_vs = model["vs"][model["vs"] > 0]
            if positive_vs.numel():
                minimum_velocity = min(minimum_velocity, _minimum(positive_vs))
        points = minimum_velocity / (frequency * max(spacing))
        required = self.config.discretization.min_points_per_wavelength
        if points >= required:
            return "passed", points
        if self.config.discretization.quality_policy == "error":
            raise WaveNumericsError(
                "Wave discretization does not meet its quality policy",
                object_name=type(self).__name__,
                field="min_points_per_wavelength",
                expected=f">= {required}",
                actual=points,
                hint="reduce spacing or source frequency, or select degraded policy",
            )
        return "degraded", points

    def _cfl_ratio(
        self,
        equation: WaveEquationProtocol,
        model: Mapping[str, torch.Tensor],
        spacing: tuple[float, ...],
    ) -> float:
        """Evaluate the compiled-equation CFL ratio before backend allocation."""
        try:
            dt_max = float(equation.cfl_limit(model, spacing))
        except (
            KeyError,
            IndexError,
            TypeError,
            RuntimeError,
            ValueError,
            OverflowError,
        ) as exc:
            raise WaveContractError(
                "Wave equation CFL evaluation failed",
                object_name=type(equation).__name__,
                field="cfl",
                expected="positive finite stability limit",
                actual="evaluation failed",
            ) from exc
        if not math.isfinite(dt_max) or dt_max <= 0.0:
            raise WaveContractError(
                "Wave equation CFL limit is invalid",
                object_name=type(equation).__name__,
                field="cfl",
                expected="positive finite stability limit",
                actual=dt_max,
            )
        ratio = float(self.survey.dt) / dt_max
        if ratio > 1.0 and self.config.discretization.strict_cfl:
            raise WaveNumericsError(
                f"CFL violated: dt={self.survey.dt:.4e} > dt_max={dt_max:.4e}",
                object_name=type(equation).__name__,
                field="dt",
                expected=f"<= {dt_max}",
                actual=self.survey.dt,
            )
        return ratio

    def _compile_acquisition(
        self,
        mesh: object,
        *,
        device: torch.device,
    ) -> CompiledAcquisition:
        mesh_shape = tuple(int(value) for value in getattr(mesh, "shape"))
        spacing = tuple(float(value) for value in getattr(mesh, "spacing"))
        origin = tuple(float(value) for value in getattr(mesh, "origin"))
        if len(mesh_shape) != self._dimension:
            raise _contract_error(
                type(self).__name__,
                "mesh",
                f"{self._dimension}-D uniform mesh",
                mesh_shape,
            )
        # TensorMesh owns platform storage as (z, x[, y]); public acquisition is
        # (x, z) / (x, y, z). The engine consumes storage-order indices exactly.
        public_to_mesh = (1, 0) if self._dimension == 2 else (2, 0, 1)

        def compile_positions(name: str, positions: torch.Tensor) -> torch.Tensor:
            reordered = positions[:, public_to_mesh]
            origin_tensor = torch.tensor(origin, dtype=torch.float64)
            spacing_tensor = torch.tensor(spacing, dtype=torch.float64)
            scaled = (reordered - origin_tensor) / spacing_tensor
            indices_cpu = torch.floor(scaled).to(torch.int64)
            valid = torch.ones(indices_cpu.shape[0], dtype=torch.bool)
            for axis, size in enumerate(mesh_shape):
                valid &= (indices_cpu[:, axis] >= 0) & (indices_cpu[:, axis] < size)
            if not bool(valid.all()):
                first = int(torch.nonzero(~valid, as_tuple=False)[0])
                raise _contract_error(
                    type(self).__name__,
                    name,
                    "finite coordinates inside the live mesh domain",
                    positions[first].tolist(),
                    hint="move the acquisition point into a mesh cell; coordinates are never clamped",
                )
            return indices_cpu.to(device=device)

        source_indices = compile_positions("survey.source_positions", self.survey.source_positions)
        receiver_indices = compile_positions(
            "survey.receiver_positions", self.survey.receiver_positions
        )
        return CompiledAcquisition(
            source_indices=source_indices,
            receiver_indices=receiver_indices,
            source_shot_index=self.survey.source_shot_index,
            receiver_shot_index=self.survey.receiver_shot_index,
            n_shot=self.survey.n_shot,
            nt=self.survey.nt,
            dt=self.survey.dt,
            t0=self.survey.t0,
            survey_fingerprint=self.survey.fingerprint,
            mesh_shape=mesh_shape,
            spacing=spacing,
        )

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        mesh = ctx.require_mesh()
        mesh_shape = tuple(int(value) for value in mesh.shape)
        equation = self._new_equation(self.config)
        self._validate_capabilities(equation)
        model = self._validate_live_model(state, mesh_shape)
        reference = model[self._model_fields[0]]
        acquisition = self._compile_acquisition(mesh, device=reference.device)
        self._validate_native_source_layout(acquisition)
        self._validate_backend_device(reference)
        quality_status, points_per_wavelength = self._quality_assessment(model, acquisition.spacing)
        cfl_ratio = self._cfl_ratio(equation, model, acquisition.spacing)
        if cfl_ratio > 1.0:
            quality_status = "degraded"
        backend = EagerWaveBackend() if self.config.backend.name == "eager" else NativeWaveBackend()
        memory = create_memory_strategy(self.config.memory.strategy)
        request = PropagationRequest(
            equation=equation,
            backend=backend,
            memory=memory,
            acquisition=acquisition,
            model=model,
            wavelets=self.wavelets,
            config=self.config,
            components=self.config.output.components,
            output_indices=self.config.output.snapshot_indices,
        )
        result = allocate_with_budget(
            request,
            lambda: backend.execute(request),
            autograd_enabled=torch.is_grad_enabled(),
        )
        differentiability = (
            DifferentiabilityLevel.CUSTOM_VJP.value
            if (self.config.memory.strategy == "boundary" or self.config.backend.name == "native")
            else DifferentiabilityLevel.FULL_AUTOGRAD.value
        )
        return assemble_forward_output(
            result,
            axis_names=("trace", "time", "component"),
            units={
                "time": "s",
                **{
                    name: _COMPONENT_UNITS.get(name, "Pa") for name in self.config.output.components
                },
            },
            component_order=self.config.output.components,
            survey_fingerprint=self.survey.fingerprint,
            backend=self.config.backend.name,
            strategy=self.config.memory.strategy,
            maturity=self._maturity,
            quality_status=quality_status,
            points_per_wavelength=points_per_wavelength,
            cfl_ratio=cfl_ratio,
            differentiability=differentiability,
            equation=equation.declaration.identifier,
            sampling={"nt": self.survey.nt, "dt": self.survey.dt, "t0": self.survey.t0},
        )


__all__ = ["_TimeDomainFacade"]
