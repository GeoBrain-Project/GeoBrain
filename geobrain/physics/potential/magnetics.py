"""Strict-SI bounded 3-D magnetic forward operators.

Both scalar induced magnetics and packed vector magnetization compile the core
``TensorMesh`` through the canonical elevation-positive-up geometry bridge and
delegate field, projection, gradient, transpose, and VJP work to the shared
Potential executor.  Earth-field inclination remains the geophysical convention
(positive downward); its Cartesian coefficients are expressed in Potential's
elevation-positive-up coordinates before execution.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math

from geobrain.core.constants import MU_0
from collections import OrderedDict
from typing import ClassVar, Literal, cast

import torch

from geobrain.core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ErrorCode,
    ForwardContext,
    ForwardOperator,
    ForwardOutput,
    ModelState,
)
from geobrain.mesh import require_capable_mesh
from geobrain.mesh.capabilities import PrismGeometryMesh
from geobrain.mesh.tensor import TensorMesh

from ._engine import build_sensitivity
from ._engine.execute import PotentialSensitivity
from ._engine.plan import PrismKernelPlan, tensor_mesh_cell_bounds_m
from .capabilities import (
    PotentialCapabilityReport,
    potential_capability_report,
    potential_input_schema,
)
from .resources import (
    PotentialResourceEstimate,
    estimate_prism_resources,
    infeasible_resource_estimate,
)
from .config import EarthField, PotentialExecutionConfig
from .errors import PotentialCapabilityError, PotentialContractError, PotentialResourceError
from .surveys import PotentialSurvey3D


MagneticComponent = Literal[
    "tmi",
    "bx",
    "by",
    "bz",
    "bxx",
    "bxy",
    "bxz",
    "byy",
    "byz",
    "bzz",
]
_MAGNETIC_COMPONENTS = frozenset(
    {"tmi", "bx", "by", "bz", "bxx", "bxy", "bxz", "byy", "byz", "bzz"}
)
_MAGNETIC_FIELD_COMPONENTS = frozenset({"tmi", "bx", "by", "bz"})
_VACUUM_PERMEABILITY_SI = MU_0  # single-sourced platform value (CODATA-2018)
_IMMUTABLE_CONFIGURATION_FIELDS = frozenset(
    {
        "survey",
        "components",
        "execution",
        "earth_field",
        "projection_field",
        "_configuration_frozen",
    }
)


def _execution_or_default(
    execution: PotentialExecutionConfig | None,
    *,
    owner: str,
) -> PotentialExecutionConfig:
    if execution is None:
        return PotentialExecutionConfig()
    if not isinstance(execution, PotentialExecutionConfig):
        raise PotentialContractError(
            "Magnetic execution must be PotentialExecutionConfig or None.",
            object_name=owner,
            field="execution",
            expected="PotentialExecutionConfig or None",
            actual=execution,
            hint="Provide an explicit validated Potential execution policy.",
        )
    return execution


def _validate_components(
    components: object,
    *,
    owner: str,
) -> tuple[MagneticComponent, ...]:
    if not isinstance(components, tuple) or not components:
        raise PotentialContractError(
            "Magnetic components must be a non-empty tuple.",
            object_name=owner,
            field="components",
            expected="non-empty ordered tuple of unique magnetic component names",
            actual=components,
            hint="Request one or more SI magnetic field or gradient components.",
        )
    if len(set(components)) != len(components):
        raise PotentialContractError(
            "Magnetic components must be unique.",
            object_name=owner,
            field="components",
            expected="unique ordered magnetic component names",
            actual=components,
            hint="Remove duplicate components while preserving order.",
        )
    unsupported = tuple(
        component
        for component in components
        if not isinstance(component, str) or component not in _MAGNETIC_COMPONENTS
    )
    if unsupported:
        raise PotentialCapabilityError(
            "Magnetic components contain unsupported names.",
            object_name=owner,
            field="components",
            expected=sorted(_MAGNETIC_COMPONENTS),
            actual=unsupported,
            hint="Use only supported TMI, Cartesian field, or magnetic-gradient names.",
        )
    return cast(tuple[MagneticComponent, ...], components)


def _require_prism_mesh(ctx: ForwardContext, *, owner: str):
    # Capability-declared requirement; no concrete-class guard. The prism
    # engine consumes only PrismGeometryMesh.cell_bounds(), so graded
    # octrees qualify alongside the core TensorMesh.
    mesh = require_capable_mesh(ctx, PrismGeometryMesh, owner=owner)
    if mesh.n_dim != 3:
        raise PotentialContractError(
            f"{owner} requires a 3-D TensorMesh.",
            object_name=owner,
            field="mesh",
            expected="3-D TensorMesh",
            actual={"type": type(mesh).__name__, "n_dim": getattr(mesh, "n_dim", None)},
            hint="Provide a 3-D core TensorMesh in ForwardContext.",
        )
    return mesh


def _validate_model_tensor(
    value: torch.Tensor,
    *,
    owner: str,
    field: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if value.layout != torch.strided:
        raise PotentialCapabilityError(
            "Magnetic model input requires a strided tensor layout.",
            object_name=owner,
            field=field,
            expected="torch.strided",
            actual=value.layout,
            code=ErrorCode.CAPABILITY_UNAVAILABLE,
            hint=f"Materialize {field} as a strided tensor before execution.",
        )
    if tuple(value.shape) != shape:
        raise PotentialContractError(
            "Magnetic model input has the wrong shape.",
            object_name=owner,
            field=field,
            expected=list(shape),
            actual=list(value.shape),
            code=ErrorCode.SHAPE_MISMATCH,
            hint=f"Provide the exact canonical {field} shape without implicit reshaping.",
        )
    if value.dtype not in {torch.float32, torch.float64}:
        raise PotentialCapabilityError(
            "Magnetic model input dtype is unsupported.",
            object_name=owner,
            field=field,
            expected=["torch.float32", "torch.float64"],
            actual=value.dtype,
            code=ErrorCode.DTYPE_UNSUPPORTED,
            hint=f"Use float32 or float64 SI {field} values.",
        )
    if value.device.type not in {"cpu", "cuda"}:
        raise PotentialCapabilityError(
            "Magnetic model input device is unsupported.",
            object_name=owner,
            field=field,
            expected=["cpu", "cuda"],
            actual=value.device,
            code=ErrorCode.DEVICE_UNAVAILABLE,
            hint=f"Move {field} explicitly to CPU or an available CUDA device.",
        )
    if not bool(torch.isfinite(value).all()):
        raise PotentialContractError(
            "Magnetic model input must be finite.",
            object_name=owner,
            field=field,
            expected="all finite values",
            actual="contains non-finite values",
            hint=f"Replace NaN and infinite {field} values before execution.",
        )
    return value


def _field_direction_up(field: EarthField) -> tuple[float, float, float]:
    """Return east, north, elevation-up direction cosines for ``field``."""
    inclination = math.radians(field.inclination_deg)
    declination = math.radians(field.declination_deg)
    horizontal = math.cos(inclination)
    return (
        horizontal * math.sin(declination),
        horizontal * math.cos(declination),
        -math.sin(inclination),
    )


def _earth_field_key(name: str, field: EarthField) -> tuple[str, str, str, str]:
    """Return exact canonical Earth-field metadata for a cache key."""

    return (
        name,
        float(field.intensity_tesla).hex(),
        float(field.inclination_deg).hex(),
        float(field.declination_deg).hex(),
    )


def _differentiability(
    execution: PotentialExecutionConfig,
    *,
    trainable_input: str,
    input_unit: str,
    output_keys: tuple[str, ...],
) -> DifferentiabilitySpec:
    level = (
        DifferentiabilityLevel.FULL_AUTOGRAD
        if execution.strategy in {"dense", "store"}
        else DifferentiabilityLevel.CUSTOM_VJP
    )
    return DifferentiabilitySpec(
        level=level,
        trainable_inputs=(trainable_input,),
        output_keys=output_keys,
        input_units={trainable_input: input_unit},
    )


class _MagneticOperator(ForwardOperator):  # type: ignore[misc, unused-ignore]
    """Shared immutable orchestration for the two public magnetic contracts."""

    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (PrismGeometryMesh,)

    survey: PotentialSurvey3D
    components: tuple[MagneticComponent, ...]
    execution: PotentialExecutionConfig
    projection_field: EarthField
    _sensitivity_cache: OrderedDict[tuple[object, ...], PotentialSensitivity]
    _configuration_frozen: bool

    def __setattr__(self, name: str, value: object) -> None:
        if name in _IMMUTABLE_CONFIGURATION_FIELDS and self.__dict__.get(
            "_configuration_frozen", False
        ):
            raise PotentialContractError(
                "Magnetic operator configuration is immutable after construction.",
                object_name=type(self).__name__,
                field=name,
                expected="construct a new operator for a different configuration",
                actual=value,
                hint="Create a fresh magnetic operator so physics, cache, metadata, and differentiation remain aligned.",
            )
        # PyTorch's type stub narrows ``Module.__setattr__`` to tensors and
        # modules even though the runtime deliberately accepts ordinary
        # configuration attributes as well.  This class stores validated,
        # non-module records (survey/config/field), so keep the runtime path
        # and localize the incomplete-stub suppression here.
        super().__setattr__(name, value)  # type: ignore[arg-type]

    def __delattr__(self, name: str) -> None:
        if name in _IMMUTABLE_CONFIGURATION_FIELDS and self.__dict__.get(
            "_configuration_frozen", False
        ):
            raise PotentialContractError(
                "Magnetic operator configuration cannot be deleted.",
                object_name=type(self).__name__,
                field=name,
                expected="immutable constructed configuration",
                actual="deletion requested",
                hint="Create a fresh magnetic operator instead of mutating this one.",
            )
        super().__delattr__(name)

    def _compile_sensitivity(
        self,
        *,
        mesh: TensorMesh,
        dtype: torch.dtype,
        device: torch.device,
        model_contract: Literal["chi", "magnetization"],
        projection: dict[str, tuple[float, float, float]],
        operator_physics: tuple[object, ...],
    ) -> tuple[PotentialSensitivity, str]:
        observations = self.survey.positions_m.to(dtype=dtype, device=device)
        bounds = tensor_mesh_cell_bounds_m(mesh).to(dtype=dtype, device=device)
        plan = PrismKernelPlan.build(
            observations_m=observations,
            cell_bounds_m=bounds,
            components=self.components,
            dtype=dtype,
            device=device,
        )
        projection_key = tuple(
            (name, tuple(float(coordinate).hex() for coordinate in vector))
            for name, vector in sorted(projection.items())
        )
        cache_key: tuple[object, ...] = (
            plan.fingerprint,
            model_contract,
            self.execution.strategy,
            self.execution.budget_bytes,
            self.execution.observation_tile_size,
            self.execution.cell_tile_size,
            projection_key,
            operator_physics,
        )
        cached = self._sensitivity_cache.pop(cache_key, None)
        if cached is not None:
            self._sensitivity_cache[cache_key] = cached
            return cached, "hit"
        sensitivity = build_sensitivity(
            plan=plan,
            execution=self.execution,
            model_contract=model_contract,
            projection=projection,
        )
        self._sensitivity_cache[cache_key] = sensitivity
        if len(self._sensitivity_cache) > 8:
            self._sensitivity_cache.popitem(last=False)
        return sensitivity, "miss"

    def _output(
        self,
        *,
        data: dict[str, torch.Tensor],
        sensitivity: PotentialSensitivity,
        cache_event: str,
        field_name: str,
        reporting_field: EarthField,
    ) -> ForwardOutput:
        estimate = sensitivity.estimate
        units = {
            component: "T" if component in _MAGNETIC_FIELD_COMPONENTS else "T/m"
            for component in self.components
        }
        return ForwardOutput(
            data=data,
            fields={},
            metadata={
                "units": units,
                "kernel": "rectangular_prism_si",
                "field_name": field_name,
                "field_intensity_tesla": reporting_field.intensity_tesla,
                "inclination_deg": reporting_field.inclination_deg,
                "declination_deg": reporting_field.declination_deg,
                "requested_strategy": self.execution.strategy,
                "selected_strategy": sensitivity.selected_strategy,
                "plan_fingerprint": sensitivity.plan.fingerprint,
                "observation_tile_size": estimate.observation_tile_size,
                "cell_tile_size": estimate.cell_tile_size,
                "cache_event": cache_event,
                "resource_estimate": estimate.to_dict(),
            },
        )


class Magnetic3D(_MagneticOperator):
    """Induced magnetic field and gradients from scalar SI susceptibility.

    Args:
        survey: 3-D station geometry.
        earth_field: inducing :class:`EarthField`.
        components: magnetic components to emit (``'tmi'``, ...).
        execution: optional strategy/tiling/budget policy.
    """

    def __init__(
        self,
        survey: PotentialSurvey3D,
        earth_field: EarthField,
        *,
        components: tuple[MagneticComponent, ...] = ("tmi",),
        execution: PotentialExecutionConfig | None = None,
    ) -> None:
        super().__init__()
        owner = type(self).__name__
        if not isinstance(survey, PotentialSurvey3D):
            raise PotentialContractError(
                "Magnetic3D requires PotentialSurvey3D.",
                object_name=owner,
                field="survey",
                expected="PotentialSurvey3D with (x, y, z) metre columns",
                actual=survey,
                hint="Construct an immutable PotentialSurvey3D first.",
            )
        if not isinstance(earth_field, EarthField):
            raise PotentialContractError(
                "Magnetic3D requires an EarthField.",
                object_name=owner,
                field="earth_field",
                expected="strict-SI EarthField",
                actual=earth_field,
                hint="Provide EarthField intensity in tesla and angles in degrees.",
            )
        self.survey = survey
        self.earth_field = earth_field
        self.components = _validate_components(components, owner=owner)
        self.execution = _execution_or_default(execution, owner=owner)
        self.differentiability = _differentiability(
            self.execution,
            trainable_input="chi",
            input_unit="dimensionless",
            output_keys=self.components,
        )
        self._sensitivity_cache = OrderedDict()
        self._configuration_frozen = True

    @classmethod
    def capabilities(cls) -> PotentialCapabilityReport:
        components = ("tmi", "bx", "by", "bz", "bxx", "bxy", "bxz", "byy", "byz", "bzz")
        return potential_capability_report(
            operator_id="magnetic-3d",
            components=components,
            output_units={
                component: "T" if component in {"tmi", "bx", "by", "bz"} else "T/m"
                for component in components
            },
            state_fields={"chi": "1"},
            survey_width=3,
        )

    @classmethod
    def input_schema(cls) -> dict[str, object]:
        return potential_input_schema(
            operator_id="magnetic-3d",
            components=("tmi", "bx", "by", "bz", "bxx", "bxy", "bxz", "byy", "byz", "bzz"),
            tensor_name="chi",
            survey_width=3,
            field_name="earth_field",
        )

    def estimate_resources(
        self,
        state: ModelState,
        ctx: ForwardContext,
    ) -> PotentialResourceEstimate:
        (chi_input,) = state.fetch("chi")
        mesh = _require_prism_mesh(ctx, owner=type(self).__name__)
        chi = _validate_model_tensor(
            chi_input,
            owner=type(self).__name__,
            field="chi",
            shape=(mesh.n_cells,),
        )
        plan = PrismKernelPlan.build(
            observations_m=self.survey.positions_m.to(dtype=chi.dtype, device=chi.device),
            cell_bounds_m=tensor_mesh_cell_bounds_m(mesh).to(
                dtype=chi.dtype,
                device=chi.device,
            ),
            components=self.components,
            dtype=chi.dtype,
            device=chi.device,
        )
        try:
            return estimate_prism_resources(
                plan=plan,
                execution=self.execution,
                model_field_count=1,
            )
        except PotentialResourceError:
            return infeasible_resource_estimate(self.execution)

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        owner = type(self).__name__
        (chi_input,) = state.fetch("chi")
        mesh = _require_prism_mesh(ctx, owner=owner)
        chi = _validate_model_tensor(
            chi_input,
            owner=owner,
            field="chi",
            shape=(mesh.n_cells,),
        )
        direction = _field_direction_up(self.earth_field)
        field_h = self.earth_field.intensity_tesla / _VACUUM_PERMEABILITY_SI
        sensitivity, cache_event = self._compile_sensitivity(
            mesh=mesh,
            dtype=chi.dtype,
            device=chi.device,
            model_contract="chi",
            projection={
                "earth_field": (
                    field_h * direction[0],
                    field_h * direction[1],
                    field_h * direction[2],
                ),
                "projection_field": direction,
            },
            operator_physics=_earth_field_key("earth_field", self.earth_field),
        )
        return self._output(
            data=sensitivity.apply({"chi": chi}),
            sensitivity=sensitivity,
            cache_event=cache_event,
            field_name="earth_field",
            reporting_field=self.earth_field,
        )


class MagneticVector3D(_MagneticOperator):
    """Magnetic field and gradients from packed Cartesian magnetization in A/m.

    Args:
        survey: 3-D station geometry.
        projection_field: :class:`EarthField` defining the TMI projection
            direction (magnetization itself is the inverted vector field).
        components: magnetic components to emit.
        execution: optional strategy/tiling/budget policy.
    """

    def __init__(
        self,
        survey: PotentialSurvey3D,
        projection_field: EarthField,
        *,
        components: tuple[MagneticComponent, ...] = ("tmi",),
        execution: PotentialExecutionConfig | None = None,
    ) -> None:
        super().__init__()
        owner = type(self).__name__
        if not isinstance(survey, PotentialSurvey3D):
            raise PotentialContractError(
                "MagneticVector3D requires PotentialSurvey3D.",
                object_name=owner,
                field="survey",
                expected="PotentialSurvey3D with (x, y, z) metre columns",
                actual=survey,
                hint="Construct an immutable PotentialSurvey3D first.",
            )
        if not isinstance(projection_field, EarthField):
            raise PotentialContractError(
                "MagneticVector3D requires an EarthField projection.",
                object_name=owner,
                field="projection_field",
                expected="strict-SI EarthField",
                actual=projection_field,
                hint="Provide the TMI projection direction as a strict-SI EarthField.",
            )
        self.survey = survey
        self.projection_field = projection_field
        self.components = _validate_components(components, owner=owner)
        self.execution = _execution_or_default(execution, owner=owner)
        self.differentiability = _differentiability(
            self.execution,
            trainable_input="magnetization",
            input_unit="A/m",
            output_keys=self.components,
        )
        self._sensitivity_cache = OrderedDict()
        self._configuration_frozen = True

    @classmethod
    def capabilities(cls) -> PotentialCapabilityReport:
        components = ("tmi", "bx", "by", "bz", "bxx", "bxy", "bxz", "byy", "byz", "bzz")
        return potential_capability_report(
            operator_id="magnetic-vector-3d",
            components=components,
            output_units={
                component: "T" if component in {"tmi", "bx", "by", "bz"} else "T/m"
                for component in components
            },
            state_fields={"magnetization": "A/m"},
            survey_width=3,
        )

    @classmethod
    def input_schema(cls) -> dict[str, object]:
        return potential_input_schema(
            operator_id="magnetic-vector-3d",
            components=("tmi", "bx", "by", "bz", "bxx", "bxy", "bxz", "byy", "byz", "bzz"),
            tensor_name="magnetization",
            survey_width=3,
            field_name="projection_field",
        )

    def estimate_resources(
        self,
        state: ModelState,
        ctx: ForwardContext,
    ) -> PotentialResourceEstimate:
        (magnetization_input,) = state.fetch("magnetization")
        mesh = _require_prism_mesh(ctx, owner=type(self).__name__)
        magnetization = _validate_model_tensor(
            magnetization_input,
            owner=type(self).__name__,
            field="magnetization",
            shape=(mesh.n_cells, 3),
        )
        plan = PrismKernelPlan.build(
            observations_m=self.survey.positions_m.to(
                dtype=magnetization.dtype,
                device=magnetization.device,
            ),
            cell_bounds_m=tensor_mesh_cell_bounds_m(mesh).to(
                dtype=magnetization.dtype,
                device=magnetization.device,
            ),
            components=self.components,
            dtype=magnetization.dtype,
            device=magnetization.device,
        )
        try:
            return estimate_prism_resources(
                plan=plan,
                execution=self.execution,
                model_field_count=3,
            )
        except PotentialResourceError:
            return infeasible_resource_estimate(self.execution)

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        owner = type(self).__name__
        (magnetization_input,) = state.fetch("magnetization")
        mesh = _require_prism_mesh(ctx, owner=owner)
        magnetization = _validate_model_tensor(
            magnetization_input,
            owner=owner,
            field="magnetization",
            shape=(mesh.n_cells, 3),
        )
        sensitivity, cache_event = self._compile_sensitivity(
            mesh=mesh,
            dtype=magnetization.dtype,
            device=magnetization.device,
            model_contract="magnetization",
            projection={"projection_field": _field_direction_up(self.projection_field)},
            operator_physics=_earth_field_key(
                "projection_field",
                self.projection_field,
            ),
        )
        return self._output(
            data=sensitivity.apply({"magnetization": magnetization}),
            sensitivity=sensitivity,
            cache_event=cache_event,
            field_name="projection_field",
            reporting_field=self.projection_field,
        )


__all__ = ["Magnetic3D", "MagneticComponent", "MagneticVector3D"]
