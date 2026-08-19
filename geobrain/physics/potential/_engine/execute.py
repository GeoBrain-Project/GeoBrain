"""Bounded dense, tiled, stored, transpose, and custom-VJP execution.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, cast

import torch

from geobrain.core import ErrorCode

from ..resources import (
    PotentialResourceEstimate,
    SelectedStrategy,
    estimate_prism_resources,
)
from ..config import PotentialExecutionConfig
from ..errors import (
    PotentialCapabilityError,
    PotentialContractError,
    PotentialNumericsError,
    PotentialResourceError,
)
from .blocks import evaluate_prism_block
from .plan import PrismKernelPlan


ModelContract: TypeAlias = Literal["rho", "chi", "magnetization"]
CoefficientVector: TypeAlias = tuple[float, float, float]
ProjectionMetadata: TypeAlias = tuple[tuple[str, CoefficientVector], ...]
_GRAVITY_COMPONENTS = frozenset({"gx", "gy", "gz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz"})
_MODEL_CONTRACTS = frozenset({"rho", "chi", "magnetization"})
_AXES = "xyz"
_MAX_RESOURCE_BYTES = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class _ResolvedTerm:
    internal_component: str
    model_axis: int
    coefficient: float


ResolvedComponents: TypeAlias = tuple[tuple[_ResolvedTerm, ...], ...]
StoredMatrices: TypeAlias = tuple[torch.Tensor, ...]


def _contract_error(
    message: str,
    *,
    field: str,
    expected: object,
    actual: object,
    hint: str,
    code: ErrorCode | None = None,
) -> PotentialContractError:
    return PotentialContractError(
        message,
        object_name="build_sensitivity",
        field=field,
        expected=expected,
        actual=actual,
        hint=hint,
        code=code,
    )


def _validate_coefficient_vector(value: object, *, field: str) -> CoefficientVector:
    if type(value) is not tuple or len(value) != 3:
        raise _contract_error(
            "Projection coefficients must be an immutable Cartesian triple.",
            field=field,
            expected="tuple of exactly three finite Python floats",
            actual=value,
            hint=f"Provide immutable ({field}_x, {field}_y, {field}_z) coefficients.",
        )
    if any(
        type(coefficient) is not float or not math.isfinite(coefficient) for coefficient in value
    ):
        raise _contract_error(
            "Projection coefficients must be finite Python floats.",
            field=field,
            expected="three finite Python floats; integers and booleans are forbidden",
            actual=value,
            hint=f"Provide finite float coefficients for {field} without implicit conversion.",
        )
    coefficients = cast(CoefficientVector, value)
    if coefficients == (0.0, 0.0, 0.0):
        raise _contract_error(
            "Projection coefficient vector cannot be identically zero.",
            field=field,
            expected="at least one non-zero Cartesian coefficient",
            actual=coefficients,
            hint=f"Provide a physically meaningful non-zero {field} vector.",
        )
    return coefficients


def _validate_projection(
    model_contract: ModelContract,
    projection: object,
) -> ProjectionMetadata | None:
    if model_contract == "rho":
        if projection is not None:
            raise _contract_error(
                "Gravity sensitivity does not accept projection metadata.",
                field="projection",
                expected=None,
                actual=projection,
                hint="Pass projection=None for the rho model contract.",
            )
        return None
    if not isinstance(projection, Mapping):
        raise _contract_error(
            "Magnetic sensitivity requires exact projection metadata.",
            field="projection",
            expected=(
                ["earth_field", "projection_field"]
                if model_contract == "chi"
                else ["projection_field"]
            ),
            actual=projection,
            hint="Provide the exact immutable coefficient mapping required by the model contract.",
        )
    keys = tuple(projection.keys())
    if any(not isinstance(key, str) for key in keys):
        raise _contract_error(
            "Projection metadata keys must be strings.",
            field="projection",
            expected="exact string key set",
            actual=keys,
            hint="Use only the named Earth/projection-field coefficient keys.",
        )
    expected_keys = (
        ("earth_field", "projection_field") if model_contract == "chi" else ("projection_field",)
    )
    if len(keys) != len(expected_keys) or frozenset(keys) != frozenset(expected_keys):
        raise _contract_error(
            "Projection metadata has the wrong key set.",
            field="projection",
            expected=list(expected_keys),
            actual=keys,
            hint="Remove extra keys and provide every coefficient vector required by the model contract.",
        )
    return tuple(
        (key, _validate_coefficient_vector(projection[key], field=key)) for key in expected_keys
    )


def _projection_vector(
    projection: ProjectionMetadata,
    name: str,
) -> CoefficientVector:
    for key, value in projection:
        if key == name:
            return value
    raise AssertionError(f"validated projection is missing {name}")


def _resolve_components(
    plan: PrismKernelPlan,
    model_contract: ModelContract,
    projection: ProjectionMetadata | None,
) -> tuple[tuple[str, ...], ResolvedComponents]:
    gravity_plan = frozenset(plan.components) <= _GRAVITY_COMPONENTS
    if gravity_plan and model_contract != "rho":
        raise PotentialCapabilityError(
            "Gravity components require the rho model contract.",
            object_name="build_sensitivity",
            field="model_contract",
            expected="rho for gravity components",
            actual=model_contract,
            hint="Use rho or build a magnetic-component plan for chi/magnetization.",
        )
    if not gravity_plan and model_contract == "rho":
        raise PotentialCapabilityError(
            "Magnetic components require chi or magnetization.",
            object_name="build_sensitivity",
            field="model_contract",
            expected=["chi", "magnetization"],
            actual=model_contract,
            hint="Select the scalar induced or packed vector-magnetization contract.",
        )

    internal_components: list[str] = []
    seen_internal: set[str] = set()
    resolved: list[tuple[_ResolvedTerm, ...]] = []
    if model_contract == "rho":
        for component in plan.components:
            term = _ResolvedTerm(component, 0, 1.0)
            resolved.append((term,))
            if component not in seen_internal:
                internal_components.append(component)
                seen_internal.add(component)
        return tuple(internal_components), tuple(resolved)

    if projection is None:  # pragma: no cover - resolver invariant
        raise AssertionError("magnetic resolution returned no projection")
    projection_field = _projection_vector(projection, "projection_field")
    earth_field = (
        _projection_vector(projection, "earth_field")
        if model_contract == "chi"
        else (1.0, 1.0, 1.0)
    )
    for component in plan.components:
        terms: list[_ResolvedTerm] = []
        if component == "tmi":
            field_axes = tuple(enumerate(_AXES))
        else:
            field_axes = ((_AXES.index(component[1]), component[1]),)
        for field_index, field_axis in field_axes:
            field_coefficient = projection_field[field_index] if component == "tmi" else 1.0
            for model_axis, magnetization_axis in enumerate(_AXES):
                if component != "tmi" and len(component) == 3:
                    internal = f"db{field_axis}_d{component[2]}_m{magnetization_axis}"
                else:
                    internal = f"b{field_axis}_m{magnetization_axis}"
                coefficient = field_coefficient
                if model_contract == "chi":
                    coefficient *= earth_field[model_axis]
                terms.append(_ResolvedTerm(internal, model_axis, coefficient))
                if internal not in seen_internal:
                    internal_components.append(internal)
                    seen_internal.add(internal)
        resolved.append(tuple(terms))
    return tuple(internal_components), tuple(resolved)


def _account_for_packed_store(
    estimate: PotentialResourceEstimate,
    *,
    model_contract: ModelContract,
) -> PotentialResourceEstimate:
    if estimate.selected_strategy != "store" or model_contract != "magnetization":
        return estimate
    persistent_bytes = estimate.persistent_bytes * 3
    peak_bytes = estimate.peak_bytes + (persistent_bytes - estimate.persistent_bytes)
    if peak_bytes > estimate.budget_bytes:
        raise PotentialResourceError(
            "Packed vector-magnetization storage exceeds the configured budget.",
            object_name="build_sensitivity",
            field="budget_bytes",
            expected=f">= {peak_bytes} bytes for packed MVI store execution",
            actual=estimate.budget_bytes,
            hint="Increase budget_bytes or select bounded tiled execution.",
        )
    return PotentialResourceEstimate(
        schema_version=estimate.schema_version,
        requested_strategy=estimate.requested_strategy,
        selected_strategy=estimate.selected_strategy,
        budget_bytes=estimate.budget_bytes,
        input_bytes=estimate.input_bytes,
        output_bytes=estimate.output_bytes,
        working_set_bytes=estimate.working_set_bytes,
        persistent_bytes=persistent_bytes,
        peak_bytes=peak_bytes,
        observation_tile_size=estimate.observation_tile_size,
        cell_tile_size=estimate.cell_tile_size,
        feasible=estimate.feasible,
        reason_code=estimate.reason_code,
    )


def _execution_resource_overflow(*, operands: tuple[int, ...]) -> PotentialResourceError:
    return PotentialResourceError(
        "Resolved Potential execution resource arithmetic exceeds the signed 64-bit contract.",
        object_name="build_sensitivity",
        field="resource_arithmetic",
        expected=f"checked result <= {_MAX_RESOURCE_BYTES}",
        actual={"operation": "multiply_or_add", "operands": operands},
        hint="Reduce geometry counts or the resolved magnetic component set before preflight.",
    )


def _checked_execution_product(*values: int) -> int:
    result = 1
    for value in values:
        if value < 0 or (value != 0 and result > _MAX_RESOURCE_BYTES // value):
            raise _execution_resource_overflow(operands=values)
        result *= value
    return result


def _checked_execution_add(*values: int) -> int:
    result = 0
    for value in values:
        if value < 0 or result > _MAX_RESOURCE_BYTES - value:
            raise _execution_resource_overflow(operands=values)
        result += value
    return result


def _account_for_resolved_execution(
    *,
    plan: PrismKernelPlan,
    estimate: PotentialResourceEstimate,
    model_contract: ModelContract,
    internal_components: tuple[str, ...],
    resolved_components: ResolvedComponents,
) -> PotentialResourceEstimate:
    observation_count = (
        estimate.observation_tile_size
        if estimate.selected_strategy == "tiled"
        else plan.n_observations
    )
    cell_count = estimate.cell_tile_size if estimate.selected_strategy == "tiled" else plan.n_cells
    if observation_count is None or cell_count is None:
        raise AssertionError("successful execution preflight did not resolve its working shape")
    itemsize = 4 if plan.dtype == torch.float32 else 8
    block_values = _checked_execution_product(observation_count, cell_count)

    # The resource estimate already counts one evaluator matrix per public
    # component; magnetic resolution expands those public projections to a
    # larger internal basis, so only the EXTRA fields are added here.
    extra_kernel_fields = max(0, len(internal_components) - len(plan.components))
    evaluator_bytes = _checked_execution_product(
        block_values,
        extra_kernel_fields,
        itemsize,
    )

    projection_bytes = 0
    if model_contract != "rho":
        maximum_terms = max(len(terms) for terms in resolved_components)
        # A resolved forward sum retains at most the unscaled term and one
        # accumulation beside its counted public output; transpose is analogous.
        forward_fields = min(2, maximum_terms)
        forward_bytes = _checked_execution_product(
            observation_count,
            forward_fields,
            itemsize,
        )
        transpose_bytes = _checked_execution_product(cell_count, 2, itemsize)
        projection_bytes = max(forward_bytes, transpose_bytes)
        if estimate.selected_strategy == "store":
            # Scalar accumulation retains old/result contribution matrices. Packed
            # MVI additionally stacks three completed axis matrices into its store.
            store_projection_fields = 3 if model_contract == "magnetization" else 2
            store_projection_bytes = _checked_execution_product(
                block_values,
                store_projection_fields,
                itemsize,
            )
            projection_bytes = max(projection_bytes, store_projection_bytes)

    working_set_bytes = _checked_execution_add(
        estimate.working_set_bytes,
        evaluator_bytes,
        projection_bytes,
    )
    peak_bytes = _checked_execution_add(
        estimate.input_bytes,
        estimate.output_bytes,
        working_set_bytes,
        estimate.persistent_bytes,
    )
    if peak_bytes > estimate.budget_bytes:
        raise PotentialResourceError(
            "Resolved Potential evaluator and projection temporaries exceed the configured budget.",
            object_name="build_sensitivity",
            field="budget_bytes",
            expected=f">= {peak_bytes} bytes for resolved {estimate.selected_strategy} execution",
            actual=estimate.budget_bytes,
            hint="Increase budget_bytes or select bounded tiled execution with smaller tiles.",
        )
    return PotentialResourceEstimate(
        schema_version=estimate.schema_version,
        requested_strategy=estimate.requested_strategy,
        selected_strategy=estimate.selected_strategy,
        budget_bytes=estimate.budget_bytes,
        input_bytes=estimate.input_bytes,
        output_bytes=estimate.output_bytes,
        working_set_bytes=working_set_bytes,
        persistent_bytes=estimate.persistent_bytes,
        peak_bytes=peak_bytes,
        observation_tile_size=estimate.observation_tile_size,
        cell_tile_size=estimate.cell_tile_size,
        feasible=estimate.feasible,
        reason_code=estimate.reason_code,
    )


def _with_requested_strategy(
    estimate: PotentialResourceEstimate,
    execution: PotentialExecutionConfig,
) -> PotentialResourceEstimate:
    if estimate.requested_strategy == execution.strategy:
        return estimate
    return PotentialResourceEstimate(
        schema_version=estimate.schema_version,
        requested_strategy=execution.strategy,
        selected_strategy=estimate.selected_strategy,
        budget_bytes=estimate.budget_bytes,
        input_bytes=estimate.input_bytes,
        output_bytes=estimate.output_bytes,
        working_set_bytes=estimate.working_set_bytes,
        persistent_bytes=estimate.persistent_bytes,
        peak_bytes=estimate.peak_bytes,
        observation_tile_size=estimate.observation_tile_size,
        cell_tile_size=estimate.cell_tile_size,
        feasible=estimate.feasible,
        reason_code=estimate.reason_code,
    )


def _resolved_tiled_estimate(
    *,
    plan: PrismKernelPlan,
    execution: PotentialExecutionConfig,
    model_field_count: int,
    model_contract: ModelContract,
    internal_components: tuple[str, ...],
    resolved_components: ResolvedComponents,
) -> PotentialResourceEstimate:
    """Search tiles against the final internal magnetic execution graph."""

    def candidate(
        observation_tile_size: int,
        cell_tile_size: int,
    ) -> PotentialResourceEstimate:
        bounded_execution = PotentialExecutionConfig(
            strategy="tiled",
            budget_bytes=execution.budget_bytes,
            observation_tile_size=observation_tile_size,
            cell_tile_size=cell_tile_size,
        )
        generic = estimate_prism_resources(
            plan=plan,
            execution=bounded_execution,
            model_field_count=model_field_count,
        )
        return _account_for_resolved_execution(
            plan=plan,
            estimate=generic,
            model_contract=model_contract,
            internal_components=internal_components,
            resolved_components=resolved_components,
        )

    def feasible(
        observation_tile_size: int,
        cell_tile_size: int,
    ) -> PotentialResourceEstimate | None:
        try:
            return candidate(observation_tile_size, cell_tile_size)
        except PotentialResourceError as error:
            if error.field != "budget_bytes":
                raise
            return None

    def largest(maximum: int, fits: Callable[[int], bool]) -> int | None:
        low = 1
        high = maximum
        answer: int | None = None
        while low <= high:
            middle = (low + high) // 2
            if fits(middle):
                answer = middle
                low = middle + 1
            else:
                high = middle - 1
        return answer

    requested_observations = execution.observation_tile_size
    requested_cells = execution.cell_tile_size
    observation_tile_size = 1 if requested_observations is None else requested_observations
    if requested_cells is None:
        cell_tile_size = largest(
            plan.n_cells,
            lambda value: feasible(observation_tile_size, value) is not None,
        )
        if cell_tile_size is None:
            return candidate(observation_tile_size, 1)
    else:
        cell_tile_size = requested_cells
        if feasible(observation_tile_size, cell_tile_size) is None:
            return candidate(observation_tile_size, cell_tile_size)

    if requested_observations is None:
        resolved_observations = largest(
            plan.n_observations,
            lambda value: feasible(value, cell_tile_size) is not None,
        )
        if resolved_observations is None:
            return candidate(1, cell_tile_size)
        observation_tile_size = resolved_observations

    # A tiled request is a real bounded execution substrate, never a disguised
    # full dense block. Preserve the historical non-full contract after the
    # final resolved-cost search rather than before it.
    if (
        observation_tile_size == plan.n_observations
        and cell_tile_size == plan.n_cells
        and (plan.n_observations > 1 or plan.n_cells > 1)
    ):
        if plan.n_observations > 1:
            observation_tile_size -= 1
        else:
            cell_tile_size -= 1

    result = candidate(observation_tile_size, cell_tile_size)
    return _with_requested_strategy(result, execution)


def _resolved_resource_estimate(
    *,
    plan: PrismKernelPlan,
    execution: PotentialExecutionConfig,
    model_field_count: int,
    model_contract: ModelContract,
    internal_components: tuple[str, ...],
    resolved_components: ResolvedComponents,
) -> PotentialResourceEstimate:
    if execution.strategy == "tiled":
        return _resolved_tiled_estimate(
            plan=plan,
            execution=execution,
            model_field_count=model_field_count,
            model_contract=model_contract,
            internal_components=internal_components,
            resolved_components=resolved_components,
        )

    selected_execution = execution
    if execution.strategy == "auto":
        selected_execution = PotentialExecutionConfig(
            strategy="dense",
            budget_bytes=execution.budget_bytes,
        )
    try:
        estimate = estimate_prism_resources(
            plan=plan,
            execution=selected_execution,
            model_field_count=model_field_count,
        )
        estimate = _account_for_packed_store(
            estimate,
            model_contract=model_contract,
        )
        estimate = _account_for_resolved_execution(
            plan=plan,
            estimate=estimate,
            model_contract=model_contract,
            internal_components=internal_components,
            resolved_components=resolved_components,
        )
    except PotentialResourceError as error:
        if execution.strategy != "auto" or error.field != "budget_bytes":
            raise
        return _resolved_tiled_estimate(
            plan=plan,
            execution=execution,
            model_field_count=model_field_count,
            model_contract=model_contract,
            internal_components=internal_components,
            resolved_components=resolved_components,
        )
    return _with_requested_strategy(estimate, execution)


def _mapping_keys(
    value: object,
    *,
    field: str,
    expected: tuple[str, ...],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PotentialContractError(
            f"{field} must be a mapping with an exact key set.",
            object_name="PotentialSensitivity",
            field=field,
            expected=list(expected),
            actual=value,
            hint=f"Provide a mapping containing exactly {list(expected)}.",
        )
    keys = tuple(value.keys())
    if (
        any(not isinstance(key, str) for key in keys)
        or len(keys) != len(expected)
        or frozenset(keys) != frozenset(expected)
    ):
        raise PotentialContractError(
            f"{field} has the wrong key set.",
            object_name="PotentialSensitivity",
            field=field,
            expected=list(expected),
            actual=keys,
            hint="Remove extra keys and provide every exact required key.",
        )
    return cast(Mapping[str, object], value)


def _validate_tensor(
    value: object,
    *,
    field: str,
    shape: tuple[int, ...],
    plan: PrismKernelPlan,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise PotentialContractError(
            "Potential execution value must be a torch.Tensor.",
            object_name="PotentialSensitivity",
            field=field,
            expected={"shape": list(shape), "dtype": plan.dtype, "device": plan.device},
            actual=value,
            hint="Provide the exact validated strided tensor without implicit conversion.",
        )
    if value.layout != torch.strided:
        raise PotentialCapabilityError(
            "Potential execution requires a strided tensor layout.",
            object_name="PotentialSensitivity",
            field=field,
            expected="torch.strided",
            actual=value.layout,
            hint="Materialize this value in a strided tensor before execution.",
        )
    if tuple(value.shape) != shape:
        raise PotentialContractError(
            "Potential execution value has the wrong shape.",
            object_name="PotentialSensitivity",
            field=field,
            expected=list(shape),
            actual=list(value.shape),
            code=ErrorCode.SHAPE_MISMATCH,
            hint="Pack the value in the exact scalar or cell-by-three contract shape.",
        )
    if value.dtype != plan.dtype:
        raise PotentialContractError(
            "Potential execution value has the wrong dtype.",
            object_name="PotentialSensitivity",
            field=field,
            expected=plan.dtype,
            actual=value.dtype,
            code=ErrorCode.DTYPE_UNSUPPORTED,
            hint="Use the plan dtype exactly; no implicit cast is performed.",
        )
    if value.device != plan.device:
        raise PotentialContractError(
            "Potential execution value is on the wrong device.",
            object_name="PotentialSensitivity",
            field=field,
            expected=plan.device,
            actual=value.device,
            code=ErrorCode.DEVICE_UNAVAILABLE,
            hint="Move the value explicitly to the exact plan device before execution.",
        )
    if not bool(torch.isfinite(value).all()):
        raise PotentialContractError(
            "Potential execution value must be finite.",
            object_name="PotentialSensitivity",
            field=field,
            expected="all finite values",
            actual="contains non-finite values",
            hint="Replace NaN and infinite values before execution.",
        )
    return value


def _checked_result(value: torch.Tensor, *, field: str) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        raise PotentialNumericsError(
            "Potential execution produced a non-finite result after valid preflight.",
            object_name="PotentialSensitivity",
            field=field,
            expected="finite SI result",
            actual={"shape": list(value.shape), "dtype": value.dtype, "device": value.device},
            hint="Use float64 or rescale the physical model without changing SI contracts.",
        )
    return value


def _evaluate_bounded_tile(
    observations: torch.Tensor,
    bounds: torch.Tensor,
    components: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    return evaluate_prism_block(observations, bounds, components)


def _component_forward(
    kernels: Mapping[str, torch.Tensor],
    terms: tuple[_ResolvedTerm, ...],
    model: torch.Tensor,
) -> torch.Tensor:
    result: torch.Tensor | None = None
    for term in terms:
        model_values = model if model.ndim == 1 else model[:, term.model_axis]
        contribution = (kernels[term.internal_component] @ model_values) * term.coefficient
        result = contribution if result is None else result + contribution
    if result is None:  # pragma: no cover - loop invariant
        raise AssertionError("execution loop produced no result")
    return result


def _projected_matrix(
    kernels: Mapping[str, torch.Tensor],
    terms: tuple[_ResolvedTerm, ...],
    *,
    vector_model: bool,
) -> torch.Tensor:
    if not vector_model:
        matrix: torch.Tensor | None = None
        for term in terms:
            contribution = kernels[term.internal_component] * term.coefficient
            matrix = contribution if matrix is None else matrix + contribution
        if matrix is None:  # pragma: no cover - store invariant
            raise AssertionError("stored execution produced no kernel matrix")
        return matrix
    matrices: list[torch.Tensor] = []
    template = kernels[terms[0].internal_component]
    for model_axis in range(3):
        axis_matrix = torch.zeros_like(template)
        for term in terms:
            if term.model_axis == model_axis:
                axis_matrix = axis_matrix + kernels[term.internal_component] * term.coefficient
        matrices.append(axis_matrix)
    return torch.stack(matrices, dim=-1)


def _component_transpose(
    kernels: Mapping[str, torch.Tensor],
    terms: tuple[_ResolvedTerm, ...],
    cotangent: torch.Tensor,
    *,
    vector_model: bool,
) -> torch.Tensor:
    cell_count = next(iter(kernels.values())).shape[1]
    if not vector_model:
        result = torch.zeros(cell_count, dtype=cotangent.dtype, device=cotangent.device)
        for term in terms:
            result = result + (kernels[term.internal_component].mT @ cotangent) * term.coefficient
        return result
    result = torch.zeros((cell_count, 3), dtype=cotangent.dtype, device=cotangent.device)
    for term in terms:
        contribution = (kernels[term.internal_component].mT @ cotangent) * term.coefficient
        result[:, term.model_axis] = result[:, term.model_axis] + contribution
    return result


class _TiledApplyFunction(torch.autograd.Function):  # type: ignore[misc, unused-ignore]
    @staticmethod
    def forward(
        ctx: Any,
        sensitivity: PotentialSensitivity,
        model: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.sensitivity = sensitivity
        ctx.set_materialize_grads(False)
        outputs = sensitivity._apply_tiled(model)
        return tuple(outputs.values())

    @staticmethod
    def backward(
        ctx: Any,
        *output_cotangents: torch.Tensor | None,
    ) -> tuple[None, torch.Tensor]:
        sensitivity = cast(PotentialSensitivity, ctx.sensitivity)
        if torch.is_grad_enabled():
            raise PotentialCapabilityError(
                "Tiled Potential custom VJP supports first-order differentiation only.",
                object_name="PotentialSensitivity",
                field="differentiation_order",
                expected="first-order model VJP",
                actual="higher-order derivative requested",
                hint="Use dense/store for full autograd or stop differentiation after the tiled VJP.",
            )
        cotangents: dict[str, torch.Tensor] = {}
        for index, component in enumerate(sensitivity.plan.components):
            output_cotangent = output_cotangents[index]
            if output_cotangent is None:
                output_cotangent = torch.zeros(
                    sensitivity.plan.n_observations,
                    dtype=sensitivity.plan.dtype,
                    device=sensitivity.plan.device,
                )
            cotangents[component] = output_cotangent
        gradient = sensitivity.transpose(cotangents)[sensitivity.model_contract]
        return None, gradient


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PotentialSensitivity:
    """Owned immutable Potential plan and its selected execution substrate."""

    plan: PrismKernelPlan
    execution: PotentialExecutionConfig
    model_contract: ModelContract
    projection: ProjectionMetadata | None
    estimate: PotentialResourceEstimate
    selected_strategy: SelectedStrategy
    _internal_components: tuple[str, ...] = field(repr=False)
    _resolved_components: ResolvedComponents = field(repr=False)
    _stored_matrices: StoredMatrices = field(repr=False)

    @classmethod
    def _create(
        cls,
        *,
        plan: PrismKernelPlan,
        execution: PotentialExecutionConfig,
        model_contract: ModelContract,
        projection: ProjectionMetadata | None,
        estimate: PotentialResourceEstimate,
        internal_components: tuple[str, ...],
        resolved_components: ResolvedComponents,
    ) -> PotentialSensitivity:
        selected_strategy = estimate.selected_strategy
        if selected_strategy is None:
            raise AssertionError("successful preflight did not select a strategy")
        sensitivity = cls.__new__(cls)
        object.__setattr__(sensitivity, "plan", plan)
        object.__setattr__(sensitivity, "execution", execution)
        object.__setattr__(sensitivity, "model_contract", model_contract)
        object.__setattr__(sensitivity, "projection", projection)
        object.__setattr__(sensitivity, "estimate", estimate)
        object.__setattr__(sensitivity, "selected_strategy", selected_strategy)
        object.__setattr__(sensitivity, "_internal_components", internal_components)
        object.__setattr__(sensitivity, "_resolved_components", resolved_components)
        object.__setattr__(sensitivity, "_stored_matrices", ())
        if selected_strategy == "store":
            matrices = sensitivity._materialize_store()
            object.__setattr__(sensitivity, "_stored_matrices", matrices)
        return sensitivity

    @property
    def _vector_model(self) -> bool:
        return self.model_contract == "magnetization"

    def _model(self, model_fields: object) -> torch.Tensor:
        mapping = _mapping_keys(
            model_fields,
            field="model_fields",
            expected=(self.model_contract,),
        )
        shape = (self.plan.n_cells, 3) if self._vector_model else (self.plan.n_cells,)
        return _validate_tensor(
            mapping[self.model_contract],
            field=self.model_contract,
            shape=shape,
            plan=self.plan,
        )

    def _cotangents(self, cotangents: object) -> dict[str, torch.Tensor]:
        mapping = _mapping_keys(
            cotangents,
            field="cotangents",
            expected=self.plan.components,
        )
        return {
            component: _validate_tensor(
                mapping[component],
                field=component,
                shape=(self.plan.n_observations,),
                plan=self.plan,
            )
            for component in self.plan.components
        }

    def _geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.plan._observations_m, self.plan._cell_bounds_m

    def _materialize_store(self) -> StoredMatrices:
        observations, bounds = self._geometry()
        kernels = _evaluate_bounded_tile(
            observations,
            bounds,
            self._internal_components,
        )
        matrices = tuple(
            _checked_result(
                _projected_matrix(
                    kernels,
                    terms,
                    vector_model=self._vector_model,
                ),
                field=component,
            ).detach()
            for component, terms in zip(
                self.plan.components,
                self._resolved_components,
                strict=True,
            )
        )
        return matrices

    def _apply_full(self, model: torch.Tensor) -> dict[str, torch.Tensor]:
        observations, bounds = self._geometry()
        kernels = _evaluate_bounded_tile(
            observations,
            bounds,
            self._internal_components,
        )
        return {
            component: _checked_result(
                _component_forward(kernels, terms, model),
                field=component,
            )
            for component, terms in zip(
                self.plan.components,
                self._resolved_components,
                strict=True,
            )
        }

    def _tile_sizes(self) -> tuple[int, int]:
        observation_tile_size = self.estimate.observation_tile_size
        cell_tile_size = self.estimate.cell_tile_size
        if observation_tile_size is None or cell_tile_size is None:
            raise AssertionError("tiled preflight did not resolve both tile sizes")
        return observation_tile_size, cell_tile_size

    def _apply_tiled(self, model: torch.Tensor) -> dict[str, torch.Tensor]:
        observations, bounds = self._geometry()
        observation_tile_size, cell_tile_size = self._tile_sizes()
        outputs = {
            component: torch.zeros(
                self.plan.n_observations,
                dtype=self.plan.dtype,
                device=self.plan.device,
            )
            for component in self.plan.components
        }
        for observation_start in range(0, self.plan.n_observations, observation_tile_size):
            observation_stop = min(
                observation_start + observation_tile_size,
                self.plan.n_observations,
            )
            observation_slice = slice(observation_start, observation_stop)
            for cell_start in range(0, self.plan.n_cells, cell_tile_size):
                cell_stop = min(cell_start + cell_tile_size, self.plan.n_cells)
                cell_slice = slice(cell_start, cell_stop)
                kernels = _evaluate_bounded_tile(
                    observations[observation_slice],
                    bounds[cell_slice],
                    self._internal_components,
                )
                model_tile = model[cell_slice]
                for component, terms in zip(
                    self.plan.components,
                    self._resolved_components,
                    strict=True,
                ):
                    contribution = _component_forward(kernels, terms, model_tile)
                    outputs[component][observation_slice] = (
                        outputs[component][observation_slice] + contribution
                    )
        return {
            component: _checked_result(outputs[component], field=component)
            for component in self.plan.components
        }

    def _apply_store(self, model: torch.Tensor) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for component, matrix in zip(
            self.plan.components,
            self._stored_matrices,
            strict=True,
        ):
            if self._vector_model:
                value = torch.sum(matrix * model.unsqueeze(0), dim=(1, 2))
            else:
                value = matrix @ model
            result[component] = _checked_result(value, field=component)
        return result

    def apply(self, model_fields: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply the exact model mapping using the preflight-selected strategy."""
        model = self._model(model_fields)
        if self.selected_strategy == "tiled":
            raw_outputs = _TiledApplyFunction.apply(  # type: ignore[no-untyped-call, unused-ignore]
                self,
                model,
            )
            outputs = (
                (raw_outputs,)
                if isinstance(raw_outputs, torch.Tensor)
                else cast(tuple[torch.Tensor, ...], raw_outputs)
            )
            return {
                component: outputs[index] for index, component in enumerate(self.plan.components)
            }
        if self.selected_strategy == "store":
            return self._apply_store(model)
        return self._apply_full(model)

    def _transpose_full(self, cotangents: Mapping[str, torch.Tensor]) -> torch.Tensor:
        observations, bounds = self._geometry()
        kernels = _evaluate_bounded_tile(
            observations,
            bounds,
            self._internal_components,
        )
        shape = (self.plan.n_cells, 3) if self._vector_model else (self.plan.n_cells,)
        result = torch.zeros(shape, dtype=self.plan.dtype, device=self.plan.device)
        for component, terms in zip(
            self.plan.components,
            self._resolved_components,
            strict=True,
        ):
            result = result + _component_transpose(
                kernels,
                terms,
                cotangents[component],
                vector_model=self._vector_model,
            )
        return result

    def _transpose_tiled(self, cotangents: Mapping[str, torch.Tensor]) -> torch.Tensor:
        observations, bounds = self._geometry()
        observation_tile_size, cell_tile_size = self._tile_sizes()
        shape = (self.plan.n_cells, 3) if self._vector_model else (self.plan.n_cells,)
        result = torch.zeros(shape, dtype=self.plan.dtype, device=self.plan.device)
        for observation_start in range(0, self.plan.n_observations, observation_tile_size):
            observation_stop = min(
                observation_start + observation_tile_size,
                self.plan.n_observations,
            )
            observation_slice = slice(observation_start, observation_stop)
            for cell_start in range(0, self.plan.n_cells, cell_tile_size):
                cell_stop = min(cell_start + cell_tile_size, self.plan.n_cells)
                cell_slice = slice(cell_start, cell_stop)
                kernels = _evaluate_bounded_tile(
                    observations[observation_slice],
                    bounds[cell_slice],
                    self._internal_components,
                )
                tile_result = torch.zeros(
                    (cell_stop - cell_start, 3)
                    if self._vector_model
                    else (cell_stop - cell_start,),
                    dtype=self.plan.dtype,
                    device=self.plan.device,
                )
                for component, terms in zip(
                    self.plan.components,
                    self._resolved_components,
                    strict=True,
                ):
                    tile_result = tile_result + _component_transpose(
                        kernels,
                        terms,
                        cotangents[component][observation_slice],
                        vector_model=self._vector_model,
                    )
                result[cell_slice] = result[cell_slice] + tile_result
        return result

    def _transpose_store(self, cotangents: Mapping[str, torch.Tensor]) -> torch.Tensor:
        shape = (self.plan.n_cells, 3) if self._vector_model else (self.plan.n_cells,)
        result = torch.zeros(shape, dtype=self.plan.dtype, device=self.plan.device)
        for component, matrix in zip(
            self.plan.components,
            self._stored_matrices,
            strict=True,
        ):
            if self._vector_model:
                contribution = torch.sum(
                    matrix * cotangents[component][:, None, None],
                    dim=0,
                )
            else:
                contribution = matrix.mT @ cotangents[component]
            result = result + contribution
        return result

    def transpose(self, cotangents: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply the exact analytic transpose to the component cotangent mapping."""
        validated = self._cotangents(cotangents)
        if self.selected_strategy == "tiled":
            result = self._transpose_tiled(validated)
        elif self.selected_strategy == "store":
            result = self._transpose_store(validated)
        else:
            result = self._transpose_full(validated)
        return {self.model_contract: _checked_result(result, field=self.model_contract)}


def build_sensitivity(
    *,
    plan: PrismKernelPlan,
    execution: PotentialExecutionConfig,
    model_contract: str,
    projection: Mapping[str, CoefficientVector] | None,
) -> PotentialSensitivity:
    """Validate, preflight, and build one immutable Potential sensitivity.

    ``rho`` requires ``projection=None``. ``chi`` requires exact immutable
    ``earth_field`` coefficients in A/m per unit susceptibility and exact
    dimensionless ``projection_field`` coefficients. ``magnetization`` uses
    only the latter projection triple for packed A/m Cartesian model columns.
    """
    if not isinstance(plan, PrismKernelPlan):
        raise _contract_error(
            "Potential sensitivity requires an immutable PrismKernelPlan.",
            field="plan",
            expected="PrismKernelPlan",
            actual=plan,
            hint="Build and validate a plan before constructing a sensitivity.",
        )
    if not isinstance(execution, PotentialExecutionConfig):
        raise _contract_error(
            "Potential sensitivity requires PotentialExecutionConfig.",
            field="execution",
            expected="PotentialExecutionConfig",
            actual=execution,
            hint="Provide an explicit validated execution policy.",
        )
    if type(model_contract) is not str or model_contract not in _MODEL_CONTRACTS:
        raise _contract_error(
            "Potential sensitivity requires exactly one supported model contract.",
            field="model_contract",
            expected=["rho", "chi", "magnetization"],
            actual=model_contract,
            hint="Select one exact scalar-density, scalar-susceptibility, or packed-vector contract.",
        )
    validated_model_contract = cast(ModelContract, model_contract)
    validated_projection = _validate_projection(validated_model_contract, projection)
    internal_components, resolved_components = _resolve_components(
        plan,
        validated_model_contract,
        validated_projection,
    )
    model_field_count = 3 if validated_model_contract == "magnetization" else 1
    estimate = _resolved_resource_estimate(
        plan=plan,
        execution=execution,
        model_field_count=model_field_count,
        model_contract=validated_model_contract,
        internal_components=internal_components,
        resolved_components=resolved_components,
    )
    return PotentialSensitivity._create(
        plan=plan,
        execution=execution,
        model_contract=validated_model_contract,
        projection=validated_projection,
        estimate=estimate,
        internal_components=internal_components,
        resolved_components=resolved_components,
    )


__all__ = ["PotentialSensitivity", "build_sensitivity"]
