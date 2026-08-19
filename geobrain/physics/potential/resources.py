"""Allocation-free resource estimation for Potential execution.

Split out of ``capabilities.py`` (family-template alignment: every physics
family keeps its resource estimation in ``resources.py``). Checked 64-bit
byte arithmetic, the :class:`PotentialResourceEstimate` record, and the
prism-kernel estimator/tiling planner live here.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, NoReturn

import torch

from ._engine.plan import PrismKernelPlan
from .config import PotentialExecutionConfig, PotentialStrategy
from .errors import PotentialContractError, PotentialResourceError

SelectedStrategy = Literal["dense", "tiled", "store"]

_RESOURCE_SCHEMA: Literal["geobrain.potential.resource/1.0"] = (
    "geobrain.potential.resource/1.0"
)
_MAX_RESOURCE_BYTES = (1 << 63) - 1
_FIXED_WORKSPACE_BYTES = 65_536
_ANALYTIC_WORKSPACE_FIELDS = 64


def _resource_overflow(*, operation: str, operands: tuple[int, ...]) -> PotentialResourceError:
    return PotentialResourceError(
        "Potential resource arithmetic exceeds the signed 64-bit byte contract.",
        object_name="estimate_prism_resources",
        field="resource_arithmetic",
        expected=f"checked result <= {_MAX_RESOURCE_BYTES}",
        actual={"operation": operation, "operands": operands},
        hint="Reduce geometry counts, component count, or model field count before preflight.",
    )


def _checked_multiply(*values: int) -> int:
    result = 1
    for value in values:
        if value < 0:
            raise _resource_overflow(operation="multiply", operands=values)
        if value != 0 and result > _MAX_RESOURCE_BYTES // value:
            raise _resource_overflow(operation="multiply", operands=values)
        result *= value
    return result


def _checked_add(*values: int) -> int:
    result = 0
    for value in values:
        if value < 0 or result > _MAX_RESOURCE_BYTES - value:
            raise _resource_overflow(operation="add", operands=values)
        result += value
    return result


@dataclass(frozen=True, slots=True)
class PotentialResourceEstimate:
    """Conservative deterministic byte partitions for one Potential plan.

    Attributes:
        schema_version: estimate schema tag.
        requested_strategy / selected_strategy: planner decision.
        budget_bytes: budget the plan was made against.
        input_bytes / output_bytes / working_set_bytes /
            persistent_bytes / peak_bytes: memory terms.
        observation_tile_size / cell_tile_size: chosen tiling.
        feasible / reason_code: whether the plan fits, and why not.
    """

    schema_version: Literal["geobrain.potential.resource/1.0"]
    requested_strategy: PotentialStrategy
    selected_strategy: SelectedStrategy | None
    budget_bytes: int
    input_bytes: int
    output_bytes: int
    working_set_bytes: int
    persistent_bytes: int
    peak_bytes: int
    observation_tile_size: int | None
    cell_tile_size: int | None
    feasible: bool
    reason_code: str | None

    def __post_init__(self) -> None:
        strategies = {"auto", "dense", "tiled", "store"}
        selected = {"dense", "tiled", "store"}
        if self.schema_version != _RESOURCE_SCHEMA:
            raise PotentialContractError(
                "Resource estimate schema version is invalid.",
                object_name=type(self).__name__,
                field="schema_version",
                expected=_RESOURCE_SCHEMA,
                actual=self.schema_version,
                hint="Construct estimates through estimate_prism_resources().",
            )
        if (
            type(self.requested_strategy) is not str
            or self.requested_strategy not in strategies
        ):
            raise PotentialContractError(
                "Resource estimate requested strategy is invalid.",
                object_name=type(self).__name__,
                field="requested_strategy",
                expected=sorted(strategies),
                actual=self.requested_strategy,
                hint="Use one exact Potential execution strategy.",
            )
        if self.selected_strategy is not None and (
            type(self.selected_strategy) is not str
            or self.selected_strategy not in selected
        ):
            raise PotentialContractError(
                "Resource estimate selected strategy is invalid.",
                object_name=type(self).__name__,
                field="selected_strategy",
                expected=sorted(selected),
                actual=self.selected_strategy,
                hint="Use dense, tiled, store, or None for an infeasible estimate.",
            )
        byte_fields = {
            "budget_bytes": self.budget_bytes,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "working_set_bytes": self.working_set_bytes,
            "persistent_bytes": self.persistent_bytes,
            "peak_bytes": self.peak_bytes,
        }
        for field_name, value in byte_fields.items():
            minimum = 1 if field_name == "budget_bytes" else 0
            if type(value) is not int or value < minimum or value > _MAX_RESOURCE_BYTES:
                raise PotentialContractError(
                    "Resource estimate byte field is invalid.",
                    object_name=type(self).__name__,
                    field=field_name,
                    expected=f"non-Boolean integer in [{minimum}, {_MAX_RESOURCE_BYTES}]",
                    actual=value,
                    hint="Construct estimates through checked resource preflight.",
                )
        expected_peak = _checked_add(
            self.input_bytes,
            self.output_bytes,
            self.working_set_bytes,
            self.persistent_bytes,
        )
        if self.peak_bytes != expected_peak:
            raise PotentialContractError(
                "Resource peak does not equal its exact byte partitions.",
                object_name=type(self).__name__,
                field="peak_bytes",
                expected=expected_peak,
                actual=self.peak_bytes,
                hint="Sum input, output, working-set, and persistent partitions exactly.",
            )
        for field_name, tile_size in (
            ("observation_tile_size", self.observation_tile_size),
            ("cell_tile_size", self.cell_tile_size),
        ):
            if tile_size is not None and (type(tile_size) is not int or tile_size <= 0):
                raise PotentialContractError(
                    "Resource estimate tile size is invalid.",
                    object_name=type(self).__name__,
                    field=field_name,
                    expected="positive non-Boolean int or None",
                    actual=tile_size,
                    hint="Use resolved positive tile sizes only for tiled execution.",
                )
        if type(self.feasible) is not bool:
            raise PotentialContractError(
                "Resource estimate feasibility flag must be Boolean.",
                object_name=type(self).__name__,
                field="feasible",
                expected="bool",
                actual=self.feasible,
                hint="Use an exact Boolean feasibility result.",
            )
        if self.feasible:
            if self.selected_strategy is None or self.reason_code is not None:
                raise PotentialContractError(
                    "Feasible resource estimates require a selected strategy and no reason code.",
                    object_name=type(self).__name__,
                    field="feasible",
                    expected="selected strategy with reason_code=None",
                    actual={
                        "selected_strategy": self.selected_strategy,
                        "reason_code": self.reason_code,
                    },
                    hint="Construct feasible estimates through successful preflight.",
                )
            if self.peak_bytes > self.budget_bytes:
                raise PotentialContractError(
                    "Feasible resource estimate exceeds its budget.",
                    object_name=type(self).__name__,
                    field="budget_bytes",
                    expected=f">= {self.peak_bytes}",
                    actual=self.budget_bytes,
                    hint="Increase the budget or use smaller tiled execution.",
                )
        elif not isinstance(self.reason_code, str) or not self.reason_code:
            raise PotentialContractError(
                "Infeasible resource estimates require a stable reason code.",
                object_name=type(self).__name__,
                field="reason_code",
                expected="non-empty string",
                actual=self.reason_code,
                hint="Record the exact reason preflight could not select a strategy.",
            )

    def to_dict(self) -> dict[str, object]:
        """Return the exact stable JSON transport for Agent clients."""
        return {
            "schema_version": self.schema_version,
            "requested_strategy": self.requested_strategy,
            "selected_strategy": self.selected_strategy,
            "budget_bytes": self.budget_bytes,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "working_set_bytes": self.working_set_bytes,
            "persistent_bytes": self.persistent_bytes,
            "peak_bytes": self.peak_bytes,
            "observation_tile_size": self.observation_tile_size,
            "cell_tile_size": self.cell_tile_size,
            "feasible": self.feasible,
            "reason_code": self.reason_code,
        }


def infeasible_resource_estimate(
    execution: PotentialExecutionConfig,
    *,
    reason_code: str = "budget_exceeded",
) -> PotentialResourceEstimate:
    """Return a closed non-throwing Agent preflight result."""
    return PotentialResourceEstimate(
        schema_version=_RESOURCE_SCHEMA,
        requested_strategy=execution.strategy,
        selected_strategy=None,
        budget_bytes=execution.budget_bytes,
        input_bytes=0,
        output_bytes=0,
        working_set_bytes=0,
        persistent_bytes=0,
        peak_bytes=0,
        observation_tile_size=execution.observation_tile_size,
        cell_tile_size=execution.cell_tile_size,
        feasible=False,
        reason_code=reason_code,
    )


def _input_output_bytes(
    plan: PrismKernelPlan,
    model_field_count: int,
) -> tuple[int, int, int]:
    itemsize = 4 if plan.dtype == torch.float32 else 8
    observation_bytes = _checked_multiply(plan.n_observations, 3, itemsize)
    bounds_bytes = _checked_multiply(plan.n_cells, 6, itemsize)
    model_bytes = _checked_multiply(plan.n_cells, model_field_count, itemsize)
    input_bytes = _checked_add(observation_bytes, bounds_bytes, model_bytes)
    output_bytes = _checked_multiply(
        plan.n_observations,
        len(plan.components),
        itemsize,
    )
    return itemsize, input_bytes, output_bytes


def _working_bytes(
    plan: PrismKernelPlan,
    *,
    model_field_count: int,
    observation_tile_size: int,
    cell_tile_size: int,
    itemsize: int,
) -> int:
    block_bytes = _checked_multiply(
        observation_tile_size,
        cell_tile_size,
        len(plan.components) + _ANALYTIC_WORKSPACE_FIELDS,
        itemsize,
    )
    observation_adjoint_values = _checked_multiply(
        observation_tile_size,
        len(plan.components),
    )
    cell_adjoint_values = _checked_multiply(cell_tile_size, model_field_count)
    adjoint_bytes = _checked_multiply(
        _checked_add(observation_adjoint_values, cell_adjoint_values),
        itemsize,
    )
    return _checked_add(_FIXED_WORKSPACE_BYTES, block_bytes, adjoint_bytes)


def _partitions(
    plan: PrismKernelPlan,
    *,
    model_field_count: int,
    observation_tile_size: int,
    cell_tile_size: int,
    persistent: bool,
) -> tuple[int, int, int, int, int]:
    itemsize, input_bytes, output_bytes = _input_output_bytes(plan, model_field_count)
    working_set_bytes = _working_bytes(
        plan,
        model_field_count=model_field_count,
        observation_tile_size=observation_tile_size,
        cell_tile_size=cell_tile_size,
        itemsize=itemsize,
    )
    persistent_bytes = (
        _checked_multiply(
            plan.n_observations,
            plan.n_cells,
            len(plan.components),
            itemsize,
        )
        if persistent
        else 0
    )
    peak_bytes = _checked_add(
        input_bytes,
        output_bytes,
        working_set_bytes,
        persistent_bytes,
    )
    return input_bytes, output_bytes, working_set_bytes, persistent_bytes, peak_bytes


def _estimate(
    *,
    plan: PrismKernelPlan,
    execution: PotentialExecutionConfig,
    model_field_count: int,
    selected_strategy: SelectedStrategy,
    observation_tile_size: int | None,
    cell_tile_size: int | None,
    partitions: tuple[int, int, int, int, int] | None = None,
) -> PotentialResourceEstimate:
    execution_observations = (
        plan.n_observations if observation_tile_size is None else observation_tile_size
    )
    execution_cells = plan.n_cells if cell_tile_size is None else cell_tile_size
    if partitions is None:
        partitions = _partitions(
            plan,
            model_field_count=model_field_count,
            observation_tile_size=execution_observations,
            cell_tile_size=execution_cells,
            persistent=selected_strategy == "store",
        )
    input_bytes, output_bytes, working_set_bytes, persistent_bytes, peak_bytes = partitions
    return PotentialResourceEstimate(
        schema_version=_RESOURCE_SCHEMA,
        requested_strategy=execution.strategy,
        selected_strategy=selected_strategy,
        budget_bytes=execution.budget_bytes,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        working_set_bytes=working_set_bytes,
        persistent_bytes=persistent_bytes,
        peak_bytes=peak_bytes,
        observation_tile_size=observation_tile_size,
        cell_tile_size=cell_tile_size,
        feasible=True,
        reason_code=None,
    )


def _raise_budget_error(
    *,
    strategy: str,
    budget_bytes: int,
    required_bytes: int,
) -> NoReturn:
    raise PotentialResourceError(
        "Potential resource preflight cannot satisfy the configured budget.",
        object_name="estimate_prism_resources",
        field="budget_bytes",
        expected=f">= {required_bytes} bytes for {strategy} execution",
        actual=budget_bytes,
        hint="Increase budget_bytes or select tiled execution with smaller explicit tiles.",
    )


def _largest_feasible(maximum: int, fits: Callable[[int], bool]) -> int | None:
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


def _tiled_estimate(
    *,
    plan: PrismKernelPlan,
    execution: PotentialExecutionConfig,
    model_field_count: int,
) -> PotentialResourceEstimate:
    requested_observations = execution.observation_tile_size
    requested_cells = execution.cell_tile_size
    if requested_observations is not None and requested_observations > plan.n_observations:
        raise PotentialContractError(
            "Observation tile exceeds the plan observation count.",
            object_name="estimate_prism_resources",
            field="observation_tile_size",
            expected=f"<= {plan.n_observations}",
            actual=requested_observations,
            hint="Use a tile no larger than the immutable plan observation count.",
        )
    if requested_cells is not None and requested_cells > plan.n_cells:
        raise PotentialContractError(
            "Cell tile exceeds the plan cell count.",
            object_name="estimate_prism_resources",
            field="cell_tile_size",
            expected=f"<= {plan.n_cells}",
            actual=requested_cells,
            hint="Use a tile no larger than the immutable plan cell count.",
        )

    observation_tile_size = 1 if requested_observations is None else requested_observations

    def cell_fits(cell_tile_size: int) -> bool:
        candidate = _partitions(
            plan,
            model_field_count=model_field_count,
            observation_tile_size=observation_tile_size,
            cell_tile_size=cell_tile_size,
            persistent=False,
        )
        return candidate[-1] <= execution.budget_bytes

    if requested_cells is None:
        cell_tile_size = _largest_feasible(plan.n_cells, cell_fits)
    else:
        cell_tile_size = requested_cells if cell_fits(requested_cells) else None
    if cell_tile_size is None:
        minimum = _partitions(
            plan,
            model_field_count=model_field_count,
            observation_tile_size=observation_tile_size,
            cell_tile_size=1,
            persistent=False,
        )[-1]
        _raise_budget_error(
            strategy="tiled",
            budget_bytes=execution.budget_bytes,
            required_bytes=minimum,
        )

    if requested_observations is None:

        def observation_fits(candidate_observations: int) -> bool:
            candidate = _partitions(
                plan,
                model_field_count=model_field_count,
                observation_tile_size=candidate_observations,
                cell_tile_size=cell_tile_size,
                persistent=False,
            )
            return candidate[-1] <= execution.budget_bytes

        resolved_observations = _largest_feasible(
            plan.n_observations,
            observation_fits,
        )
        if resolved_observations is None:
            minimum = _partitions(
                plan,
                model_field_count=model_field_count,
                observation_tile_size=1,
                cell_tile_size=cell_tile_size,
                persistent=False,
            )[-1]
            _raise_budget_error(
                strategy="tiled",
                budget_bytes=execution.budget_bytes,
                required_bytes=minimum,
            )
        observation_tile_size = resolved_observations

    return _estimate(
        plan=plan,
        execution=execution,
        model_field_count=model_field_count,
        selected_strategy="tiled",
        observation_tile_size=observation_tile_size,
        cell_tile_size=cell_tile_size,
    )


def estimate_prism_resources(
    *,
    plan: PrismKernelPlan,
    execution: PotentialExecutionConfig,
    model_field_count: int,
) -> PotentialResourceEstimate:
    """Preflight Potential bytes with checked integers and no tensor allocation."""
    if not isinstance(plan, PrismKernelPlan):
        raise PotentialContractError(
            "Resource preflight requires an immutable PrismKernelPlan.",
            object_name="estimate_prism_resources",
            field="plan",
            expected="PrismKernelPlan",
            actual=plan,
            hint="Build and validate a plan before resource estimation.",
        )
    if not isinstance(execution, PotentialExecutionConfig):
        raise PotentialContractError(
            "Resource preflight requires PotentialExecutionConfig.",
            object_name="estimate_prism_resources",
            field="execution",
            expected="PotentialExecutionConfig",
            actual=execution,
            hint="Provide a validated explicit execution policy.",
        )
    if type(model_field_count) is not int or model_field_count <= 0:
        raise PotentialContractError(
            "Model field count must be a positive non-Boolean integer.",
            object_name="estimate_prism_resources",
            field="model_field_count",
            expected="positive int; bool is forbidden",
            actual=model_field_count,
            hint="Use 1 for scalar properties or 3 for vector magnetization.",
        )

    dense_partitions = _partitions(
        plan,
        model_field_count=model_field_count,
        observation_tile_size=plan.n_observations,
        cell_tile_size=plan.n_cells,
        persistent=False,
    )
    if execution.strategy == "auto":
        if dense_partitions[-1] <= execution.budget_bytes:
            return _estimate(
                plan=plan,
                execution=execution,
                model_field_count=model_field_count,
                selected_strategy="dense",
                observation_tile_size=None,
                cell_tile_size=None,
                partitions=dense_partitions,
            )
        return _tiled_estimate(
            plan=plan,
            execution=execution,
            model_field_count=model_field_count,
        )
    if execution.strategy == "tiled":
        return _tiled_estimate(
            plan=plan,
            execution=execution,
            model_field_count=model_field_count,
        )
    selected: SelectedStrategy = "dense" if execution.strategy == "dense" else "store"
    partitions = (
        dense_partitions
        if selected == "dense"
        else _partitions(
            plan,
            model_field_count=model_field_count,
            observation_tile_size=plan.n_observations,
            cell_tile_size=plan.n_cells,
            persistent=True,
        )
    )
    if partitions[-1] > execution.budget_bytes:
        _raise_budget_error(
            strategy=selected,
            budget_bytes=execution.budget_bytes,
            required_bytes=partitions[-1],
        )
    return _estimate(
        plan=plan,
        execution=execution,
        model_field_count=model_field_count,
        selected_strategy=selected,
        observation_tile_size=None,
        cell_tile_size=None,
        partitions=partitions,
    )


__all__ = [
    "PotentialResourceEstimate",
    "SelectedStrategy",
    "estimate_prism_resources",
    "infeasible_resource_estimate",
]
