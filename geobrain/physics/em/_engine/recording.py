"""Bounded state recording and deterministic recurrence checkpointing.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import math
import sys
from typing import Literal, cast

import torch
from torch.utils.checkpoint import checkpoint

from geobrain.physics.em.errors import EMCapabilityError, EMContractError, EMResourceError

from .contracts import RecordingDiagnostics


RecordingMode = Literal["output_only", "gate_states", "checkpoint_recompute"]
StepFunction = Callable[..., torch.Tensor]
ObservationFunction = Callable[[torch.Tensor], torch.Tensor]


def _contract_error(message: str, *, field: str, expected: object, actual: object) -> None:
    raise EMContractError(
        message,
        object_name="RecordingPolicy",
        field=field,
        expected=expected,
        actual=actual,
        details={
            "field": field,
            "received_type": type(actual).__qualname__,
            "remediation": "provide a bounded canonical TEM recording policy",
        },
        hint="use output_only, gate_states, or checkpoint_recompute",
    )


def _checked_mul(*values: int, field: str) -> int:
    total = 1
    for value in values:
        if value != 0 and total > sys.maxsize // value:
            raise EMResourceError(
                "TEM recording size exceeds the addressable integer range",
                object_name="prepare_recording",
                field=field,
                expected=f"value <= {sys.maxsize}",
                actual="overflow",
                details={
                    "field": field,
                    "limit_bytes": sys.maxsize,
                    "operands": values,
                    "remediation": "reduce state, gate, or checkpoint counts",
                },
                hint="reduce the requested TEM recording",
            )
        total *= value
    return total


def _checked_add(*values: int, field: str) -> int:
    total = 0
    for value in values:
        if value > sys.maxsize - total:
            raise EMResourceError(
                "TEM recording size exceeds the addressable integer range",
                object_name="prepare_recording",
                field=field,
                expected=f"value <= {sys.maxsize}",
                actual="overflow",
                details={
                    "field": field,
                    "limit_bytes": sys.maxsize,
                    "operands": values,
                    "remediation": "reduce state, gate, or checkpoint counts",
                },
                hint="reduce the requested TEM recording",
            )
        total += value
    return total


@dataclass(frozen=True, slots=True)
class RecordingPolicy:
    """Explicit bounded state-retention policy for one TEM forward."""

    mode: RecordingMode
    checkpoint_interval: int | None = None
    budget_bytes: int | None = None

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in (
            "output_only",
            "gate_states",
            "checkpoint_recompute",
        ):
            _contract_error(
                "unsupported TEM recording mode",
                field="mode",
                expected=("checkpoint_recompute", "gate_states", "output_only"),
                actual=self.mode,
            )
        if self.mode == "checkpoint_recompute":
            if type(self.checkpoint_interval) is not int or self.checkpoint_interval <= 0:
                _contract_error(
                    "checkpoint recomputation requires a positive interval",
                    field="checkpoint_interval",
                    expected="positive int",
                    actual=self.checkpoint_interval,
                )
        elif self.checkpoint_interval is not None:
            _contract_error(
                "checkpoint interval is valid only for checkpoint recomputation",
                field="checkpoint_interval",
                expected=None,
                actual=self.checkpoint_interval,
            )
        if self.budget_bytes is not None and (
            type(self.budget_bytes) is not int or self.budget_bytes < 0
        ):
            _contract_error(
                "recording budget must be a non-negative exact integer or None",
                field="budget_bytes",
                expected="non-negative int or None",
                actual=self.budget_bytes,
            )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-safe policy snapshot."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecordingPlan:
    """Validated allocation preflight for one recorded recurrence."""

    policy: RecordingPolicy
    gate_history_indices: tuple[int, ...]
    state_shape: tuple[int, ...]
    state_dtype: torch.dtype
    state_device: torch.device
    observation_shape: tuple[int, ...]
    observation_dtype: torch.dtype
    diagnostics: RecordingDiagnostics


def prepare_recording(
    policy: RecordingPolicy,
    *,
    n_steps: int,
    gate_history_indices: Sequence[int],
    state: torch.Tensor,
    observation_shape: Sequence[int],
    observation_dtype: torch.dtype,
    requires_gradient: bool,
) -> RecordingPlan:
    """Validate and budget a recurrence before its first numerical step."""
    if type(policy) is not RecordingPolicy:
        _contract_error(
            "recording policy must be an exact RecordingPolicy",
            field="policy",
            expected="RecordingPolicy",
            actual=policy,
        )
    if type(n_steps) is not int or n_steps <= 0:
        _contract_error(
            "TEM recording requires a positive exact step count",
            field="n_steps",
            expected="positive int",
            actual=n_steps,
        )
    if not isinstance(state, torch.Tensor) or state.numel() == 0:
        _contract_error(
            "TEM recording requires a non-empty tensor state",
            field="state",
            expected="non-empty torch.Tensor",
            actual=state,
        )
    if type(requires_gradient) is not bool:
        _contract_error(
            "requires_gradient must be bool",
            field="requires_gradient",
            expected="bool",
            actual=requires_gradient,
        )
    gates = tuple(gate_history_indices)
    if (
        not gates
        or any(type(index) is not int for index in gates)
        or any(index < 1 or index > n_steps for index in gates)
        or any(right <= left for left, right in zip(gates, gates[1:], strict=False))
    ):
        _contract_error(
            "gate history indices must be unique, ascending, and within the recurrence",
            field="gate_history_indices",
            expected=f"ascending exact integers in [1, {n_steps}]",
            actual=list(gates),
        )
    shape = tuple(observation_shape)
    if not shape or any(type(size) is not int or size <= 0 for size in shape):
        _contract_error(
            "observation shape must contain positive exact dimensions",
            field="observation_shape",
            expected="non-empty positive integer shape",
            actual=list(shape),
        )
    if not isinstance(observation_dtype, torch.dtype):
        _contract_error(
            "observation dtype must be a torch dtype",
            field="observation_dtype",
            expected="torch.dtype",
            actual=observation_dtype,
        )
    if policy.mode == "checkpoint_recompute" and not requires_gradient:
        raise EMCapabilityError(
            "checkpoint recomputation requires a differentiable TEM request",
            object_name="prepare_recording",
            field="requires_gradient",
            expected=True,
            actual=False,
            details={
                "field": "requires_gradient",
                "recording_mode": policy.mode,
                "remediation": "enable gradients or select output_only",
            },
            hint="use checkpoint_recompute only for reverse-mode execution",
        )

    state_bytes = _checked_mul(state.numel(), state.element_size(), field="state_bytes")
    observation_elements = math.prod(shape)
    observation_bytes = _checked_mul(
        observation_elements,
        observation_dtype.itemsize,
        field="observation_bytes",
    )
    n_gates = len(gates)
    checkpoint_count = 0
    retained_state_count = 0
    recomputed_step_count = 0
    if policy.mode == "gate_states":
        retained_state_count = n_gates
        recording_bytes = _checked_mul(n_gates, state_bytes, field="recording_bytes")
    else:
        gate_bytes = _checked_mul(n_gates, observation_bytes, field="gate_output_bytes")
        if policy.mode == "checkpoint_recompute":
            interval = cast(int, policy.checkpoint_interval)
            checkpoint_count = max(0, math.ceil(n_steps / interval) - 1)
            checkpoint_bytes = _checked_mul(
                checkpoint_count,
                state_bytes,
                field="checkpoint_bytes",
            )
            recording_bytes = _checked_add(
                gate_bytes,
                checkpoint_bytes,
                field="recording_bytes",
            )
            recomputed_step_count = n_steps
        else:
            recording_bytes = gate_bytes
    if policy.budget_bytes is not None and recording_bytes > policy.budget_bytes:
        raise EMResourceError(
            "TEM recording exceeds the explicit byte budget",
            object_name="prepare_recording",
            field="budget_bytes",
            expected=f">= {recording_bytes}",
            actual=policy.budget_bytes,
            details={
                "budget_bytes": policy.budget_bytes,
                "recording_mode": policy.mode,
                "required_bytes": recording_bytes,
                "remediation": "increase budget_bytes or select a leaner recording mode",
            },
            hint="increase the explicit recording budget",
        )
    diagnostics = RecordingDiagnostics(
        n_steps=n_steps,
        n_gates=n_gates,
        live_state_count_max=0,
        retained_state_count=retained_state_count,
        checkpoint_count=checkpoint_count,
        recomputed_step_count=recomputed_step_count,
        recording_bytes=recording_bytes,
    )
    return RecordingPlan(
        policy=policy,
        gate_history_indices=gates,
        state_shape=tuple(state.shape),
        state_dtype=state.dtype,
        state_device=state.device,
        observation_shape=shape,
        observation_dtype=observation_dtype,
        diagnostics=diagnostics,
    )


def _validate_state(plan: RecordingPlan, state: torch.Tensor) -> None:
    if (
        not isinstance(state, torch.Tensor)
        or tuple(state.shape) != plan.state_shape
        or state.dtype != plan.state_dtype
        or state.device != plan.state_device
    ):
        _contract_error(
            "TEM recurrence changed state shape, dtype, or device",
            field="state",
            expected={
                "shape": list(plan.state_shape),
                "dtype": str(plan.state_dtype),
                "device": str(plan.state_device),
            },
            actual=(
                {
                    "shape": list(state.shape),
                    "dtype": str(state.dtype),
                    "device": str(state.device),
                }
                if isinstance(state, torch.Tensor)
                else type(state).__qualname__
            ),
        )


def execute_recorded_recurrence(
    plan: RecordingPlan,
    initial_state: torch.Tensor,
    *,
    step: StepFunction,
    observe: ObservationFunction,
    differentiable_inputs: tuple[torch.Tensor, ...] = (),
) -> tuple[tuple[torch.Tensor, ...], RecordingDiagnostics]:
    """Run one recurrence with only gate outputs or configured checkpoints retained."""
    if type(plan) is not RecordingPlan:
        _contract_error(
            "recording execution requires an exact RecordingPlan",
            field="plan",
            expected="RecordingPlan",
            actual=plan,
        )
    _validate_state(plan, initial_state)
    if not callable(step) or not callable(observe):
        _contract_error(
            "recording callbacks must be callable",
            field="callbacks",
            expected="callable step and observe",
            actual="non-callable callback",
        )
    if any(not isinstance(value, torch.Tensor) for value in differentiable_inputs):
        _contract_error(
            "differentiable recurrence inputs must be tensors",
            field="differentiable_inputs",
            expected="tuple[torch.Tensor, ...]",
            actual="non-tensor input",
        )

    def record_observation(state: torch.Tensor) -> torch.Tensor:
        value = observe(state)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != plan.observation_shape
            or value.dtype != plan.observation_dtype
            or value.device != plan.state_device
        ):
            _contract_error(
                "TEM gate observation differs from its recording preflight",
                field="observation",
                expected={
                    "shape": list(plan.observation_shape),
                    "dtype": str(plan.observation_dtype),
                    "device": str(plan.state_device),
                },
                actual=(
                    {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "device": str(value.device),
                    }
                    if isinstance(value, torch.Tensor)
                    else type(value).__qualname__
                ),
            )
        return value

    gates = plan.gate_history_indices
    gate_set = frozenset(gates)
    if plan.policy.mode != "checkpoint_recompute":
        current = initial_state
        gate_states: dict[int, torch.Tensor] = {}
        gate_outputs: dict[int, torch.Tensor] = {}
        for index in range(plan.diagnostics.n_steps):
            next_state = step(index, current, *differentiable_inputs)
            plan.diagnostics.live_state_count_max = max(
                plan.diagnostics.live_state_count_max,
                2,
            )
            _validate_state(plan, next_state)
            current = next_state
            history_index = index + 1
            if history_index in gate_set:
                if plan.policy.mode == "gate_states":
                    gate_states[history_index] = current
                else:
                    gate_outputs[history_index] = record_observation(current)
        if plan.policy.mode == "gate_states":
            gate_outputs = {index: record_observation(gate_states[index]) for index in gates}
        return tuple(gate_outputs[index] for index in gates), plan.diagnostics

    if not differentiable_inputs or not any(value.requires_grad for value in differentiable_inputs):
        raise EMCapabilityError(
            "checkpoint recomputation requires at least one trainable tensor input",
            object_name="execute_recorded_recurrence",
            field="differentiable_inputs",
            expected="at least one requires_grad tensor",
            actual="no trainable tensor",
            details={
                "field": "differentiable_inputs",
                "recording_mode": plan.policy.mode,
                "remediation": "pass the material tensors that drive the recurrence",
            },
            hint="thread differentiable material tensors through the checkpoint",
        )

    interval = cast(int, plan.policy.checkpoint_interval)
    current = initial_state
    gate_outputs = {}

    def make_segment(
        start: int,
        stop: int,
        segment_gates: tuple[int, ...],
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        def run_segment(state: torch.Tensor, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
            outputs: list[torch.Tensor] = []
            current_state = state
            for index in range(start, stop):
                current_state = step(index, current_state, *inputs)
                plan.diagnostics.live_state_count_max = max(
                    plan.diagnostics.live_state_count_max,
                    2,
                )
                if index + 1 in segment_gates:
                    outputs.append(record_observation(current_state))
            return (current_state, *outputs)

        return run_segment

    for start in range(0, plan.diagnostics.n_steps, interval):
        stop = min(start + interval, plan.diagnostics.n_steps)
        segment_gates = tuple(index for index in gates if start < index <= stop)
        segment = make_segment(start, stop, segment_gates)
        result = checkpoint(
            segment,
            current,
            *differentiable_inputs,
            use_reentrant=False,
        )
        current = result[0]
        _validate_state(plan, current)
        for history_index, value in zip(segment_gates, result[1:], strict=True):
            gate_outputs[history_index] = value
    return tuple(gate_outputs[index] for index in gates), plan.diagnostics


__all__ = [
    "RecordingDiagnostics",
    "RecordingPlan",
    "RecordingPolicy",
    "execute_recorded_recurrence",
    "prepare_recording",
]
