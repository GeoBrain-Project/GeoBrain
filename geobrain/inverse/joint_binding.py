"""Compile per-term field, output, mesh, and context wiring for joint inversion.

The public :class:`JointForward` configuration describes explicit wiring for
one joint-inversion term. :func:`compile_joint_forward` validates that wiring
against the operator contract and returns an immutable
:class:`CompiledJointForward` consumed by :class:`geobrain.inverse.JointProblem`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

import torch

from ..core.composition import OperatorChain, _terminates_in_observation
from ..core.containers import ForwardOutput, ModelState
from ..core.context import ForwardContext
from ..core.errors import GeoBrainError
from ..core.operator import Operator

__all__ = [
    "CompiledJointForward",
    "JointForward",
    "compile_joint_forward",
]


class _EarthModelLike(Protocol):
    def with_values(self, **updates: torch.Tensor) -> _EarthModelLike: ...

    def resolve(self, *names: str) -> ModelState: ...

    def trainables(self) -> Mapping[str, torch.Tensor]: ...


class JointForward:
    """Describe explicit wiring for one term in a joint inverse problem.

    A bare operator uses its external trainable-input names directly. This
    wrapper is the opt-in path for model-field renaming, output selection,
    differentiable mesh adaptation, or a term-local context overlay.

    Args:
        op: Observation-terminating operator for the term.
        fields: Optional mapping from resolved model-field names to operator
            input names. Values must match the operator's external inputs
            exactly.
        output: Optional selected output channel. It is required when the
            operator declares more than one output.
        field_to_mesh: Optional differentiable tensor adapter applied to every
            resolved field before the operator runs.
        ctx_overrides: Optional term-local context overlay. Device movement is
            never inferred here; use an explicit ``field_to_mesh`` adapter when
            a term intentionally changes tensor device or representation.

    Raises:
        GeoBrainError: If any binding component violates its typed contract.
    """

    __slots__ = ("op", "fields", "output", "field_to_mesh", "ctx_overrides")

    op: Operator
    fields: Mapping[str, str] | None
    output: str | None
    field_to_mesh: Callable[[torch.Tensor], torch.Tensor] | None
    ctx_overrides: Mapping[str, Any] | None

    def __init__(
        self,
        op: Operator,
        *,
        fields: Mapping[str, str] | None = None,
        output: str | None = None,
        field_to_mesh: Callable[[torch.Tensor], torch.Tensor] | None = None,
        ctx_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(op, Operator):
            raise GeoBrainError(
                "JointForward op= must be an Operator",
                object_name="JointForward",
                field="op",
                expected=Operator,
                actual=type(op),
            )
        if fields is not None:
            if not isinstance(fields, Mapping) or not fields:
                raise GeoBrainError(
                    "JointForward fields= must be a non-empty mapping when given",
                    object_name="JointForward",
                    field="fields",
                    expected="non-empty Mapping[str, str]",
                    actual=fields,
                )
            for model_name, input_name in fields.items():
                if (
                    not isinstance(model_name, str)
                    or not model_name
                    or not isinstance(input_name, str)
                    or not input_name
                ):
                    raise GeoBrainError(
                        "JointForward fields= keys/values must be non-empty strings",
                        object_name="JointForward",
                        field="fields",
                        actual=(model_name, input_name),
                    )
        if output is not None and (not isinstance(output, str) or not output):
            raise GeoBrainError(
                "JointForward output= must be a non-empty string or None",
                object_name="JointForward",
                field="output",
                expected="non-empty str or None",
                actual=output,
            )
        if field_to_mesh is not None and not callable(field_to_mesh):
            raise GeoBrainError(
                "JointForward field_to_mesh= must be callable",
                object_name="JointForward",
                field="field_to_mesh",
                expected="callable(Tensor) -> Tensor",
                actual=type(field_to_mesh),
            )
        if ctx_overrides is not None and not isinstance(ctx_overrides, Mapping):
            raise GeoBrainError(
                "JointForward ctx_overrides= must be a Mapping or None",
                object_name="JointForward",
                field="ctx_overrides",
                expected="Mapping or None",
                actual=type(ctx_overrides),
            )
        if ctx_overrides is not None and "device" in ctx_overrides:
            raise GeoBrainError(
                "JointForward does not move input tensors from ctx_overrides",
                object_name="JointForward",
                field="ctx_overrides['device']",
                expected=(
                    "an explicit field_to_mesh adapter that performs the "
                    "intentional device transfer"
                ),
                actual=ctx_overrides["device"],
            )

        self.op = op
        self.fields = MappingProxyType(dict(fields)) if fields is not None else None
        self.output = output
        self.field_to_mesh = field_to_mesh
        self.ctx_overrides = (
            MappingProxyType(dict(ctx_overrides)) if ctx_overrides is not None else None
        )

    def __repr__(self) -> str:
        """Return a concise representation of the explicit binding."""
        fields = dict(self.fields) if self.fields is not None else None
        overrides = dict(self.ctx_overrides) if self.ctx_overrides is not None else None
        return (
            f"JointForward({type(self.op).__name__}, fields={fields}, "
            f"output={self.output!r}, "
            f"field_to_mesh={'set' if self.field_to_mesh else None}, "
            f"ctx_overrides={overrides})"
        )


@dataclass(frozen=True)
class CompiledJointForward:
    """Validated immutable wiring for one named joint-forward term.

    Args:
        name: Joint-objective term name.
        op: Validated observation-terminating operator.
        model_fields: Exact resolved model fields read by the term.
        input_names: Mapping from resolved model names to operator input names.
        output_name: Selected operator output channel.
        field_to_mesh: Optional differentiable per-field mesh adapter.
        ctx_overrides: Immutable term-local context overlay.
    """

    name: str
    op: Operator
    model_fields: tuple[str, ...]
    input_names: Mapping[str, str]
    output_name: str
    field_to_mesh: Callable[[torch.Tensor], torch.Tensor] | None
    ctx_overrides: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Defensively freeze both mapping-valued wiring components."""
        object.__setattr__(
            self,
            "input_names",
            MappingProxyType(dict(self.input_names)),
        )
        object.__setattr__(
            self,
            "ctx_overrides",
            MappingProxyType(dict(self.ctx_overrides)),
        )

    def run(
        self,
        resolved_tensors: Mapping[str, torch.Tensor],
        base_context: ForwardContext,
    ) -> ForwardOutput:
        """Build the exact term input and execute its observation operator.

        Args:
            resolved_tensors: Shared-model tensors containing
                :attr:`model_fields`.
            base_context: Per-call context before this binding's overlay.

        Returns:
            The full forward output produced by :attr:`op`.

        Raises:
            GeoBrainError: If an observation-terminating operator violates its
                declared output type at runtime.
        """
        context = _forward_context(self, base_context)
        input_state = _build_input_state(self, resolved_tensors)
        result = self.op(input_state, context)
        if not isinstance(result, ForwardOutput):
            raise GeoBrainError(
                "compiled joint forward must return ForwardOutput",
                object_name="CompiledJointForward",
                field=f"forwards[{self.name!r}]",
                expected=ForwardOutput,
                actual=type(result),
            )
        return result


def _chain_walk_needed(op: Operator) -> tuple[str, ...]:
    members = op.operators if isinstance(op, OperatorChain) else (op,)
    produced: set[str] = set()
    seen: set[str] = set()
    needed: list[str] = []
    for member in members:
        specification = member.differentiability
        for name in specification.trainable_inputs:
            if name not in produced and name not in seen:
                needed.append(name)
                seen.add(name)
        produced.update(specification.output_keys)
    return tuple(needed)


def _forward_context(
    compiled: CompiledJointForward,
    base_context: ForwardContext,
) -> ForwardContext:
    if not compiled.ctx_overrides:
        return base_context
    return base_context.with_overrides(compiled.ctx_overrides)


def _build_input_state(
    compiled: CompiledJointForward,
    resolved_tensors: Mapping[str, torch.Tensor],
) -> ModelState:
    tensors: dict[str, torch.Tensor] = {}
    for model_name, input_name in compiled.input_names.items():
        tensor = resolved_tensors[model_name]
        if compiled.field_to_mesh is not None:
            tensor = compiled.field_to_mesh(tensor)
        tensors[input_name] = tensor
    return ModelState(tensors=tensors)


def _validate_model_reachability(
    model: _EarthModelLike,
    model_fields: tuple[str, ...],
) -> None:
    trainables = model.trainables()
    if not trainables:
        return
    leaves = {
        name: tensor.detach().clone().requires_grad_(True) for name, tensor in trainables.items()
    }
    resolved = model.with_values(**leaves).resolve(*model_fields)
    total: torch.Tensor | None = None
    for tensor in resolved.tensors.values():
        term = tensor.pow(2).sum()
        total = term if total is None else total + term
    if total is None or not total.requires_grad:
        orphaned = sorted(leaves)
    else:
        gradients = torch.autograd.grad(total, list(leaves.values()), allow_unused=True)
        orphaned = sorted(name for name, gradient in zip(leaves, gradients) if gradient is None)
    if orphaned:
        raise GeoBrainError(
            "EarthModel trainable(s) not reached by any JointProblem forward's resolved sub-DAG",
            object_name="JointProblem",
            field="model",
            expected="every model.trainables() name reachable from ≥1 forward",
            actual=orphaned,
        )


def compile_joint_forward(
    name: str,
    binding: Operator | JointForward,
) -> CompiledJointForward:
    """Validate and compile one named operator or explicit joint binding.

    Args:
        name: Non-empty joint-objective term name.
        binding: Bare operator for wire-by-name behavior, or
            :class:`JointForward` for explicit wiring.

    Returns:
        Immutable compiled wiring used for state construction and execution.

    Raises:
        GeoBrainError: If the name, operator, external fields, or selected
            output violates the joint-binding contract.
    """
    if not isinstance(name, str) or not name:
        raise GeoBrainError(
            "compiled joint-forward name must be a non-empty string",
            object_name="CompiledJointForward",
            field="name",
            expected="non-empty str",
            actual=name,
        )

    if isinstance(binding, JointForward):
        op = binding.op
        explicit_fields = binding.fields
        explicit_output = binding.output
        field_to_mesh = binding.field_to_mesh
        ctx_overrides = binding.ctx_overrides
    elif isinstance(binding, Operator):
        op = binding
        explicit_fields = None
        explicit_output = None
        field_to_mesh = None
        ctx_overrides = None
    else:
        raise GeoBrainError(
            "JointProblem forwards values must be an Operator or a JointForward wrapper",
            object_name="JointProblem",
            field=f"forwards[{name!r}]",
            expected="Operator or JointForward",
            actual=type(binding),
        )

    if not _terminates_in_observation(op):
        raise GeoBrainError(
            "JointProblem forward must be observation-terminating "
            "(a ForwardOperator, OperatorBundle, or OperatorChain ending in one)",
            object_name="JointProblem",
            field=f"forwards[{name!r}]",
            expected="observation-terminating Operator",
            actual=type(op).__name__,
        )

    default_fields = _chain_walk_needed(op)
    if explicit_fields is not None:
        input_names = dict(explicit_fields)
        explicit_inputs = tuple(input_names.values())
        if (
            len(explicit_inputs) != len(default_fields)
            or set(explicit_inputs) != set(default_fields)
        ):
            raise GeoBrainError(
                "JointForward fields= must rename exactly the operator's "
                "needed inputs one-to-one (trainable_inputs, chain "
                "member-walked), no missing, extra, or duplicate targets",
                object_name="JointProblem",
                field=f"forwards[{name!r}].fields",
                expected=sorted(default_fields),
                actual=sorted(explicit_inputs),
            )
        model_fields = tuple(input_names)
    else:
        input_names = {model_name: model_name for model_name in default_fields}
        model_fields = default_fields

    output_keys = op.differentiability.output_keys
    if explicit_output is not None:
        if explicit_output not in output_keys:
            raise GeoBrainError(
                "JointForward output= is not among the operator's declared output_keys",
                object_name="JointProblem",
                field=f"forwards[{name!r}].output",
                expected=sorted(output_keys),
                actual=explicit_output,
            )
        output_name = explicit_output
    else:
        if len(output_keys) != 1:
            raise GeoBrainError(
                "forward produces multiple output channels; wrap it in "
                "JointForward(op, output=<channel>) to select one",
                object_name="JointProblem",
                field=f"forwards[{name!r}]",
                expected=("exactly one output_keys entry, or an explicit JointForward(output=...)"),
                actual=sorted(output_keys),
            )
        output_name = output_keys[0]

    return CompiledJointForward(
        name=name,
        op=op,
        model_fields=model_fields,
        input_names=input_names,
        output_name=output_name,
        field_to_mesh=field_to_mesh,
        ctx_overrides=ctx_overrides or {},
    )
