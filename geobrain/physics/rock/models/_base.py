"""
Abstract base classes for rock-physics component / composite models,
plus the generic :class:`RockPhysicsTransform` PropertyTransform wrapper.

Three-level hierarchy, slightly trimmed for the new GeoBrain architecture:

- :class:`BaseModel`: ``nn.Module`` + ``ABC`` with metadata /
  parameter-range storage. Subclasses must implement ``forward``.
- :class:`ComponentModel`: single-purpose rock-physics equation
  (Voigt average, Gassmann substitution, …). Adds the optional
  ``compute_dry_rock`` workflow-integration hook.
- :class:`CompositeModel`: orchestrates multiple ``ComponentModel``
  instances (mineral mix → dry rock → fluid sub) into one chain.
- :class:`RockPhysicsTransform`: adapts any ``ComponentModel`` (or
  callable) to the ``ModelState`` / ``ForwardContext`` contract.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.nn as nn

from ....core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    GeoBrainError,
    ModelState,
    PropertyTransform,
)


class BaseModel(nn.Module, ABC):
    """
    Abstract base for all rock-physics models.

    Inherits from :class:`torch.nn.Module` (autograd / ``.to(device)`` /
    ``.state_dict()``) and :class:`abc.ABC` (forces subclasses to
    implement :meth:`forward`).
    """

    def __init__(self) -> None:
        super().__init__()
        self._metadata: dict[str, Any] = {}
        self._param_ranges: dict[str, tuple[float, float]] = {}

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Compute model output. Must be overridden by subclasses."""
        raise NotImplementedError

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def set_parameter_range(self, name: str, range_tuple: tuple[float, float]) -> None:
        self._param_ranges[name] = range_tuple

    def get_parameter_range(self, name: str) -> tuple[float, float] | None:
        return self._param_ranges.get(name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ComponentModel(BaseModel):
    """
    Single-equation rock-physics model.

    Concrete subclasses (Voigt, Gassmann, Hertz-Mindlin, etc.) override
    ``forward`` with their natural physical signature. Implementations
    that want to participate in :class:`RockPhysicsWorkflow` orchestration
    additionally implement :meth:`compute_dry_rock` (or the
    category-specific hook on their :mod:`~geobrain.physics.rock.models._categories` base).
    """

    def compute_dry_rock(self, K0, G0, phi, **params):
        """
        Default workflow hook: invoke ``forward(K0, G0, phi)``.

        Override when the model's natural ``forward`` signature differs
        from ``(K0, G0, phi)`` and you want it usable from
        :class:`RockPhysicsWorkflow`.
        """
        return self.forward(K0, G0, phi)


class CompositeModel(BaseModel):
    """
    Composite of multiple :class:`ComponentModel` instances.

    Holds named children in :attr:`_components` for introspection /
    serialization; subclasses (RockPhysicsWorkflow, Screening, RPT)
    define the orchestration logic in :meth:`forward`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._components: dict[str, ComponentModel] = {}

    def add_component(self, name: str, component: ComponentModel) -> None:
        self._components[name] = component
        setattr(self, name, component)

    def get_component(self, name: str) -> ComponentModel | None:
        return self._components.get(name)

    def list_components(self) -> list[str]:
        return list(self._components.keys())


# ---------------------------------------------------------------------------
# Generic PropertyTransform wrapper
# ---------------------------------------------------------------------------


class RockPhysicsTransform(PropertyTransform):
    """
    Wrap a :class:`ComponentModel` (or callable) as a ``PropertyTransform``.

    :class:`RockPhysicsTransform` adapts any pure :class:`ComponentModel`
    (or any plain callable returning tensors) to the GeoBrain
    :class:`ModelState` / :class:`ForwardContext` contract by mapping named
    fields in / out.

    When per-operator behaviour is straightforward; pull some named tensors
    out of state, run a deterministic computation, push named tensors back --
    this generic wrapper is enough. For operators whose I/O depends on
    ``ctx.extras`` keys, fluid-in-state mode flags, or non-trivial per-shape
    validation, prefer a hand-written :class:`PropertyTransform` subclass
    that *delegates* its math to a :class:`ComponentModel` (the pattern used
    by Gassmann, Wood, Brie, etc. in the same package).

    Args:
        model: Pure :class:`ComponentModel` (typically from
            ``geobrain.physics.rock.models``) or any callable that takes
            named tensor kwargs and returns either a single tensor or a
            tuple matching ``output_fields``.
        input_fields: ``ModelState`` keys to pull and pass positionally
            to the model in this order.
        output_fields: ``ModelState`` keys to assign the model output to.
            If the model returns a single tensor and ``len(output_fields) == 1``,
            it is stored under that key. If the model returns a tuple,
            its length must match ``len(output_fields)``.
    """

    def __init__(
        self,
        model: ComponentModel | Callable,
        *,
        input_fields: tuple[str, ...],
        output_fields: tuple[str, ...],
    ) -> None:
        super().__init__()
        if not callable(model):
            raise GeoBrainError(
                "RockPhysicsTransform model must be callable",
                object_name="RockPhysicsTransform",
                field="model",
                expected="ComponentModel or callable",
                actual=type(model).__name__,
            )
        if not input_fields:
            raise GeoBrainError(
                "RockPhysicsTransform needs at least one input field",
                object_name="RockPhysicsTransform",
                field="input_fields",
                expected="non-empty tuple",
                actual=input_fields,
            )
        if not output_fields:
            raise GeoBrainError(
                "RockPhysicsTransform needs at least one output field",
                object_name="RockPhysicsTransform",
                field="output_fields",
                expected="non-empty tuple",
                actual=output_fields,
            )

        self.model = model
        if isinstance(model, torch.nn.Module):
            # Register so .to(device) / .state_dict() / params propagate.
            self.add_module("_rock_model", model)
        self.input_fields = tuple(input_fields)
        self.output_fields = tuple(output_fields)
        self.differentiability = DifferentiabilitySpec(
            level=DifferentiabilityLevel.FULL_AUTOGRAD,
            trainable_inputs=tuple(self.input_fields),
            output_keys=tuple(self.output_fields),
        )

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        inputs = state.fetch(*self.input_fields)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        result = self.model(*inputs)

        if isinstance(result, torch.Tensor):
            if len(self.output_fields) != 1:
                raise GeoBrainError(
                    "Model returned a single tensor but operator expects multiple outputs",
                    object_name="RockPhysicsTransform",
                    field="output_fields",
                    expected=f"1 tensor for {self.output_fields!r}",
                    actual=f"{len(self.output_fields)} fields",
                )
            return state.with_tensors(**{self.output_fields[0]: result})

        if not isinstance(result, tuple):
            raise GeoBrainError(
                "Model must return a tensor or tuple of tensors",
                object_name="RockPhysicsTransform",
                field="model output",
                expected="Tensor or tuple",
                actual=type(result).__name__,
            )
        if len(result) != len(self.output_fields):
            raise GeoBrainError(
                "Model output arity does not match output_fields",
                object_name="RockPhysicsTransform",
                field="model output",
                expected=len(self.output_fields),
                actual=len(result),
            )
        return state.with_tensors(**dict(zip(self.output_fields, result)))


__all__ = ["BaseModel", "ComponentModel", "CompositeModel", "RockPhysicsTransform"]
