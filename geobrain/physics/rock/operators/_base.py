"""Declarative Agent-ready facade plumbing for canonical Rock kernels.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Mapping

import torch

from geobrain.core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    ForwardOperator,
    ForwardOutput,
    ModelState,
)

from ..capabilities import RockCapabilityReport, RockUnsupportedCombination
from ..contracts import RockFieldSpec
from ..errors import RockContractError
from ..resources import RockResourceEstimate, estimate_elementwise_resources


@dataclass(frozen=True, slots=True)
class RockOperatorDeclaration:
    """Immutable execution, schema, capability, and provenance declaration."""

    model: str
    subfamily: str
    inputs: tuple[RockFieldSpec, ...]
    outputs: tuple[RockFieldSpec, ...]
    citation: str
    calibrated_domain: tuple[tuple[str, str], ...]
    workspace_tensors: int = 4
    saved_tensors: int = 4
    iterative: bool = False
    maturity: str = "production"

    @property
    def differentiability(self) -> DifferentiabilitySpec:
        """Return the exact facade differentiation declaration."""

        return DifferentiabilitySpec(
            level=DifferentiabilityLevel.FULL_AUTOGRAD,
            trainable_inputs=tuple(field.name for field in self.inputs),
            output_keys=tuple(field.name for field in self.outputs),
            input_units={field.name: field.unit for field in self.inputs},
        )

    def input_schema(self) -> Mapping[str, object]:
        """Return JSON Schema 2020-12 for the ModelState tensor mapping."""

        properties = {
            field.name: {
                "description": field.description,
                "unit": field.unit,
                "type": "tensor",
                "dtypes": ["float32", "float64"],
            }
            for field in self.inputs
        }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "schema_version": "geobrain.rock.input/1.0",
            "title": self.model,
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "model": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": properties,
                    "required": [field.name for field in self.inputs],
                }
            },
            "required": ["model"],
        }

    def capability_report(self) -> RockCapabilityReport:
        """Return the immutable public capability report."""

        return RockCapabilityReport(
            physics="rock",
            model=self.model,
            subfamily=self.subfamily,
            maturity="production" if self.maturity == "production" else "experimental",
            required_model_fields=self.inputs,
            output_fields=self.outputs,
            dtypes=("float32", "float64"),
            devices=("cpu",),
            broadcast=True,
            complex_outputs=(),
            differentiability=DifferentiabilityLevel.FULL_AUTOGRAD,
            differentiable_model_fields=tuple(field.name for field in self.inputs),
            iterative=self.iterative,
            resource_estimate_supported=True,
            citation=self.citation,
            calibrated_domain=self.calibrated_domain,
            unsupported=(
                RockUnsupportedCombination(
                    selection=(("device", "cuda"),),
                    reason="CPU evidence is complete; accelerator evidence is future work.",
                    remediation="Run the accelerator qualification benchmarks before advertising CUDA.",
                ),
            ),
        )


class RockForwardOperator(ForwardOperator):
    """Stable ModelState-to-ForwardOutput facade over one canonical kernel."""

    declaration: ClassVar[RockOperatorDeclaration]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Install the immutable declaration as the class-level core contract."""

        super().__init_subclass__(**kwargs)
        declaration = getattr(cls, "declaration", None)
        if isinstance(declaration, RockOperatorDeclaration):
            cls.differentiability = declaration.differentiability

    def __init__(self, *, budget_bytes: int | None = None) -> None:
        super().__init__()
        if budget_bytes is not None and (
            type(budget_bytes) is not int or budget_bytes <= 0
        ):
            raise RockContractError(
                "invalid Rock resource budget",
                object_name=type(self).__name__,
                field="budget_bytes",
                expected="positive integer or None",
                actual=budget_bytes,
            )
        self._budget_bytes = budget_bytes
        self.differentiability = self.declaration.differentiability

    @classmethod
    def capabilities(cls) -> RockCapabilityReport:
        """Return the class capability report without constructing the operator."""

        return cls.declaration.capability_report()

    @classmethod
    def input_schema(cls) -> Mapping[str, object]:
        """Return the class Agent input schema."""

        return cls.declaration.input_schema()

    def estimate_resources(
        self,
        state: ModelState,
        *,
        autograd_enabled: bool,
        budget_bytes: int | None = None,
    ) -> RockResourceEstimate:
        """Preflight broadcast working bytes without scientific allocation."""

        # fetch() raises a structured MissingFieldError on absent inputs, so a
        # direct preflight call fails as loudly as the forward path does.
        tensors = state.fetch(*(field.name for field in self.declaration.inputs))
        if not tensors:
            raise RockContractError(
                "Rock facade has no declared inputs",
                object_name=type(self).__name__,
                field="declaration.inputs",
                expected="at least one field",
                actual=0,
            )
        shape = torch.broadcast_shapes(*(tensor.shape for tensor in tensors))
        estimate = estimate_elementwise_resources(
            broadcast_shape=shape,
            dtype=tensors[0].dtype,
            input_tensors=len(tensors),
            output_tensors=len(self.declaration.outputs),
            workspace_tensors=self.declaration.workspace_tensors,
            saved_tensors=self.declaration.saved_tensors if autograd_enabled else 0,
        )
        selected_budget = self._budget_bytes if budget_bytes is None else budget_bytes
        if selected_budget is not None:
            estimate.require_budget(selected_budget, object_name=type(self).__name__)
        return estimate

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        estimate = self.estimate_resources(
            state,
            autograd_enabled=torch.is_grad_enabled(),
        )
        data, diagnostics = self._evaluate(state)
        metadata = {
            "schema": "geobrain.rock.output/1.0",
            "version": "0.2.0",
            "model": self.declaration.model,
            "subfamily": self.declaration.subfamily,
            "units": {field.name: field.unit for field in self.declaration.outputs},
            "citation": self.declaration.citation,
            "calibrated_domain": [list(row) for row in self.declaration.calibrated_domain],
            "resources": estimate.to_dict(),
            "diagnostics": diagnostics,
        }
        return ForwardOutput(data=data, metadata=metadata)

    @abstractmethod
    def _evaluate(
        self,
        state: ModelState,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        """Fetch state fields, invoke one canonical kernel, and package outputs."""


def field(name: str, unit: str, description: str) -> RockFieldSpec:
    """Construct one concise SI field declaration."""

    return RockFieldSpec(name=name, unit=unit, description=description)


__all__ = [
    "RockForwardOperator",
    "RockOperatorDeclaration",
    "field",
]
