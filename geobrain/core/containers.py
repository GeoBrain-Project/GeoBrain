"""
Frozen state containers passed between operators.

- :class:`ModelState`: parameters being inverted or transformed (vp, rho, σ, ...).
- :class:`ForwardOutput`: what a :class:`ForwardOperator` produces (traces, gz, ...).

Both are physics-agnostic: fields are dict-of-tensors, with no specific schema
enforced at the core layer. The typed per-call configuration that travels
alongside them (:class:`~geobrain.core.context.ForwardContext` and its
sub-contexts) lives in the peer module :mod:`geobrain.core.context`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from .validation import freeze_mapping, validate_mapping_key, validate_tensor_mapping
from .errors import MissingFieldError


@dataclass(frozen=True)
class ModelState:
    """
    Parameter container.

    Attributes:
        tensors: Tensor name → :class:`torch.Tensor`. Trainable tensors have
            ``requires_grad=True``.
        metadata: Free-form metadata (e.g. units, spatial extent). Not used by
            autograd.
    """

    tensors: Mapping[str, torch.Tensor] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate tensor types and freeze ``tensors`` / ``metadata`` read-only."""
        validate_tensor_mapping("ModelState", "tensors", self.tensors,
                                message="ModelState tensors must be torch.Tensor")
        for name in self.metadata:
            validate_mapping_key("ModelState", "metadata", name)
        object.__setattr__(self, "tensors", freeze_mapping(self.tensors))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def fetch(self, *names: str) -> tuple[torch.Tensor, ...]:
        """
        Return the tensors for the given names, in order.

        Args:
            *names: Tensor names to fetch.

        Returns:
            A tuple of tensors, one per requested name.

        Raises:
            MissingFieldError: If any requested tensor is absent.
        """
        out = []
        for name in names:
            if name not in self.tensors:
                raise MissingFieldError(
                    "ModelState missing required field",
                    object_name="ModelState",
                    field=name,
                    expected=f"present in {sorted(self.tensors)}",
                )
            out.append(self.tensors[name])
        return tuple(out)

    def with_tensors(self, **updates: torch.Tensor) -> ModelState:
        """
        Return a new :class:`ModelState` with tensors added or overwritten.

        Args:
            **updates: Tensor name → tensor to add or replace.

        Returns:
            A new state; metadata is carried over unchanged.
        """
        new = dict(self.tensors)
        new.update(updates)
        return ModelState(tensors=new, metadata=dict(self.metadata))


@dataclass(frozen=True)
class ForwardOutput:
    """
    Forward-model output.

    Attributes:
        data: Receiver-channel name → tensor (e.g. ``"pressure"``, ``"gz"``,
            ``"ex"``).
        fields: Optional snapshot fields (e.g. wavefield) for diagnostics or
            regularization.
        metadata: Free-form (e.g. time axis, frequency axis, units).
    """

    data: Mapping[str, torch.Tensor] = field(default_factory=dict)
    fields: Mapping[str, torch.Tensor] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate tensor types and freeze ``data`` / ``fields`` / ``metadata``."""
        validate_tensor_mapping("ForwardOutput", "data", self.data,
                                message="ForwardOutput.data entries must be torch.Tensor")
        validate_tensor_mapping("ForwardOutput", "fields", self.fields,
                                message="ForwardOutput.fields entries must be torch.Tensor")
        for name in self.metadata:
            validate_mapping_key("ForwardOutput", "metadata", name)
        object.__setattr__(self, "data", freeze_mapping(self.data))
        object.__setattr__(self, "fields", freeze_mapping(self.fields))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def fetch(self, *names: str) -> tuple[torch.Tensor, ...]:
        """
        Return the data channels for the given names, in order.

        Args:
            *names: Channel names to fetch.

        Returns:
            A tuple of tensors, one per requested channel.

        Raises:
            MissingFieldError: If any requested channel is absent.
        """
        out = []
        for name in names:
            if name not in self.data:
                raise MissingFieldError(
                    "ForwardOutput missing required data channel",
                    object_name="ForwardOutput",
                    field=name,
                    expected=f"present in {sorted(self.data)}",
                )
            out.append(self.data[name])
        return tuple(out)

