"""
Flattened parameter vector to :class:`~geobrain.core.ModelState`.

``Parameterization`` is the small bridge between optimizers that prefer a single
vector and GeoBrain operators that consume named fields.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from ..core import FieldShapeError, GeoBrainError, ModelState


class _TensorTransform(Protocol):
    """Structural tensor-to-tensor transform accepted by a parameter block."""

    def __call__(self, value: torch.Tensor, /) -> torch.Tensor: ...


if TYPE_CHECKING:
    class _ModuleBase:
        def __init__(self) -> None: ...

        def forward(self, theta: torch.Tensor) -> ModelState: ...

        def __call__(self, theta: torch.Tensor) -> ModelState: ...

else:
    _ModuleBase = torch.nn.Module


@dataclass(frozen=True)
class ParameterBlock:
    """One contiguous block in a flattened model vector.

    Args:
        name: Internal block name, unique inside one parameterization.
        shape: Tensor shape reconstructed from the flat vector.
        output: Field name written into ``ModelState.tensors``.
        transform: Optional callable or ``nn.Module`` applied after reshape.
    """

    name: str
    shape: tuple[int, ...]
    output: str
    transform: _TensorTransform | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise GeoBrainError(
                "ParameterBlock name must be a non-empty string",
                object_name="ParameterBlock",
                field="name",
                expected="non-empty str",
                actual=self.name,
            )
        if not isinstance(self.output, str) or not self.output:
            raise GeoBrainError(
                "ParameterBlock output must be a non-empty string",
                object_name="ParameterBlock",
                field="output",
                expected="non-empty str",
                actual=self.output,
            )
        if not isinstance(self.shape, tuple):
            raise GeoBrainError(
                "ParameterBlock shape must be a tuple",
                object_name="ParameterBlock",
                field="shape",
                expected=tuple,
                actual=type(self.shape),
            )
        for dim in self.shape:
            if not isinstance(dim, int) or dim <= 0:
                raise GeoBrainError(
                    "ParameterBlock shape dimensions must be positive integers",
                    object_name="ParameterBlock",
                    field="shape",
                    expected="positive int dimensions",
                    actual=self.shape,
                )
        if self.transform is not None and not callable(self.transform):
            raise GeoBrainError(
                "ParameterBlock transform must be callable",
                object_name="ParameterBlock",
                field="transform",
                expected="callable or None",
                actual=type(self.transform),
            )

    @property
    def size(self) -> int:
        """Number of scalar entries consumed by this block."""
        if self.shape == ():
            return 1
        size = 1
        for dim in self.shape:
            size *= dim
        return size


class Parameterization(_ModuleBase):
    """Map a flat tensor into a named :class:`ModelState`."""

    blocks: tuple[ParameterBlock, ...]
    module_transforms: Mapping[str, _TensorTransform]

    def __init__(self, blocks: Iterable[ParameterBlock]) -> None:
        super().__init__()
        self.blocks = tuple(blocks)
        if not self.blocks:
            raise GeoBrainError(
                "Parameterization requires at least one block",
                object_name="Parameterization",
                field="blocks",
                expected="non-empty iterable",
            )

        names = [block.name for block in self.blocks]
        if len(set(names)) != len(names):
            raise GeoBrainError(
                "ParameterBlock names must be unique",
                object_name="Parameterization",
                field="blocks",
                expected="unique block names",
                actual=names,
            )

        outputs = [block.output for block in self.blocks]
        if len(set(outputs)) != len(outputs):
            raise GeoBrainError(
                "ParameterBlock outputs must be unique",
                object_name="Parameterization",
                field="blocks",
                expected="unique output names",
                actual=outputs,
            )

        self.module_transforms = torch.nn.ModuleDict(
            {
                block.name: block.transform
                for block in self.blocks
                if isinstance(block.transform, torch.nn.Module)
            }
        )

    @property
    def size(self) -> int:
        """Total number of scalar entries expected in the flat vector."""
        return sum(block.size for block in self.blocks)

    def forward(self, theta: torch.Tensor) -> ModelState:
        """
        Unpack a flat parameter vector into a named :class:`ModelState`.

        Args:
            theta: 1-D tensor of length :attr:`size`. Sliced block by block,
                reshaped, and passed through each block's optional transform.

        Returns:
            A :class:`ModelState` keyed by each block's ``output`` name.

        Raises:
            FieldShapeError: If ``theta`` is not 1-D or has the wrong length.
            GeoBrainError: If a block transform returns a non-tensor.
        """
        if not isinstance(theta, torch.Tensor):
            raise GeoBrainError(
                "Parameterization.forward expects a torch.Tensor",
                object_name="Parameterization",
                field="theta",
                expected=torch.Tensor,
                actual=type(theta),
            )
        if theta.ndim != 1 or theta.numel() != self.size:
            raise FieldShapeError(
                "Parameter vector has wrong shape",
                object_name="Parameterization",
                field="theta",
                expected=(self.size,),
                actual=tuple(theta.shape),
            )

        tensors: dict[str, torch.Tensor] = {}
        start = 0
        for block in self.blocks:
            stop = start + block.size
            value = theta[start:stop].reshape(block.shape)
            transform: _TensorTransform | None
            if block.name in self.module_transforms:
                transform = self.module_transforms[block.name]
            else:
                transform = block.transform
            if transform is not None:
                value = transform(value)
                if not isinstance(value, torch.Tensor):
                    raise GeoBrainError(
                        "ParameterBlock transform must return a torch.Tensor",
                        object_name="Parameterization",
                        field=block.name,
                        expected=torch.Tensor,
                        actual=type(value),
                    )
            tensors[block.output] = value
            start = stop

        return ModelState(tensors=tensors)
