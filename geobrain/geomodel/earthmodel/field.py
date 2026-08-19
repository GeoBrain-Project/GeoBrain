"""``Field``: a typed model-space unknown for :class:`~geobrain.geomodel.earthmodel.EarthModel`.

A pure descriptor: it carries no live parameter state of its own (the
trainable leaf tensors live on the owning ``EarthModel``, so a functional
``with_values``-produced model can share the SAME ``Field`` objects while
holding DIFFERENT leaf values). Constructing a ``Field`` only validates and
normalizes its arguments.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from geobrain.core.errors import GeoBrainError
from geobrain.core.transforms import AffineSigmoid, InvertibleTransform

__all__ = ["Field"]

_Generator = nn.Module | tuple[nn.Module, torch.Tensor]


class Field:
    """A single named unknown in an :class:`~geobrain.geomodel.earthmodel.EarthModel`.

    Exactly one of ``init=``/``generator=`` must be given:

    - ``init=``: a mesh-shaped :class:`torch.Tensor`. Its PHYSICAL value seeds
      the model; when a ``transform=``/``bounds=`` is also given, the stored
      trainable leaf is ``transform.inverse(init)`` (unconstrained space);
      ``init`` itself is always physical-space.
    - ``generator=``: an ``nn.Module`` (or ``(nn.Module, fixed_input)`` tuple
      for a network that needs a fixed input) whose weights become the
      trainables; the network's own ``forward()`` must return a mesh-shaped
      tensor (see ``EarthModel``'s generator-seam docs for the
      ``torch.func.functional_call`` mechanics).

    Args:
        init: Mesh-shaped physical-space tensor. Mutually exclusive with
            ``generator=``.
        generator: An ``nn.Module`` producing the field, or
            ``(nn.Module, fixed_input)`` when the network needs a fixed
            input tensor every forward. Mutually exclusive with ``init=``
            and with ``bounds=``/``transform=`` (a generator is responsible
            for its own output range).
        transform: A custom :class:`~geobrain.core.transforms.InvertibleTransform`,
            or a list of them applied as a chain (``forward`` left-to-right,
            ``inverse`` right-to-left). Mutually exclusive with ``bounds=``.
        bounds: ``(lo, hi)`` SUGAR, internally constructs
            ``AffineSigmoid(lo, hi)`` as the (sole) transform. Mutually
            exclusive with ``transform=``.
        unit: Explicit SI unit string. ``None`` (the default) defers to
            :data:`geobrain.geomodel.earthmodel.field_specs.FIELD_SPECS`'s by-name default,
            resolved by the owning ``EarthModel`` (a bare ``Field`` does not
            know its own name).
    """

    __slots__ = ("init", "generator", "transforms", "unit")

    def __init__(
        self,
        init: torch.Tensor | None = None,
        *,
        generator: _Generator | None = None,
        transform: InvertibleTransform | Sequence[InvertibleTransform] | None = None,
        bounds: tuple[float, float] | None = None,
        unit: str | None = None,
    ) -> None:
        if (init is None) == (generator is None):
            raise GeoBrainError(
                "Field requires exactly one of init= or generator=",
                object_name="Field",
                field="init/generator",
                expected="exactly one of init=, generator= set",
                actual=(init is not None, generator is not None),
            )
        if bounds is not None and transform is not None:
            raise GeoBrainError(
                "Field bounds= and transform= are mutually exclusive "
                "(bounds= is sugar that constructs the transform internally)",
                object_name="Field",
                field="bounds/transform",
                expected="at most one of bounds=, transform=",
                actual=(bounds, transform),
            )
        if generator is not None and (bounds is not None or transform is not None):
            raise GeoBrainError(
                "Field generator= cannot be combined with bounds=/transform= "
                "; a generator network is responsible for its own output range",
                object_name="Field",
                field="generator",
                expected="bounds=None and transform=None when generator= is set",
                actual=(bounds, transform),
            )

        if init is not None:
            if not isinstance(init, torch.Tensor):
                raise GeoBrainError(
                    "Field init= must be a torch.Tensor",
                    object_name="Field",
                    field="init",
                    expected=torch.Tensor,
                    actual=type(init),
                )

        if generator is not None:
            if isinstance(generator, tuple):
                if len(generator) != 2 or not isinstance(generator[0], nn.Module) \
                        or not isinstance(generator[1], torch.Tensor):
                    raise GeoBrainError(
                        "Field generator= tuple must be (nn.Module, fixed_input: Tensor)",
                        object_name="Field",
                        field="generator",
                        expected="(nn.Module, torch.Tensor)",
                        actual=tuple(type(g) for g in generator),
                    )
            elif not isinstance(generator, nn.Module):
                raise GeoBrainError(
                    "Field generator= must be an nn.Module or (nn.Module, fixed_input)",
                    object_name="Field",
                    field="generator",
                    expected="nn.Module or (nn.Module, torch.Tensor)",
                    actual=type(generator),
                )

        transforms: tuple[InvertibleTransform, ...]
        if bounds is not None:
            try:
                lo, hi = bounds
            except (TypeError, ValueError):
                raise GeoBrainError(
                    "Field bounds= must be a (lo, hi) pair",
                    object_name="Field",
                    field="bounds",
                    expected="(lo, hi)",
                    actual=bounds,
                ) from None
            transforms = (AffineSigmoid(float(lo), float(hi)),)
        elif transform is not None:
            raw = transform if isinstance(transform, (list, tuple)) else (transform,)
            for t in raw:
                if not isinstance(t, InvertibleTransform):
                    raise GeoBrainError(
                        "Field transform= entries must be InvertibleTransform instances",
                        object_name="Field",
                        field="transform",
                        expected=InvertibleTransform,
                        actual=type(t),
                    )
            transforms = tuple(raw)
        else:
            transforms = ()

        if unit is not None and (not isinstance(unit, str) or not unit):
            raise GeoBrainError(
                "Field unit= must be a non-empty string or None",
                object_name="Field",
                field="unit",
                expected="non-empty str or None",
                actual=unit,
            )

        self.init = init
        self.generator = generator
        self.transforms = transforms
        self.unit = unit

    def __repr__(self) -> str:
        if self.generator is not None:
            src = f"generator={type(self.generator).__name__}" if not isinstance(
                self.generator, tuple
            ) else f"generator=({type(self.generator[0]).__name__}, fixed_input)"
        else:
            assert self.init is not None  # generator-less Field carries init
            src = f"init=Tensor{tuple(self.init.shape)}"
        tf = f", transforms={self.transforms}" if self.transforms else ""
        unit = f", unit={self.unit!r}" if self.unit else ""
        return f"Field({src}{tf}{unit})"
