"""``Link``: a differentiable derived field for :class:`~geobrain.geomodel.earthmodel.EarthModel`.

A pure descriptor over a caller-supplied callable: ``fn(*args, **params)`` is
evaluated by ``EarthModel.resolve`` after every named ``args`` dependency
(base fields or other links) has itself been resolved to its PHYSICAL value.
``params`` floats are tensorised at ``EarthModel`` construction time (the
model's unified dtype/device); see ``EarthModel``'s docstring; a ``Link``
itself stores whatever raw ``float``/``Tensor`` values the caller passed.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import torch

from geobrain.core.errors import GeoBrainError

__all__ = ["Link"]


class Link:
    """A derived field: ``value = fn(*[resolved(a) for a in args], **params)``.

    Args:
        fn: Differentiable callable taking one positional tensor per name in
            ``args`` (in order), plus the keyword ``params``. Must return a
            single :class:`torch.Tensor`.
        args: Field/link names (base fields or other link outputs) this link
            depends on, in the positional order ``fn`` expects. A bare string
            is accepted as sugar for a single-name tuple.
        params: ``{name: float | Tensor}``, additional keyword arguments to
            ``fn``. Floats are TENSORISED at ``EarthModel`` construction
            (the model's dtype/device); a parameter is trainable iff its
            (post-tensorisation) tensor has ``requires_grad=True``; pass an
            already-``requires_grad_()`` tensor to make a link coefficient
            an inversion unknown (namespaced ``links.<name>.<param>`` in
            :meth:`~geobrain.geomodel.earthmodel.EarthModel.trainables`).
        unit: Explicit SI unit string for this link's output. ``None`` (the
            default) defers to :data:`geobrain.geomodel.earthmodel.field_specs.FIELD_SPECS`'s
            by-name default, resolved by the owning ``EarthModel``.

    Note:
        No ``bounds=`` kwarg by design: a rock chain that needs a soft
        range clamps inside ``fn`` in one line (see the extending guide in
        the documentation for the ``soft_clamp`` idiom).
    """

    __slots__ = ("fn", "args", "params", "unit")

    def __init__(
        self,
        fn: Callable[..., torch.Tensor],
        args: str | Sequence[str],
        params: Mapping[str, float | torch.Tensor] | None = None,
        unit: str | None = None,
    ) -> None:
        if not callable(fn):
            raise GeoBrainError(
                "Link fn= must be callable",
                object_name="Link",
                field="fn",
                expected="callable",
                actual=type(fn),
            )
        arg_tuple = (args,) if isinstance(args, str) else tuple(args)
        if not arg_tuple:
            raise GeoBrainError(
                "Link args= must be a non-empty name or sequence of names",
                object_name="Link",
                field="args",
                expected="non-empty str or sequence of str",
                actual=args,
            )
        for a in arg_tuple:
            if not isinstance(a, str) or not a:
                raise GeoBrainError(
                    "Link args= entries must be non-empty strings",
                    object_name="Link",
                    field="args",
                    expected="non-empty string names",
                    actual=a,
                )

        params_dict = dict(params or {})
        for name, value in params_dict.items():
            if not isinstance(name, str) or not name:
                raise GeoBrainError(
                    "Link params= keys must be non-empty strings",
                    object_name="Link",
                    field="params",
                    expected="non-empty string keys",
                    actual=name,
                )
            if not isinstance(value, (int, float, torch.Tensor)) or isinstance(value, bool):
                raise GeoBrainError(
                    "Link params= values must be float or torch.Tensor",
                    object_name="Link",
                    field=f"params[{name!r}]",
                    expected="float or torch.Tensor",
                    actual=type(value),
                )

        if unit is not None and (not isinstance(unit, str) or not unit):
            raise GeoBrainError(
                "Link unit= must be a non-empty string or None",
                object_name="Link",
                field="unit",
                expected="non-empty str or None",
                actual=unit,
            )

        self.fn = fn
        self.args = arg_tuple
        self.params = params_dict
        self.unit = unit

    def __repr__(self) -> str:
        fn_name = getattr(self.fn, "__name__", type(self.fn).__name__)
        unit = f", unit={self.unit!r}" if self.unit else ""
        return f"Link({fn_name}, args={self.args}{unit})"
