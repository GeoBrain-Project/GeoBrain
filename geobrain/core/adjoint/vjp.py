"""
Explicit vector–Jacobian product through an Operator chain.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from ..errors import GeoBrainError
from ..operator import Operator
from ..containers import ModelState, ForwardOutput
from ..context import ForwardContext


def vjp(
    operator: Operator,
    state: ModelState,
    ctx: ForwardContext,
    *,
    cotangent: Mapping[str, torch.Tensor],
    field_names: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute ``Jᵀ · v`` of ``operator`` w.r.t. its trainable input fields.

    Args:
        operator: any :class:`Operator`. Reads its trainable input field
            names from ``operator.differentiability.trainable_inputs``.
        state: a ``ModelState`` whose ``.tensors`` contains every trainable
            input. Tensors are detached and cloned with ``requires_grad=True``
            before the forward; the caller does not need to set
            ``requires_grad`` themselves.
        ctx: the ``ForwardContext`` the operator needs.
        cotangent: dict ``output_name → tensor`` of cotangent vectors. For an
            :class:`ForwardOperator`, keys are entries of ``prediction.data``;
            for a :class:`PropertyTransform`, keys are entries of the output
            ``ModelState.tensors``. Each tensor must match the corresponding
            output's shape. Complex output channels contract under the
            Wirtinger convention: each term is ``Re(sum(conj(v) * out))``,
            so for a real leaf the gradient is
            ``Re(sum(conj(v) * d out / d theta))``. On real channels this
            realification is a no-op (bit-identical to ``(v * out).sum()``).
        field_names: optional subset of input field names to differentiate.
            Defaults to ``operator.differentiability.trainable_inputs``.

    Returns:
        ``dict[name, grad]`` mapping each requested input field name to its
        VJP. Inputs that contribute zero gradient (e.g. an unused field)
        appear with a zero tensor, never missing.
    """
    spec = operator.differentiability
    requested = tuple(field_names) if field_names is not None else tuple(spec.trainable_inputs)
    if not requested:
        raise GeoBrainError(
            "vjp requires at least one trainable input",
            object_name="vjp",
            field="field_names",
            expected="non-empty",
            actual=requested,
        )

    # Re-attach requires_grad on a clean clone so we don't pollute the caller.
    leafs: dict[str, torch.Tensor] = {}
    rebuilt_tensors: dict[str, torch.Tensor] = dict(state.tensors)
    for name in requested:
        if name not in state.tensors:
            raise GeoBrainError(
                "vjp missing input field",
                object_name="vjp",
                field=name,
                expected=f"present in {sorted(state.tensors)}",
                actual="absent",
            )
        leaf = state.tensors[name].detach().clone().requires_grad_(True)
        leafs[name] = leaf
        rebuilt_tensors[name] = leaf
    new_state = ModelState(tensors=rebuilt_tensors, metadata=state.metadata)

    out = operator(new_state, ctx)

    if isinstance(out, ForwardOutput):
        scalar_terms = []
        for k, v in cotangent.items():
            if k not in out.data:
                raise GeoBrainError(
                    "vjp cotangent key not in prediction.data",
                    object_name="vjp",
                    field=k,
                    expected=f"present in {sorted(out.data)}",
                    actual="absent",
                )
            scalar_terms.append(torch.real((v.conj() * out.data[k]).sum()))
    elif isinstance(out, ModelState):
        scalar_terms = []
        for k, v in cotangent.items():
            if k not in out.tensors:
                raise GeoBrainError(
                    "vjp cotangent key not in output state.tensors",
                    object_name="vjp",
                    field=k,
                    expected=f"present in {sorted(out.tensors)}",
                    actual="absent",
                )
            scalar_terms.append(torch.real((v.conj() * out.tensors[k]).sum()))
    else:                                       # pragma: no cover
        raise GeoBrainError(
            "vjp: operator returned unexpected type",
            object_name="vjp",
            field="output",
            expected="ForwardOutput or ModelState",
            actual=type(out),
        )

    if not scalar_terms:
        raise GeoBrainError(
            "vjp requires at least one cotangent term",
            object_name="vjp",
            field="cotangent",
            expected="non-empty mapping",
            actual=dict(cotangent),
        )

    scalar = torch.stack(scalar_terms).sum()
    grads = torch.autograd.grad(
        scalar, list(leafs.values()), retain_graph=False, allow_unused=True,
    )
    out_grads: dict[str, torch.Tensor] = {}
    for name, g in zip(leafs.keys(), grads):
        out_grads[name] = g if g is not None else torch.zeros_like(leafs[name])
    return out_grads
