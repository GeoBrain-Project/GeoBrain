"""
Private implementation of :meth:`Inverter.from_function`.

Lives next to :mod:`inverter.py` purely to keep that file at a tractable size.
Builds a one-off :class:`ForwardOperator` + :class:`InverseProblem` from a raw
forward function, then hands it to the user-supplied ``cls`` (always
:class:`geobrain.optim.Inverter` in practice) for construction.

Why a module-level helper? ``Inverter.from_function`` needs ``cls`` so that
subclasses still get a sub-instance back. Passing ``cls`` in keeps the
classmethod contract intact while letting this (sizeable) body live out of the
way.

Deep-image-prior / latent-code convenience factories are deliberately NOT
provided here; that job belongs to the operator-native
reparameterization seam (:class:`geobrain.nn.WeightReparameterization` /
:class:`geobrain.nn.LatentReparameterization`, composed as
``physics_op @ reparam``), which supports multi-channel problems and every
bayes sampler instead of just the Inverter.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import torch

from geobrain.core import ForwardContext, GeoBrainError, ModelState
from geobrain.inverse import GaussianLikelihood, InverseProblem, Likelihood
from .config import AdamConfig, LBFGSConfig
from .processing import GradientProcessor, StepProjection

_ParamDict = dict[str, torch.Tensor]
_Regularizer = Callable[[_ParamDict], torch.Tensor]


def _inverter_from_function(
    cls: type,
    forward_fn: Callable[..., torch.Tensor],
    observed: torch.Tensor,
    *,
    params: Mapping[str, torch.Tensor],
    likelihood: Likelihood | None = None,
    optimizer: AdamConfig | LBFGSConfig = AdamConfig(),
    learning_rates: Mapping[str, float] | None = None,
    regularizer: _Regularizer | None = None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    ctx: ForwardContext | None = None,
    gradient_processors: Sequence[GradientProcessor] = (),
    step_projections: Sequence[StepProjection] = (),
) -> Any:
    """Build an :class:`Inverter` from a bare forward function.

    Implementation backing :meth:`Inverter.from_function`.
    """
    # Import locally to avoid bloating module import path.
    from geobrain.core.differentiability import (
        DifferentiabilityLevel,
        DifferentiabilitySpec,
    )
    from geobrain.core.containers import ForwardOutput

    if TYPE_CHECKING:
        class _ForwardOperator:
            differentiability: DifferentiabilitySpec

            def _forward(
                self, state: ModelState, ctx: ForwardContext
            ) -> ModelState | ForwardOutput: ...

    else:
        from geobrain.core.operator import ForwardOperator as _ForwardOperator

    if not isinstance(observed, torch.Tensor):
        raise GeoBrainError(
            "Inverter.from_function: observed must be a torch.Tensor",
            object_name="Inverter.from_function",
            field="observed",
            expected=torch.Tensor,
            actual=type(observed),
        )
    if not params:
        raise GeoBrainError(
            "Inverter.from_function: params must be non-empty",
            object_name="Inverter.from_function",
            field="params",
            expected="non-empty mapping",
            actual={},
        )
    trainable_keys: tuple[str, ...] = tuple(params.keys())
    _output_channel = "data"

    class _FnForwardOperator(_ForwardOperator):
        differentiability = DifferentiabilitySpec(
            level=DifferentiabilityLevel.FULL_AUTOGRAD,
            trainable_inputs=trainable_keys,
            output_keys=(_output_channel,),
        )

        def _forward(
            self, state: ModelState, ctx: ForwardContext
        ) -> ForwardOutput:
            kwargs_in = {
                k: state.tensors[k] for k in trainable_keys
            }
            y = forward_fn(**kwargs_in)
            if not isinstance(y, torch.Tensor):
                raise GeoBrainError(
                    "forward_fn must return a torch.Tensor",
                    object_name="Inverter.from_function",
                    field="forward_fn",
                    expected=torch.Tensor,
                    actual=type(y),
                )
            return ForwardOutput(data={_output_channel: y})

    op = _FnForwardOperator()
    problem = InverseProblem(
        forward=op,
        observed={_output_channel: observed},
        likelihood=likelihood if likelihood is not None else GaussianLikelihood(std=1.0),
    )
    return cls(
        problem,
        params=params,
        optimizer=optimizer,
        learning_rates=learning_rates,
        regularizer=regularizer,
        bounds=bounds,
        ctx=ctx,
        gradient_processors=gradient_processors,
        step_projections=step_projections,
    )


__all__ = [
    "_inverter_from_function",
]
