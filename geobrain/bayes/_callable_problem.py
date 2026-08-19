"""
Adapter: wrap a log-probability callable so the samplers can consume it.

The samplers (HMC/NUTS/LangevinDynamics/SVGD) take a full
:class:`InverseProblem` and
internally call ``problem.log_posterior(state, ctx)``; they read ``state.tensors``
to pull out parameter Tensors and treat ``ctx`` as opaque.

Callable-mode users carry a plain ``θ → scalar`` log-probability function. The
:class:`_CallableProblem` adapter lets such a callable masquerade as an
``InverseProblem`` so the existing sampler internals work unchanged.

Design choices:

- The adapter implements only the **subset** of ``InverseProblem`` the four
  samplers actually invoke: ``log_posterior`` (HMC, NUTS, SVGD) and, for
  completeness, ``log_likelihood``. ``predict`` / ``evaluate`` / ``loss`` are
  omitted; they would need a forward operator callable-mode users do not supply.
- ``log_prob_fn`` receives a ``dict[str, Tensor]`` keyed by ``init.keys()``,
  matching the standard "params dict" convention the samplers use.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch


class _CallableProblem:
    """Minimal ``InverseProblem`` look-alike around a log-probability callable.

    Args:
        log_prob_fn: callable ``dict[str, Tensor] → scalar Tensor``. Must be
            autograd-aware (returns a Tensor that participates in ``grad``).
        param_keys: ordered keys to pluck from ``state.tensors`` and pass
            into ``log_prob_fn`` as a dict.
    """

    def __init__(
        self,
        log_prob_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
        param_keys: tuple[str, ...],
    ) -> None:
        self._log_prob_fn = log_prob_fn
        self._param_keys = tuple(param_keys)

    def log_posterior(self, state: Any, ctx: Any = None) -> torch.Tensor:
        """Evaluate the wrapped callable on the named subset of state tensors."""
        theta = {k: state.tensors[k] for k in self._param_keys}
        return self._log_prob_fn(theta)

    # Provide the likelihood-named alias too. Treat the callable as the full
    # posterior (prior already baked in by the caller).
    def log_likelihood(self, state: Any, ctx: Any = None) -> torch.Tensor:
        return self.log_posterior(state, ctx)


def build_callable_problem(
    log_prob_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    init: Mapping[str, torch.Tensor],
) -> _CallableProblem:
    """Convenience constructor used by sampler ``from_callable`` factories."""
    if not callable(log_prob_fn):
        raise TypeError(
            f"log_prob_fn must be callable, got {type(log_prob_fn).__name__}"
        )
    if not init:
        raise ValueError("init must be a non-empty mapping of param tensors")
    return _CallableProblem(log_prob_fn, tuple(init.keys()))
