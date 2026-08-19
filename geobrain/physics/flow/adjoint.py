# pyright: reportPrivateImportUsage=false
"""
Adjoint-state framework for the flow module.

Two layers stacked on top of each other:

Layer 1: primitives:

- :func:`solve_steady_adjoint`:            single-shot adjoint linear solve
  ``J^T λ = ∂L/∂x`` at a converged state. Sparse path goes through
  scipy ``spsolve``; dense through ``torch.linalg.solve``.
- :func:`newton_solve_with_adjoint`: Newton solve + implicit-FT
  cleanup so the returned ``x_star`` carries the right autograd link.
- :class:`ParameterSet`:             flat-view wrapper for a list of
  ``nn.Parameter`` tensors (perm / porosity / well controls).

Layer 2: transient adjoint:

- :class:`TransientAdjoint`:         multi-step adjoint walking the
  saved Newton checkpoints backwards. Per step does one adjoint solve
  via :func:`solve_steady_adjoint`, accumulates ``λ^T · ∂R/∂θ`` via either a
  user-supplied JVP callback or central FD on the parameter tensor.

Original FLOW-T1 docstring follows.
====================================================================
Newton solve with implicit-function-theorem adjoint.

At a converged state ``x*`` satisfying ``R(x*, θ) = 0``, the
implicit-function theorem gives::

    ∂x*/∂θ = − J⁻¹ · ∂R/∂θ          where J = ∂R/∂x | _{x*}

Composing with a scalar loss ``L(x*)``::

    ∂L/∂θ = (∂L/∂x*) · ∂x*/∂θ = − λᵀ ∂R/∂θ      with    Jᵀ λ = ∂L/∂x*

This is **one adjoint linear solve**; no Newton iterations have to
appear in the autograd graph.

This module wires that pattern into Newton: the forward solve runs
under :func:`torch.no_grad` (fast, memory-light); a single residual
re-evaluation at ``x*`` plus
:func:`geobrain.core.adjoint.linear_solve_with_adjoint` then
installs the correct backward pass, without ever building a custom
``torch.autograd.Function``.

The trick is that, at the converged ``x*``, ``R(x*, θ)`` is
**numerically tiny but algebraically nonzero**. Computing::

    x = x* − J⁻¹ · R(x*, θ)

is a no-op forward (``x ≈ x*``) but installs an autograd link from
``θ`` to ``x`` whose Jacobian is exactly ``−J⁻¹ ∂R/∂θ``. PyTorch's
own VJP for ``torch.linalg.solve`` then handles the adjoint solve.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import scipy.sparse.linalg as spla  # type: ignore[import-untyped]
import torch
import torch.nn as nn

from ...core import (
    GeoBrainError,
    linear_solve_with_adjoint,
    sparse_linear_solve_with_adjoint,
)
from .config import FlowHistoryConfig
from .errors import FlowContractError
from .history import FlowHistory, FlowHistoryWriter, _owned_control
from .solvers.linear_solvers import _to_scipy_csr
from .solvers import NewtonResult, NewtonSolver


class _AdjointModel(Protocol):
    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor: ...

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor: ...


_ControlledModelCall = Callable[
    [torch.Tensor, torch.Tensor, float, Mapping[str, object]],
    torch.Tensor,
]


def newton_solve_with_adjoint(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    jacobian_fn: Callable[[torch.Tensor], torch.Tensor],
    state0: torch.Tensor,
    newton_solver: NewtonSolver | None = None,
    adjoint_jacobian_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, NewtonResult]:
    """
    Solve ``R(x; θ) = 0`` for ``x`` with implicit-FT adjoint.

    Returns ``(x, NewtonResult)``. The returned ``x`` is the converged
    state with a clean autograd link to whatever trainable parameters
    appear inside ``residual_fn`` / ``jacobian_fn`` (typically through
    closure over ``rock.permeability_m2``, PVT compressibilities, etc.).

    Args:
        residual_fn: ``x → R(x; θ)``. Must be autograd-aware (will be
            called once at the converged state with autograd ENABLED).
        jacobian_fn: ``x → J(x; θ)``. Used for the forward Newton steps.
            May return dense or sparse; an approximate ``J`` (e.g. colored
            FD) is fine here; it only affects the step path, not the
            converged root.
        state0:        initial guess.
        newton_solver: an already-configured :class:`NewtonSolver`.
            Defaults to a fresh ``NewtonSolver()``.
        adjoint_jacobian_fn: optional ``x → J(x; θ)`` used **only** for the
            implicit-FT cleanup at ``x*`` (the linear operator whose
            transpose the adjoint solve inverts). The parameter gradient's
            accuracy is governed by this ``J``, so when ``jacobian_fn`` is a
            cheap approximation (colored FD), pass an exact one here to keep
            ``∂L/∂θ`` exact while the forward stays fast. Defaults to
            ``jacobian_fn`` (the single-Jacobian behaviour).

    Raises:
        GeoBrainError: if Newton does not converge in the no-grad phase.
    """
    solver = newton_solver if newton_solver is not None else NewtonSolver()

    # ---- Forward solve (no autograd graph) ----
    state0_d = state0.detach()
    with torch.no_grad():

        def _r(x: torch.Tensor) -> torch.Tensor:
            return residual_fn(x).detach()

        def _j(x: torch.Tensor) -> torch.Tensor:
            # Keep the Jacobian in its native storage: the default NewtonSolver
            # linear solver routes sparse J through scipy sparse LU instead of
            # densifying, so a sparse forward solve stays sparse all the way
            # through. Custom solvers passed by the caller are expected to
            # accept whatever ``jacobian_fn`` returns.
            return jacobian_fn(x).detach()

        result = solver.solve(_r, _j, state0_d)

    if not result.converged or result.state is None:
        raise GeoBrainError(
            "newton_solve_with_adjoint: Newton did not converge",
            object_name="newton_solve_with_adjoint",
            field="newton",
            expected="converged",
            actual=f"iters={result.iterations}, |R|={result.residual_norm:.3e}",
        )

    x_star = result.state.detach()

    # ---- IFT cleanup: one autograd-tracked evaluation at x_star ----
    # R(x_star, θ) is numerically tiny but its θ-derivative is the right
    # one: autograd will compute it.
    R_star = residual_fn(x_star)
    adj_jac = adjoint_jacobian_fn if adjoint_jacobian_fn is not None else jacobian_fn
    J_star = adj_jac(x_star)

    # x = x_star − J⁻¹ R(x_star, θ). Forward: x ≈ x_star to Newton tol.
    # Backward: ∂x/∂θ = −J⁻¹ ∂R/∂θ via the solver's implicit-FT VJP.
    # Route through the solver matching the Jacobian's storage: a sparse J
    # stays on the scipy-splu path (O(nnz) factor, reused for the adjoint
    # solve) instead of being densified to an O(n²) matrix solved in O(n³).
    if J_star.is_sparse:
        correction = sparse_linear_solve_with_adjoint(J_star, R_star)
    else:
        correction = linear_solve_with_adjoint(J_star, R_star)
    return x_star - correction, result


# ---------------------------------------------------------------------------
# solve_steady_adjoint: single-shot adjoint linear solve
# ---------------------------------------------------------------------------


def solve_steady_adjoint(
    jacobian: torch.Tensor,
    grad_state: torch.Tensor,
) -> torch.Tensor:
    """
    Solve ``J^T · λ = ∂L/∂x`` for the adjoint variable ``λ``.

    Given the converged Newton Jacobian ``J = ∂R/∂x | _{x*}`` and the
    upstream gradient ``∂L/∂x`` (typically the ``grad_output`` argument
    of a ``torch.autograd.Function.backward``), this returns ``λ``,
    the dual variable from which parameter gradients are reconstructed
    via ``∂L/∂θ = − λ^T · ∂R/∂θ``.

    Sparse path: scipy ``spsolve`` on ``J^T``. Dense path:
    ``torch.linalg.solve(J^T, ·)``. Returns ``λ`` on the same device
    and dtype as ``grad_state``.
    """
    device = grad_state.device
    dtype = grad_state.dtype
    if jacobian.is_sparse:
        J_csc = _to_scipy_csr(jacobian).T.tocsc()
        b = grad_state.detach().cpu().numpy()
        lam = spla.spsolve(J_csc, b)
        return torch.tensor(lam, device=device, dtype=dtype)
    return torch.linalg.solve(jacobian.T, grad_state)


# ---------------------------------------------------------------------------
# ParameterSet: flat-view wrapper around nn.Parameters
# ---------------------------------------------------------------------------


class ParameterSet:
    """
    Container wrapping flow-model parameters into a flat tensor view.

    ``params`` is a sequence of ``nn.Parameter`` (typically
    ``rock.permeability_m2``, ``rock.porosity``, well targets, …).
    :meth:`flat` returns a detached flat ``torch.Tensor`` view (useful
    for handing the parameter vector to a sampler / optimiser that
    wants a 1-D state). :meth:`scatter` writes a flat gradient back
    into each parameter's ``.grad`` field so a PyTorch optimiser can
    consume it.
    """

    def __init__(self, params: Sequence[nn.Parameter]) -> None:
        self.params = list(params)
        self.shapes = [tuple(p.shape) for p in self.params]
        self.sizes = [int(p.numel()) for p in self.params]
        self.total = int(sum(self.sizes))

    def flat(self) -> torch.Tensor:
        if not self.params:
            return torch.zeros(0)
        return torch.cat([p.detach().reshape(-1) for p in self.params])

    def scatter(self, flat_grad: torch.Tensor) -> None:
        offset = 0
        for p, n in zip(self.params, self.sizes):
            g = flat_grad[offset : offset + n].reshape(p.shape).to(p.dtype)
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)
            offset += n


# ---------------------------------------------------------------------------
# TransientAdjoint: multi-step adjoint with FD parameter-JVP fallback
# ---------------------------------------------------------------------------


@dataclass
class _StepCheckpoint:
    """Snapshot of one converged Newton step for adjoint backprop."""

    state: torch.Tensor
    state_old: torch.Tensor
    dt: float
    jacobian: torch.Tensor | None
    control: Mapping[str, object]
    accepted_step: int = 0


class TransientAdjoint:
    """
    Time-step adjoint for a transient simulation.

    Usage pattern::

        ta = TransientAdjoint(model)
        for t, dt in schedule:
            result = newton.solve(...)
            ta.record(result.state, state_old, dt, result.jacobian)
            state_old = result.state

        grad_theta = ta.gradient_wrt_parameter(
            perm_tensor,
            dJ_dx_per_step=lambda t, x: dJ_dx(t, x),
            residual_param_jvp=residual_param_jvp,   # optional
        )

    Walks the time series in reverse, accumulating parameter-wise
    gradients via the chain rule on each step's Jacobian. ``model``
    must expose ``model.residual(state, state_old, dt)``.
    """

    def __init__(
        self,
        model: _AdjointModel,
        *,
        history_config: FlowHistoryConfig | None = None,
        accepted_step_bound: int | None = None,
        recompute_solver: NewtonSolver | None = None,
    ) -> None:
        self.model = model
        self.history_config = (
            FlowHistoryConfig(mode="all") if history_config is None else history_config
        )
        if self.history_config.mode != "all" and accepted_step_bound is None:
            raise FlowContractError(
                "bounded adjoint history requires an accepted-step bound",
                object_name=type(self).__name__,
                field="accepted_step_bound",
                expected="explicit integer for non-'all' history",
                actual=None,
            )
        writer_bound = (2**63 - 1) if accepted_step_bound is None else accepted_step_bound
        self._writer = FlowHistoryWriter(
            self.history_config,
            accepted_step_bound=writer_bound,
        )
        self._accepted_step_bound = accepted_step_bound
        self._initial_state: torch.Tensor | None = None
        self._time_s = 0.0
        self._retained: dict[int, _StepCheckpoint] = {}
        self._recompute_solver = (
            recompute_solver if recompute_solver is not None else NewtonSolver(keep_jacobian=True)
        )
        self.history: list[_StepCheckpoint] = []

    @property
    def execution_history(self) -> FlowHistory:
        """Return the shared bounded history/accounting record."""
        if self._initial_state is None:
            raise FlowContractError(
                "adjoint history is empty",
                object_name=type(self).__name__,
                field="history",
                expected="at least one recorded step",
                actual="empty",
            )
        return self._writer.finalize()

    def _residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        control: Mapping[str, object],
    ) -> torch.Tensor:
        controlled = getattr(self.model, "residual_with_control", None)
        if callable(controlled):
            controlled_call = cast(_ControlledModelCall, controlled)
            return controlled_call(state, state_old, dt, control)
        if self._has_effective_control(control):
            raise FlowContractError(
                "recorded controls require model.residual_with_control for replay",
                object_name=type(self).__name__,
                field="control_schedule",
                expected="control-aware residual",
                actual=type(self.model).__name__,
            )
        return self.model.residual(state, state_old, dt)

    @staticmethod
    def _has_effective_control(control: Mapping[str, object]) -> bool:
        """Distinguish a canonical no-well snapshot from active controls."""
        if not control:
            return False
        return not (
            set(control) == {"wells"}
            and isinstance(control["wells"], (tuple, list))
            and len(control["wells"]) == 0
        )

    def _jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        control: Mapping[str, object],
    ) -> torch.Tensor:
        controlled = getattr(self.model, "jacobian_with_control", None)
        if callable(controlled):
            controlled_call = cast(_ControlledModelCall, controlled)
            return controlled_call(state, state_old, dt, control)
        if self._has_effective_control(control):
            raise FlowContractError(
                "recorded controls require model.jacobian_with_control for replay",
                object_name=type(self).__name__,
                field="control_schedule",
                expected="control-aware Jacobian",
                actual=type(self.model).__name__,
            )
        return self.model.jacobian(state, state_old, dt)

    def record(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        jacobian: torch.Tensor | None,
        *,
        control: Mapping[str, object] | None = None,
        residual_evaluations: int = 1,
        jacobian_assemblies: int | None = None,
        linear_solves: int = 1,
    ) -> None:
        if self._initial_state is None:
            self._initial_state = state_old.detach().clone()
            self._writer.record_initial(
                time_s=0.0,
                state={"state": self._initial_state},
            )
        index = self._writer.accepted_steps + 1
        checkpoint = _StepCheckpoint(
            state=state.detach().clone(),
            state_old=state_old.detach().clone(),
            dt=float(dt),
            jacobian=(jacobian.detach().clone() if jacobian is not None else None),
            control=_owned_control(control),
            accepted_step=index,
        )
        self._time_s += float(dt)
        self._writer.record_accepted(
            time_s=self._time_s,
            state={"state": checkpoint.state},
            dt_s=float(dt),
            control=checkpoint.control,
            residual_evaluations=residual_evaluations,
            jacobian_assemblies=(0 if jacobian is None else 1)
            if jacobian_assemblies is None
            else jacobian_assemblies,
            linear_solves=linear_solves,
        )
        retained_indices = set(
            self._writer.finalize(require_complete_reports=False).retained_step_indices
        )
        self._retained = {
            retained_index: retained
            for retained_index, retained in self._retained.items()
            if retained_index in retained_indices
        }
        if index in retained_indices:
            self._retained[index] = checkpoint
        self.history = [self._retained[key] for key in sorted(self._retained)]

    def _recompute_segment(
        self,
        *,
        start_step: int,
        end_step: int,
        state_start: torch.Tensor,
        dt_schedule: tuple[float, ...],
        control_schedule: tuple[Mapping[str, object], ...],
    ) -> list[_StepCheckpoint]:
        """Replay one bounded segment from a retained accepted state."""
        replayed: list[_StepCheckpoint] = []
        state_old = state_start.detach().clone()
        residual_evaluations = 0
        jacobian_assemblies = 0
        linear_solves = 0
        for accepted_step in range(start_step + 1, end_step + 1):
            dt = dt_schedule[accepted_step - 1]
            control = control_schedule[accepted_step - 1]
            def residual_at_step(
                x: torch.Tensor,
                old: torch.Tensor = state_old,
                step_dt: float = dt,
                schedule: Mapping[str, object] = control,
            ) -> torch.Tensor:
                return self._residual(x, old, step_dt, schedule)

            def jacobian_at_step(
                x: torch.Tensor,
                old: torch.Tensor = state_old,
                step_dt: float = dt,
                schedule: Mapping[str, object] = control,
            ) -> torch.Tensor:
                return self._jacobian(x, old, step_dt, schedule)

            result = self._recompute_solver.solve(
                residual_fn=residual_at_step,
                jacobian_fn=jacobian_at_step,
                state0=state_old,
            )
            if not result.converged or result.state is None:
                raise GeoBrainError(
                    "TransientAdjoint replay Newton solve did not converge",
                    object_name=type(self).__name__,
                    field="recompute",
                    expected="converged replay of the recorded schedule",
                    actual={
                        "accepted_step": accepted_step,
                        "iterations": result.iterations,
                        "residual_norm": result.residual_norm,
                    },
                )
            residual_evaluations += result.iterations + 1
            jacobian_assemblies += result.iterations
            linear_solves += result.iterations
            solved_state = result.state
            jacobian = result.jacobian
            if jacobian is None:
                jacobian = self._jacobian(solved_state, state_old, dt, control)
                jacobian_assemblies += 1
            replayed.append(
                _StepCheckpoint(
                    state=solved_state.detach().clone(),
                    state_old=state_old.detach().clone(),
                    dt=dt,
                    jacobian=jacobian.detach().clone(),
                    control=control,
                    accepted_step=accepted_step,
                )
            )
            state_old = solved_state.detach()
        self._writer.record_recomputed(
            len(replayed),
            residual_evaluations=residual_evaluations,
            jacobian_assemblies=jacobian_assemblies,
            linear_solves=linear_solves,
        )
        return replayed

    def _reverse_checkpoints(self) -> list[_StepCheckpoint]:
        """Materialize at most one configured segment for reverse traversal.

        The returned list is a compatibility seam for the existing adjoint loop.
        Recompute mode bounds the persistent history; a segment is released before
        the next one is replayed by :meth:`gradient_wrt_parameter`.
        """
        if self.history_config.mode == "all":
            return list(reversed(self.history))
        raise FlowContractError(
            "bounded reverse checkpoints must be consumed segment-wise",
            object_name=type(self).__name__,
            field="history",
            expected="segment iterator",
            actual=self.history_config.mode,
        )

    def _reverse_segments(self) -> Iterator[list[_StepCheckpoint]]:
        if self._initial_state is None:
            return
        if self.history_config.mode == "all":
            yield list(reversed(self.history))
            return
        execution = self._writer.finalize()
        total = execution.accounting.accepted_steps
        states_by_step: dict[int, torch.Tensor] = {0: self._initial_state}
        states_by_step.update(
            {index: checkpoint.state for index, checkpoint in self._retained.items()}
        )
        boundaries = sorted({0, total, *states_by_step})
        for position in range(len(boundaries) - 1, 0, -1):
            start = boundaries[position - 1]
            end = boundaries[position]
            replayed = self._recompute_segment(
                start_step=start,
                end_step=end,
                state_start=states_by_step[start],
                dt_schedule=execution.accepted_dt_s,
                control_schedule=execution.control_schedule,
            )
            yield list(reversed(replayed))

    def gradient_wrt_parameter(
        self,
        parameter: torch.Tensor,
        dJ_dx_per_step: Callable[[int, torch.Tensor], torch.Tensor],
        residual_param_jvp: Callable[
            [torch.Tensor, torch.Tensor, float, torch.Tensor],
            torch.Tensor,
        ]
        | None = None,
        residual_state_old_jvp: Callable[
            [torch.Tensor, torch.Tensor, float, torch.Tensor],
            torch.Tensor,
        ]
        | None = None,
        eps_rel: float = 1e-7,
    ) -> torch.Tensor:
        """
        Compute ``dJ/dθ`` via reverse time loop.

        ``dJ_dx_per_step(t_idx, state)`` returns the per-step
        ``∂J_t / ∂x`` (Mayer term). For pure terminal-time objectives,
        return zeros except at the last step.

        ``residual_param_jvp(state, state_old, dt, λ)`` should return
        ``λ^T · ∂R / ∂θ`` for the *single* ``parameter`` argument passed to
        this method; ``residual_state_old_jvp(state, state_old, dt, λ)``
        likewise returns ``λ^T · ∂R / ∂x_old``. When either is omitted the
        term is obtained by a single reverse-mode autograd VJP through
        ``model.residual``: exact, and ``O(1)`` residual evaluations per step
        instead of the old per-element central-FD fallback's ``O(n_dof)``.
        ``eps_rel`` is retained only for the explicit FD reference methods.
        """
        if self._initial_state is None or not self.history:
            return torch.zeros_like(parameter)
        final_checkpoint = self.history[-1]
        n_dof = final_checkpoint.state.numel()
        device = final_checkpoint.state.device
        dtype = final_checkpoint.state.dtype
        lam_next = torch.zeros(n_dof, device=device, dtype=dtype)
        grad_total = torch.zeros_like(parameter)
        for segment in self._reverse_segments():
            for cp in segment:
                t_idx = cp.accepted_step - 1
                dJ_dx = dJ_dx_per_step(t_idx, cp.state)
                rhs = dJ_dx + lam_next
                if cp.jacobian is None:
                    raise GeoBrainError(
                        "TransientAdjoint: checkpoint is missing its Jacobian "
                        "(pass keep_jacobian=True to NewtonSolver).",
                        object_name="TransientAdjoint",
                        field="jacobian",
                        expected="non-None torch.Tensor",
                        actual="None",
                    )
                lam = solve_steady_adjoint(cp.jacobian, rhs)
                if residual_param_jvp is not None:
                    grad_step = residual_param_jvp(cp.state, cp.state_old, cp.dt, lam)
                else:
                    grad_step = self._autograd_param_jvp(cp, lam, parameter)
                grad_total = grad_total - grad_step
                # Propagate λ back through state_old of this step to feed
                # the previous step's rhs: contribution = − ∂R/∂x_old · λ.
                if residual_state_old_jvp is not None:
                    lam_next = -residual_state_old_jvp(cp.state, cp.state_old, cp.dt, lam)
                else:
                    lam_next = -self._autograd_state_old_jvp(cp, lam)
        return grad_total

    def _autograd_param_jvp(
        self,
        cp: _StepCheckpoint,
        lam: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        """
        ``λ^T · ∂R/∂θ`` via one reverse-mode autograd VJP.

        ``parameter`` is the same tensor the model's ``residual`` reads
        (e.g. ``rock.permeability_m2``). Enabling grad on it and back-propagating ``lam``
        through a single residual evaluation gives ``λ^T ∂R/∂θ`` exactly, one
        residual eval per step, versus the FD fallback's ``2·numel`` evals with
        a host sync per element. Returns zeros if ``parameter`` does not appear
        in the residual graph (its true derivative is then zero).
        """
        had_grad = parameter.requires_grad
        parameter.requires_grad_(True)
        try:
            with torch.enable_grad():
                r = self._residual(cp.state, cp.state_old, cp.dt, cp.control)
                (grad,) = torch.autograd.grad(
                    r,
                    parameter,
                    grad_outputs=lam,
                    retain_graph=False,
                    allow_unused=True,
                )
        finally:
            parameter.requires_grad_(had_grad)
        if grad is None:
            return torch.zeros_like(parameter)
        return grad

    def _autograd_state_old_jvp(
        self,
        cp: _StepCheckpoint,
        lam: torch.Tensor,
    ) -> torch.Tensor:
        """``λ^T · ∂R/∂x_old`` via one reverse-mode autograd VJP.

        Replaces the per-cell central-FD loop over ``state_old`` (which cost
        ``2·n_dof`` residual evals per step) with a single VJP through a fresh
        leaf copy of ``state_old``.
        """
        xo = cp.state_old.detach().requires_grad_(True)
        with torch.enable_grad():
            r = self._residual(cp.state, xo, cp.dt, cp.control)
            (grad,) = torch.autograd.grad(
                r,
                xo,
                grad_outputs=lam,
                retain_graph=False,
                allow_unused=True,
            )
        if grad is None:
            return torch.zeros(
                cp.state_old.numel(),
                device=lam.device,
                dtype=lam.dtype,
            )
        return grad.reshape(-1)

    def _fd_param_jvp(
        self,
        cp: _StepCheckpoint,
        lam: torch.Tensor,
        parameter: torch.Tensor,
        eps_rel: float,
    ) -> torch.Tensor:
        """
        ``λ^T · ∂R/∂θ`` via central FD on ``parameter`` (reference / debug
        fallback; the default path is :meth:`_autograd_param_jvp`).

        Wrapped in ``try/finally`` so a residual evaluation that
        raises (NaN, failed Newton, etc.) cannot leave the live
        ``parameter`` buffer in a perturbed state. Perturbation is
        applied to ``parameter.data`` to avoid the in-place-leaf-op
        error when the parameter has ``requires_grad=True``.
        """
        data = parameter.data if hasattr(parameter, "data") else parameter
        if parameter.dim() == 0:
            # Relative floor: an absolute +1e-8 floor dwarfs SI-scale
            # parameters (permeability ~1e-14 m²) and destroys the probe.
            h = eps_rel * float(parameter.abs()) if float(parameter.abs()) > 0.0 else eps_rel
            p_save = data.detach().clone()
            try:
                with torch.no_grad():
                    data.add_(h)
                    r_plus = self._residual(cp.state, cp.state_old, cp.dt, cp.control)
                    data.copy_(p_save - h)
                    r_minus = self._residual(cp.state, cp.state_old, cp.dt, cp.control)
            finally:
                with torch.no_grad():
                    data.copy_(p_save)
            return (lam @ (r_plus - r_minus)) / (2.0 * h)
        out = torch.zeros_like(parameter)
        flat = data.view(-1)
        out_flat = out.view(-1)
        flat_save = flat.detach().clone()
        # Relative floor for zero entries: never an absolute constant, which
        # dwarfs SI-scale parameters (e.g. permeability ~1e-14 m²).
        magnitude_floor = 1e-6 * float(flat_save.abs().max()) or 1.0e-30
        try:
            with torch.no_grad():
                for i in range(flat.numel()):
                    h = eps_rel * (abs(flat_save[i].item()) + magnitude_floor)
                    save = flat_save[i].item()
                    flat[i] = save + h
                    r_plus = self._residual(cp.state, cp.state_old, cp.dt, cp.control)
                    flat[i] = save - h
                    r_minus = self._residual(cp.state, cp.state_old, cp.dt, cp.control)
                    flat[i] = save
                    out_flat[i] = float((lam @ (r_plus - r_minus)) / (2.0 * h))
        finally:
            with torch.no_grad():
                flat.copy_(flat_save)
        return out

    def _fd_state_old_jvp(
        self,
        cp: _StepCheckpoint,
        lam: torch.Tensor,
        eps_rel: float,
    ) -> torch.Tensor:
        """``λ^T · ∂R/∂x_old`` via central FD on ``state_old``."""
        n = cp.state_old.numel()
        out = torch.zeros(n, device=lam.device, dtype=lam.dtype)
        with torch.no_grad():
            for i in range(n):
                h = eps_rel * (abs(cp.state_old[i].item()) + 1e-8)
                xo = cp.state_old.clone()
                xo[i] += h
                r_plus = self._residual(cp.state, xo, cp.dt, cp.control)
                xo[i] -= 2 * h
                r_minus = self._residual(cp.state, xo, cp.dt, cp.control)
                out[i] = (lam @ (r_plus - r_minus)) / (2.0 * h)
        return out


__all__ = [
    "ParameterSet",
    "TransientAdjoint",
    "newton_solve_with_adjoint",
    "solve_steady_adjoint",
]
