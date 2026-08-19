"""
Time-step controllers for the flow time loop.

Two stock controllers cover the bulk of single-physics workflows:

- :class:`TimeStepScheduler`: fixed list of dt's (matching observed
  time stations or user-supplied schedule).
- :class:`AdaptiveTimeStepper`: Newton-iteration-driven dt control.
  Grows dt on easy steps; cuts dt on hard / failed Newton solves.

The CFL estimator :func:`estimate_cfl_dt` is a standalone helper for
IMPES / explicit-saturation stepping (no implicit kernel consumes it).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Protocol

import torch

from ._defaults import (
    DT_CUT_FACTOR,
    DT_GROW_FACTOR,
    EPS,
    TARGET_ITER,
)
from .solvers import NewtonSolver
from .config import FlowHistoryConfig
from .errors import FlowContractError, FlowConvergenceError
from .history import FlowHistory, FlowHistoryWriter
from .solvers.diagnostics import (
    FlowConvergenceDiagnostics,
    convergence_diagnostics,
    normalize_convergence_diagnostics,
)

# SI time-step bounds (seconds). The pre-SI FIELD defaults were 1e-6..365
# days; these are the same physical bounds expressed in seconds.
DT_MIN: float = 8.64e-2
DT_MAX: float = 3.1536e7

# A "converged" TPFA/model transient step can still be grossly non-physical (a
# saturation outside [0,1]) under prescribed-source over-withdrawal / over-injection:
# the OilWater/BlackOil residual clamps saturation only where it feeds relperm, so the
# accumulation happily balances an impossible Sw. Guard the accepted state to the same
# tolerance the NFVM family uses (``nfvm_two_phase`` fails loud at |Sw−[0,1]|>1e-2).
_SAT_BOUND_TOL = 1e-2
_SATURATION_KEYS = ("sw", "sg", "so")


class _TransientModel(Protocol):
    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]: ...

    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        **kwargs: object,
    ) -> torch.Tensor: ...

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        **kwargs: object,
    ) -> torch.Tensor: ...


def _first_nonphysical_saturation(
    model: _TransientModel,
    state: torch.Tensor,
) -> tuple[str, float, float] | None:
    """Return ``(key, s_min, s_max)`` for the first saturation field of ``model`` that
    escapes ``[0, 1]`` by more than ``_SAT_BOUND_TOL``, else ``None``.

    Uses ``model.state_split`` to read the saturations, so it is a no-op for
    pressure-only / thermal-only states (no saturation keys) and never trips on a
    healthy multiphase run (Sw stays within [0,1] to the tolerance)."""
    split = getattr(model, "state_split", None)
    if split is None:
        return None
    parts = split(state)
    for key in _SATURATION_KEYS:
        val = parts.get(key)
        if val is None:
            continue
        s_min = float(val.min())
        s_max = float(val.max())
        if s_min < -_SAT_BOUND_TOL or s_max > 1.0 + _SAT_BOUND_TOL:
            return key, s_min, s_max
    return None


@dataclass
class TimeStepScheduler:
    """User-provided list of dt values."""

    dt_list: list[float]

    def steps(self) -> list[float]:
        return list(self.dt_list)


class AdaptiveTimeStepper:
    """
    Newton-iteration-driven adaptive dt control.

    After each Newton solve:

    - ``not converged`` → cut ``dt`` by ``cut_factor`` (clamped to ``dt_min``)
    - ``iterations <= target_iter // 2`` → grow ``dt`` by ``grow_factor``
      (clamped to ``dt_max``)
    - ``iterations > target_iter``       → cut ``dt`` by ``cut_factor``
    - otherwise → keep ``dt`` unchanged

    Args:
        dt_init: first step size [s].
        dt_min / dt_max: step-size bounds [s].
        target_iter: Newton iterations aimed for per step.
        grow_factor / cut_factor: step multipliers on easy / hard steps.
    """

    def __init__(
        self,
        dt_init: float = 1.0,
        dt_min: float = DT_MIN,
        dt_max: float = DT_MAX,
        target_iter: int = TARGET_ITER,
        grow_factor: float = DT_GROW_FACTOR,
        cut_factor: float = DT_CUT_FACTOR,
    ) -> None:
        self.dt = float(dt_init)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.target_iter = int(target_iter)
        self.grow_factor = float(grow_factor)
        self.cut_factor = float(cut_factor)

    def update(self, iterations: int, converged: bool) -> float:
        if not converged:
            self.dt = max(self.dt * self.cut_factor, self.dt_min)
        elif iterations <= self.target_iter // 2:
            self.dt = min(self.dt * self.grow_factor, self.dt_max)
        elif iterations > self.target_iter:
            self.dt = max(self.dt * self.cut_factor, self.dt_min)
        return self.dt


@dataclass
class TransientResult:
    """Outcome of an adaptive transient march.

    ``times``/``states`` are the recorded report stations (always including the
    initial and final state); ``dt_history`` the accepted step sizes;
    ``n_cuts`` the number of rejected (Newton-failed) attempts.
    """

    times: list[float]
    states: list[torch.Tensor]
    dt_history: list[float] = field(default_factory=list)
    n_cuts: int = 0
    rejected: list[FlowConvergenceDiagnostics] = field(default_factory=list)
    history: FlowHistory | None = None


def run_transient(
    model: _TransientModel,
    state: torch.Tensor,
    t_end: float,
    stepper: AdaptiveTimeStepper,
    *,
    solver: NewtonSolver | None = None,
    report_times: list[float] | None = None,
    history_config: FlowHistoryConfig | None = None,
    accepted_step_bound: int | None = None,
    max_consecutive_cuts: int = 25,
    **residual_kwargs: object,
) -> TransientResult:
    """Adaptive-dt transient march for any model with ``residual(state, state_old,
    dt, **kw)`` + ``jacobian(...)`` (e.g. the MPFA two-/three-phase / compositional
    kernels).

    Each step is solved by ``solver`` (a fully-implicit Newton). On success the
    step is **accepted** (state + time advance) and :class:`AdaptiveTimeStepper`
    grows/keeps ``dt`` from the Newton iteration count; on failure the step is
    **rejected**; ``dt`` is cut and the *same* step is retried from the unchanged
    state. ``dt`` is capped so the march lands exactly on ``t_end`` and on each
    ``report_times`` station (where states are recorded; otherwise every accepted
    step is recorded). Raises if ``dt`` reaches ``dt_min`` while still failing.

    Returns a :class:`TransientResult`. ``**residual_kwargs`` (e.g.
    ``source_water=...``) are forwarded to ``residual``/``jacobian``.
    """
    solver = solver if solver is not None else NewtonSolver()
    if history_config is not None and report_times is not None:
        raise FlowContractError(
            "history report stations must have one authority",
            object_name="run_transient",
            field="report_times",
            expected="history_config or report_times, not both",
            actual="both supplied",
        )
    if history_config is None:
        history_config = (
            FlowHistoryConfig(
                mode="report",
                report_times_s=tuple(sorted(float(r) for r in report_times)),
            )
            if report_times
            else FlowHistoryConfig(mode="all")
        )
    reports = list(history_config.report_times_s) if history_config.mode == "report" else None
    if reports is not None:
        outside_window = tuple(report for report in reports if report > float(t_end) + 1e-12)
        if outside_window:
            raise FlowContractError(
                "report_times_s must lie inside the requested transient window",
                object_name="run_transient",
                field="report_times_s",
                expected=f"0 <= report <= {float(t_end)}",
                actual=outside_window,
            )
    step_bound = (
        accepted_step_bound
        if accepted_step_bound is not None
        else math.ceil(t_end / stepper.dt_min) + (0 if reports is None else len(reports))
    )
    history_writer = FlowHistoryWriter(
        history_config,
        accepted_step_bound=step_bound,
    )
    history_writer.record_initial(time_s=0.0, state={"state": state})
    ri = 0
    if reports is not None:  # skip stations at/before t=0
        while ri < len(reports) and reports[ri] <= 1e-12:
            ri += 1
    t = 0.0
    res = TransientResult(
        times=[0.0],
        states=[state],
        dt_history=[],
        n_cuts=0,
        rejected=[],
    )
    cuts = 0
    last_failure: FlowConvergenceError | None = None
    last_diagnostics: FlowConvergenceDiagnostics | None = None
    step_index = 0

    def sync_bounded_history(*, require_complete_reports: bool = True) -> None:
        history = history_writer.finalize(require_complete_reports=require_complete_reports)
        res.history = history
        res.times = list(history.times_s)
        res.states = [entry["state"] for entry in history.states]
        res.rejected = list(history.rejected)

    while t < t_end - 1e-12:
        dt = min(stepper.dt, t_end - t)
        if reports is not None and ri < len(reports):
            dt = min(dt, reports[ri] - t)
        accepted_state: torch.Tensor | None = None
        try:
            out = solver.solve(
                lambda s: model.residual(s, state, dt, **residual_kwargs),
                lambda s: model.jacobian(s, state, dt, **residual_kwargs),
                state,
            )
            if not out.converged:
                if out.diagnostics is None:
                    normalized = convergence_diagnostics(
                        stage="nonlinear",
                        converged=False,
                        reason="line_search" if out.no_progress else "max_iterations",
                        iterations=out.iterations,
                        max_iterations=solver.max_iter,
                        initial_residual_norm=(
                            float(out.history[0])
                            if out.history
                            else float(out.residual_norm)
                        ),
                        residual_norm=float(out.residual_norm),
                        residual_history=tuple(out.history),
                    )
                else:
                    normalized = normalize_convergence_diagnostics(
                        out.diagnostics,
                        fallback_max_iterations=solver.max_iter,
                    )
                raise FlowConvergenceError(
                    "Newton solver returned a non-converged result",
                    object_name="run_transient",
                    field="convergence",
                    expected="converged nonlinear state",
                    actual=normalized.to_dict(),
                    diagnostics=normalized,
                )
            iterations = out.iterations
        except FlowConvergenceError as error:
            # A too-large dt may fail a nonlinear or linear stage. Preserve the
            # exact typed diagnostic while retrying the unchanged state.
            last_failure = error
            normalized = normalize_convergence_diagnostics(
                error.diagnostics,
                fallback_max_iterations=solver.max_iter,
            )
            last_diagnostics = replace(
                normalized,
                time_s=t,
                step_index=step_index,
            )
            history_writer.record_rejected(
                last_diagnostics,
                residual_evaluations=last_diagnostics.iterations + 1,
                jacobian_assemblies=last_diagnostics.iterations,
                linear_solves=last_diagnostics.iterations,
            )
            out, converged, iterations = None, False, stepper.target_iter + 1
        else:
            assert out is not None
            if not isinstance(out.state, torch.Tensor):
                diagnostics = convergence_diagnostics(
                    stage="nonlinear",
                    converged=False,
                    reason="invalid_state",
                    iterations=out.iterations,
                    max_iterations=solver.max_iter,
                    initial_residual_norm=(
                        float(out.history[0]) if out.history else float(out.residual_norm)
                    ),
                    residual_norm=float(out.residual_norm),
                    residual_history=tuple(out.history),
                    time_s=t,
                    step_index=step_index,
                )
                state_error = FlowConvergenceError(
                    "Newton solver reported convergence without a tensor state",
                    object_name="run_transient",
                    field="state",
                    expected="torch.Tensor",
                    actual=type(out.state).__name__,
                    diagnostics=diagnostics,
                )
                sync_bounded_history(require_complete_reports=False)
                state_error.partial_result = res
                raise state_error
            accepted_state = out.state
            converged = True
        if converged:
            assert out is not None
            assert accepted_state is not None
            state = accepted_state
            t += dt
            cuts = 0
            # Post-convergence physical-bound guard: a "converged" step whose saturation
            # has run outside [0,1] (over-withdrawal drained a cell past its pore volume)
            # is non-physical, not a solution: fail loud instead of marching on a wrong
            # state, mirroring the NFVM family. ``res`` (the accepted states so far) is
            # attached as ``.partial_result``.
            bad = _first_nonphysical_saturation(model, state)
            if bad is not None:
                key, s_min, s_max = bad
                split = model.state_split(state)
                values = split[key]
                failed = tuple(
                    int(index)
                    for index in torch.nonzero(
                        (values < -_SAT_BOUND_TOL) | (values > 1.0 + _SAT_BOUND_TOL),
                        as_tuple=False,
                    ).flatten()
                )
                diagnostics = convergence_diagnostics(
                    stage="nonlinear",
                    converged=False,
                    reason="invalid_state",
                    iterations=iterations,
                    max_iterations=solver.max_iter,
                    initial_residual_norm=float(out.residual_norm),
                    residual_norm=float(out.residual_norm),
                    residual_history=tuple(out.history),
                    failed_cells=failed,
                    time_s=t,
                    step_index=step_index,
                )
                err = FlowConvergenceError(
                    "transient step converged to a non-physical saturation",
                    object_name="run_transient",
                    field=key,
                    expected="saturation in [0, 1]",
                    actual=(s_min, s_max),
                    diagnostics=diagnostics,
                )
                sync_bounded_history(require_complete_reports=False)
                err.partial_result = res
                raise err
            res.dt_history.append(dt)
            history_writer.record_accepted(
                time_s=t,
                state={"state": state},
                dt_s=dt,
                residual_evaluations=iterations + 1,
                jacobian_assemblies=iterations,
                linear_solves=iterations,
            )
            stepper.update(iterations=iterations, converged=True)
            on_report = (
                reports is not None
                and ri < len(reports)
                and abs(t - reports[ri]) <= 1e-9 * max(1.0, reports[ri])
            )
            if on_report:
                ri += 1
        else:
            cuts += 1
            res.n_cuts += 1
            stepper.update(iterations=stepper.target_iter + 1, converged=False)
            if cuts > max_consecutive_cuts or stepper.dt <= stepper.dt_min * (1.0 + 1e-9):
                prior = last_diagnostics
                if prior is None:
                    prior = convergence_diagnostics(
                        stage="nonlinear",
                        converged=False,
                        reason="max_iterations",
                        iterations=0,
                        max_iterations=solver.max_iter,
                        initial_residual_norm=float("inf"),
                        residual_norm=float("inf"),
                        residual_history=(),
                    )
                diagnostics = replace(prior, time_s=t, step_index=step_index)
                err = FlowConvergenceError(
                    "transient solve exhausted its timestep retry budget",
                    object_name="run_transient",
                    field="dt",
                    expected="convergence above dt_min",
                    actual=stepper.dt,
                    diagnostics=diagnostics,
                )
                sync_bounded_history(require_complete_reports=False)
                err.partial_result = res  # the accepted states so far (mirrors bayes samplers)
                if last_failure is None:
                    raise err
                raise err from last_failure
        step_index += 1
    sync_bounded_history()
    return res


def estimate_cfl_dt(
    cell_volumes: torch.Tensor,
    porosity: torch.Tensor,
    cell_flux_abs: torch.Tensor,
    cfl_max: float = 0.5,
) -> float:
    """
    CFL-limited dt for advection of saturation fronts.

    ``dt_cell = φ · V / Σ|q_face|``;
    ``dt_cfl  = cfl_max · min_cell(dt_cell)``.
    """
    flux = cell_flux_abs.clamp_min(EPS)
    return float(cfl_max * (porosity * cell_volumes / flux).min())


__all__ = [
    "AdaptiveTimeStepper",
    "TimeStepScheduler",
    "TransientResult",
    "run_transient",
    "estimate_cfl_dt",
]
