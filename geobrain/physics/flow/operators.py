# pyright: reportPrivateImportUsage=false
"""
ForwardOperator wrappers for the flow module.

Three layers sit above the residual kernels:

- :class:`FlowEvolutionOperator`: single-step forward, takes one
  ``ModelState`` and one ``dt`` from the ``ForwardContext``,
  evaluates one Newton solve, returns the updated state. Works on
  any of :class:`SinglePhaseModel`, :class:`OilWaterModel`,
  :class:`BlackOilModel` (duck-typed on the residual signature).

- :class:`TransientFlowOperator`: full time march. Takes a
  :class:`TimeStepScheduler` or :class:`AdaptiveTimeStepper` from
  ``ctx.time``, marches Newton-step-by-Newton-step, and returns a
  ``ForwardOutput`` with the final state plus optional time series.

- :class:`WellObservationOperator`: extracts per-well BHP and per-
  phase rates from a state vector. Differentiable, plays nicely with
  history-match losses.

One adapter sits beside them:

- :class:`ParametricFlowOperator`: lifts constructor-bound model
  properties (perm/poro/...) into ``ModelState`` trainable inputs and
  runs the transient march, so the flow family joins
  ``InverseProblem``/``JointProblem`` wire-by-name for parameter
  inversion (the 5-spot adapter pattern as a library citizen).

Wells, when supplied, are folded into the residual through the
:class:`WellGroup` per-phase source-rate aggregator (no well code
lives inside the residual kernels; see :mod:`geobrain.physics.flow.wells`).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Mapping, cast

import torch

from ...core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    GeoBrainError,
    ModelState,
    ForwardOperator,
    ForwardOutput,
)
from .adjoint import newton_solve_with_adjoint
from .capabilities import (
    FlowCapabilityReport,
    flow_evolution_capabilities,
    flow_evolution_input_schema,
    validate_flow_evolution_input,
)
from .config import FlowExecutionConfig, FlowHistoryConfig
from .errors import FlowCapabilityError, FlowContractError, FlowConvergenceError
from .history import FlowHistory, FlowHistoryWriter
from .resources import (
    FlowResourceEstimate,
    FlowResourceRequest,
    enforce_flow_resource_budget,
    estimate_flow_resources,
)
from .solvers.diagnostics import normalize_convergence_diagnostics
from .solvers import (
    BiCGSTABSolver,
    DirectSolver,
    GMRESSolver,
    SparseDirectSolver,
)
from .models.black_oil import BlackOilModel
from .models.oil_water import OilWaterModel
from .models.single_phase import SinglePhaseModel
from .models._base import (
    FlowModel,
)
from .solvers import NewtonResult, NewtonSolver
from .timestep import AdaptiveTimeStepper, TimeStepScheduler
from .wells import BHPControl, WellGroup
from geobrain.physics.flow._operator_support import (  # noqa: F401  re-export: split section
    _LinearSolver,
    _ModelAdapter,
    _PhaseAdapter,
    _PressureAdapter,
    _ReplaySegment,
    _ResidualAdapter,
    _SparseJacobianModel,
    _control_snapshot,
    _data_keys_for_model,
    _fixed_dt_schedule,
    _jacobian_with_wells,
    _model_sparsity_spec,
    _pack_state,
    _plain_control,
    _require_newton_state,
    _residual_with_wells,
    _resource_request_for_model,
    _snapshot_well_group,
    _trainable_execution_tensors,
    _typed_well_adapter,
    _unpack_state,
)




# ---------------------------------------------------------------------------
# FlowStep: the single well-coupled residual/Jacobian of one implicit step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowStep:
    """Residual and Jacobian of one implicit reservoir time step.

    The single source of truth for the flow Newton step, bound to a model and an
    optional :class:`~geobrain.physics.flow.wells.WellGroup`. The differentiable
    march (:class:`FlowEvolutionOperator` / :class:`TransientFlowOperator`, via
    the implicit-function-theorem adjoint) and hand-rolled *forward* time loops
    (the visual demos) both build a ``FlowStep`` and call the same
    ``residual`` / ``jacobian``, so there is exactly one well-coupled
    residual/Jacobian in the codebase, and a fix to it reaches every caller.

    Args:
        model: the flow model (single-phase / oil-water / black-oil).
        wells: optional well group whose BHP/rate source terms are folded into
            the residual (``None`` ⇒ no wells). Its ``n_cells`` auto-attaches.
    """

    model: FlowModel
    wells: WellGroup | None = None

    def __post_init__(self) -> None:
        if self.wells is not None and self.wells.n_cells is None:
            # Mutating the (mutable) well group, not the frozen FlowStep.
            self.wells.n_cells = self.model.n_cells

    def residual(self, state: torch.Tensor, state_old: torch.Tensor, dt: float) -> torch.Tensor:
        """Well-coupled mass-balance residual ``R(state)`` for one step."""
        return _residual_with_wells(self.model, state, state_old, dt, self.wells)

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        *,
        exact: bool = False,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """``∂R/∂state``. ``exact=True`` forces the exact autograd path on the
        sparse Jacobian (for the IFT-adjoint cleanup); forward Newton iterations
        use the cheaper default (colored-FD on the sparse path, dense autograd
        otherwise)."""
        return _jacobian_with_wells(
            self.model,
            state,
            state_old,
            dt,
            self.wells,
            exact=exact,
            create_graph=create_graph,
        )

    def advance(
        self,
        state_old: torch.Tensor,
        dt: float,
        newton: NewtonSolver,
    ) -> NewtonResult:
        """Advance one step *forward* (no autograd) with the given Newton solver.

        The forward-only counterpart of :class:`FlowEvolutionOperator` (which
        wraps the same step in the IFT adjoint). Returns the solver's
        ``NewtonResult`` so the caller can feed the iteration count back into an
        adaptive stepper and snapshot at report boundaries, the orchestration
        the visual demos own while sharing this one residual/Jacobian.
        """
        with torch.no_grad():
            return newton.solve(
                residual_fn=lambda x: self.residual(x, state_old, dt),
                jacobian_fn=lambda x: self.jacobian(x, state_old, dt),
                state0=state_old,
            )


# ---------------------------------------------------------------------------
# FlowEvolutionOperator: single Newton time step
# ---------------------------------------------------------------------------


















class FlowEvolutionOperator(ForwardOperator):  # type: ignore[misc]  # core is skipped at this strict-check boundary.
    """
    Single Newton-step advance of a flow state.

    Inputs (ModelState):
        - ``pressure``                   (single / oil-water / black-oil)
        - ``sw``                         (oil-water + black-oil)
        - ``sg``                         (black-oil only)

    ForwardContext:
        - ``ctx.time.dt``: timestep [s]
        - optional ``ctx.flow.wells``: :class:`WellGroup`

    Outputs (ForwardOutput.data):
        - same keys as inputs, advanced one timestep
        - Newton diagnostics are returned under ``ForwardOutput.metadata``.

    Args:
        model: the flow model providing residual and Jacobian blocks.
        config: :class:`FlowExecutionConfig` execution policy.
        newton: optional pre-configured Newton solver override.
    """

    @classmethod
    def input_schema(cls) -> Mapping[str, object]:
        """Return a fresh strict JSON schema for Agent and UI clients."""
        schema: Mapping[str, object] = flow_evolution_input_schema()
        return schema

    @classmethod
    def validate_input(cls, payload: Mapping[str, object]) -> None:
        """Enforce the schema's cross-array Agent runtime constraints."""
        validate_flow_evolution_input(payload)

    @classmethod
    def capabilities(cls) -> FlowCapabilityReport:
        """Return immutable support metadata without executing a model."""
        return flow_evolution_capabilities()

    def __init__(
        self,
        model: FlowModel,
        *,
        config: FlowExecutionConfig,
        newton: NewtonSolver | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, FlowExecutionConfig):
            raise FlowCapabilityError(
                "FlowEvolutionOperator requires an explicit execution config",
                object_name=type(self).__name__,
                field="config",
                expected=FlowExecutionConfig,
                actual=type(config),
            )
        self.model = model
        self.config = config
        if newton is not None:
            raise FlowCapabilityError(
                "injected NewtonSolver is unsupported; execution uses the exact "
                "built-in solver configuration declared by FlowExecutionConfig",
                object_name=type(self).__name__,
                field="newton",
                expected=None,
                actual=type(newton).__name__,
            )
        self.newton = self._build_newton(config)
        level = {
            "full": DifferentiabilityLevel.FULL_AUTOGRAD,
            "implicit": DifferentiabilityLevel.IMPLICIT_VJP,
            "detached": DifferentiabilityLevel.FORWARD_ONLY,
        }[config.autograd_mode]
        # Refresh the spec's trainable / output keys to match the model.
        self.differentiability = DifferentiabilitySpec(
            level=level,
            trainable_inputs=_data_keys_for_model(model),
            output_keys=_data_keys_for_model(model),
        )

    @staticmethod
    def _build_newton(config: FlowExecutionConfig) -> NewtonSolver:
        """Build the nonlinear solver selected by one immutable config."""
        linear_solver: _LinearSolver
        if config.linear_solver == "gmres":
            linear_solver = GMRESSolver()
        elif config.linear_solver == "bicgstab":
            linear_solver = BiCGSTABSolver()
        elif config.linear_solver == "dense_direct":
            # Explicit dense-direct selection is the graph-safe small-problem path.
            linear_solver = DirectSolver()
        else:
            linear_solver = SparseDirectSolver()
        nonlinear = config.nonlinear
        return NewtonSolver(
            linear_solver=linear_solver,
            tol=nonlinear.residual_tolerance,
            tol_rel=nonlinear.update_tolerance,
            max_iter=nonlinear.max_iterations,
            line_search_max_halvings=nonlinear.line_search_max_iterations,
        )

    @staticmethod
    def _validate_injected_newton(
        config: FlowExecutionConfig,
        newton: NewtonSolver,
    ) -> None:
        """Keep the immutable execution config authoritative over injection."""
        if type(newton) is not NewtonSolver:
            raise FlowCapabilityError(
                "injected NewtonSolver must use the exact built-in implementation",
                object_name="FlowEvolutionOperator",
                field="newton",
                expected=NewtonSolver,
                actual=type(newton),
            )
        expected_solver = {
            "dense_direct": DirectSolver,
            "sparse_direct": SparseDirectSolver,
            "gmres": GMRESSolver,
            "bicgstab": BiCGSTABSolver,
        }[config.linear_solver]
        if type(newton.linear_solver) is not expected_solver:
            raise FlowCapabilityError(
                "injected NewtonSolver must use the exact built-in declared linear solver",
                object_name="FlowEvolutionOperator",
                field="newton.linear_solver",
                expected=expected_solver.__name__,
                actual=type(newton.linear_solver).__name__,
            )
        reference_newton = FlowEvolutionOperator._build_newton(config)
        reference_solver = reference_newton.linear_solver
        parameter_names = tuple(
            name
            for name in ("tol", "max_iter", "restart")
            if hasattr(reference_solver, name)
        )

        def typed_value(value: object) -> tuple[str, object]:
            return (type(value).__qualname__, value)

        def exact_value(actual: object, expected: object) -> bool:
            return type(actual) is type(expected) and actual == expected

        def preconditioner_signature(solver: object) -> tuple[object, ...] | None:
            preconditioner = getattr(solver, "precond", None)
            if preconditioner is None:
                return None
            return (
                type(preconditioner).__qualname__,
                typed_value(getattr(preconditioner, "eps", None)),
                "setup" in vars(preconditioner),
                "apply" in vars(preconditioner),
            )

        actual_linear = (
            tuple(
                typed_value(getattr(newton.linear_solver, name))
                for name in parameter_names
            ),
            preconditioner_signature(newton.linear_solver),
            "solve" in vars(newton.linear_solver),
            tuple(sorted(set(vars(newton.linear_solver)) - {"last_stats"})),
        )
        expected_linear = (
            tuple(typed_value(getattr(reference_solver, name)) for name in parameter_names),
            preconditioner_signature(reference_solver),
            False,
            tuple(sorted(set(vars(reference_solver)) - {"last_stats"})),
        )
        actual_preconditioner = getattr(newton.linear_solver, "precond", None)
        expected_preconditioner = getattr(reference_solver, "precond", None)
        if actual_preconditioner is None or expected_preconditioner is None:
            preconditioner_matches = actual_preconditioner is expected_preconditioner
        else:
            preconditioner_matches = (
                type(actual_preconditioner) is type(expected_preconditioner)
                and exact_value(
                    getattr(actual_preconditioner, "eps", None),
                    getattr(expected_preconditioner, "eps", None),
                )
                and "setup" not in vars(actual_preconditioner)
                and "apply" not in vars(actual_preconditioner)
                and set(vars(actual_preconditioner)) == set(vars(expected_preconditioner))
            )
        linear_values_match = all(
            exact_value(
                getattr(newton.linear_solver, name),
                getattr(reference_solver, name),
            )
            for name in parameter_names
        )
        linear_instance_matches = (
            "solve" not in vars(newton.linear_solver)
            and set(vars(newton.linear_solver)) - {"last_stats"}
            == set(vars(reference_solver)) - {"last_stats"}
        )
        if not (
            linear_values_match and preconditioner_matches and linear_instance_matches
        ):
            raise FlowCapabilityError(
                "execution requires the exact built-in solver configuration",
                object_name="FlowEvolutionOperator",
                field="newton.linear_solver.configuration",
                expected=expected_linear,
                actual=actual_linear,
            )
        newton_parameter_names = (
            "max_iter",
            "tol",
            "tol_rel",
            "line_search_max_halvings",
            "line_search",
            "keep_jacobian",
            "verbose",
        )
        actual_newton = (
            tuple(typed_value(getattr(newton, name)) for name in newton_parameter_names),
            tuple(sorted(vars(newton))),
        )
        expected_newton = (
            tuple(
                typed_value(getattr(reference_newton, name))
                for name in newton_parameter_names
            ),
            tuple(sorted(vars(reference_newton))),
        )
        newton_values_match = all(
            exact_value(getattr(newton, name), getattr(reference_newton, name))
            for name in newton_parameter_names
        )
        newton_instance_matches = set(vars(newton)) == set(vars(reference_newton))
        if not (newton_values_match and newton_instance_matches):
            raise FlowCapabilityError(
                "execution requires the exact built-in Newton configuration",
                object_name="FlowEvolutionOperator",
                field="newton.configuration",
                expected=expected_newton,
                actual=actual_newton,
            )

    def _validate_execution(self) -> None:
        """Reject unsupported graph/device/layout combinations before solve."""
        # ``newton`` remains inspectable for diagnostics, so revalidate it on
        # every call. This closes post-construction replacement of the solver
        # with a graph-breaking subclass while metadata still declares full
        # autograd.
        self._validate_injected_newton(self.config, self.newton)
        mode = self.config.autograd_mode
        linear_solver = self.config.linear_solver
        sparse_model = getattr(self.model, "_sparsity_spec", None) is not None
        grid = getattr(self.model, "grid", None)
        device = torch.device(getattr(grid, "device", "cpu"))
        if device.type not in ("cpu", "cuda"):
            raise FlowCapabilityError(
                "Flow execution is accepted on CPU and CUDA only",
                object_name=type(self).__name__,
                field="device",
                expected=("cpu", "cuda"),
                actual=str(device),
            )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise FlowCapabilityError(
                "CUDA Flow execution requires an available CUDA runtime",
                object_name=type(self).__name__,
                field="device",
                expected="available CUDA device",
                actual=str(device),
            )
        if self.model.schema.unit_system == "FIELD" and device.type != "cpu":
            raise FlowCapabilityError(
                "experimental FIELD/TPFA execution is accepted on CPU only",
                object_name=type(self).__name__,
                field="device",
                expected="cpu for FIELD/TPFA models",
                actual=str(device),
            )
        if mode == "full" and linear_solver != "dense_direct":
            raise FlowCapabilityError(
                "full autograd requires the explicit graph-safe dense direct solver",
                object_name=type(self).__name__,
                field="linear_solver",
                expected="dense_direct for autograd_mode='full'",
                actual=linear_solver,
            )
        if linear_solver == "dense_direct" and sparse_model:
            raise FlowCapabilityError(
                "dense direct execution is unavailable for a sparse-declared model",
                object_name=type(self).__name__,
                field="linear_solver",
                expected="sparse_direct/gmres/bicgstab for sparse models",
                actual=linear_solver,
            )
        if linear_solver == "sparse_direct" and not sparse_model:
            raise FlowCapabilityError(
                "sparse direct execution requires an explicit sparse Jacobian pattern",
                object_name=type(self).__name__,
                field="linear_solver",
                expected="dense_direct for dense models or enable sparse Jacobian",
                actual=linear_solver,
            )
        if mode == "full" and getattr(self.model, "_sparsity_spec", None) is not None:
            raise FlowCapabilityError(
                "full autograd does not support the detached colored sparse Jacobian",
                object_name=type(self).__name__,
                field="autograd_mode",
                expected="implicit/detached for sparse TPFA, or a dense model for full",
                actual={"mode": mode, "model": type(self.model).__name__},
            )

    def resource_request(
        self,
        *,
        accepted_step_bound: int = 1,
        wells: int = 0,
    ) -> FlowResourceRequest:
        """Return an allocation-free resource request for this operator."""
        return _resource_request_for_model(
            self.model,
            self.config,
            accepted_step_bound=accepted_step_bound,
            wells=wells,
        )

    def estimate_resources(
        self,
        request: FlowResourceRequest | None = None,
    ) -> FlowResourceEstimate:
        """Return the deterministic component-wise execution estimate."""
        return estimate_flow_resources(self.resource_request() if request is None else request)

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        dt = ctx.require_dt()
        wells: WellGroup | None = ctx.flow.wells if ctx.flow else None
        if (
            wells is not None
            and len(wells)
            and not isinstance(self.model, (SinglePhaseModel, OilWaterModel, BlackOilModel))
        ):
            raise FlowCapabilityError(
                "typed well coupling is unavailable for this Flow model",
                object_name=type(self).__name__,
                field="wells",
                expected="no wells, or a TPFA single/oil-water/black-oil model",
                actual=type(self.model).__name__,
            )
        # Execution capability errors take precedence over resource accounting:
        # an impossible mode/solver/model combination has no valid allocation
        # request to estimate.
        self._validate_execution()
        request = self.resource_request(
            wells=0 if wells is None else len(wells),
        )
        estimate = enforce_flow_resource_budget(request, self.config.resource_budget_bytes)
        # FlowStep is the single source of truth for the well-coupled
        # residual/Jacobian (it auto-attaches the well group's n_cells); the
        # forward demos build the same FlowStep and drive it forward-only.
        step = FlowStep(self.model, wells)
        state_old = _pack_state(self.model, state)
        mode = self.config.autograd_mode
        if mode == "implicit":
            # Forward runs without an iteration graph; one converged-residual
            # cleanup installs the exact implicit-function VJP.
            x_star, res = newton_solve_with_adjoint(
                residual_fn=lambda x: step.residual(x, state_old, dt),
                jacobian_fn=lambda x: step.jacobian(x, state_old, dt),
                state0=state_old,
                newton_solver=self.newton,
                adjoint_jacobian_fn=lambda x: step.jacobian(x, state_old, dt, exact=True),
            )
        elif mode == "full":
            # Preserve every accepted Newton operation in the caller-selected
            # autograd graph, including the exact Jacobian construction.
            res = self.newton.solve(
                residual_fn=lambda x: step.residual(x, state_old, dt),
                jacobian_fn=lambda x: step.jacobian(
                    x, state_old, dt, exact=True, create_graph=True
                ),
                state0=state_old,
            )
            x_star = _require_newton_state(res)
        else:
            # Detachment is intentional and declared by the config, never a
            # hidden fallback from a requested differentiable mode.
            with torch.no_grad():
                res = self.newton.solve(
                    residual_fn=lambda x: step.residual(x, state_old, dt),
                    jacobian_fn=lambda x: step.jacobian(x, state_old, dt),
                    state0=state_old,
                )
            x_star = _require_newton_state(res).detach()
        mask_reader = getattr(self.model, "accepted_discrete_masks", None)
        accepted_masks: Mapping[str, tuple[bool, ...]] = {}
        if callable(mask_reader):
            with torch.no_grad():
                accepted_masks = mask_reader(x_star.detach())
        split = _unpack_state(self.model, x_star)
        data = {k: split[k] for k in _data_keys_for_model(self.model)}
        return ForwardOutput(
            data=data,
            fields={"state": x_star},
            metadata={
                "iterations": res.iterations,
                "residual_norm": res.residual_norm,
                "autograd_mode": mode,
                "linear_solver": self.config.linear_solver,
                "accepted_discrete_masks": dict(accepted_masks),
                "resource_estimate": estimate.to_dict(),
            },
        )


# ---------------------------------------------------------------------------
# WellObservationOperator
# ---------------------------------------------------------------------------


class WellObservationOperator(ForwardOperator):  # type: ignore[misc]  # core is skipped at this strict-check boundary.
    """
    Extract per-well BHP and per-phase rates from a flow state.

    Inputs (ModelState): same state keys consumed by the underlying
    :class:`FlowEvolutionOperator`.

    ForwardContext:
        - ``ctx.flow.wells``: the same :class:`WellGroup` that
          was used to advance the state.

    Outputs (ForwardOutput.data):
        - ``bhp_pa``: ``(n_wells,)`` bottom-hole pressure [Pa].
        - ``oil_surface_m3_s`` / ``water_surface_m3_s``: declared-standard
          liquid rates [m³/s].
        - ``gas_standard_m3_s``: declared-standard gas rate [m³/s].
        - ``reservoir_m3_s``: in-situ total reservoir rate [m³/s].

    Rate signs follow the canonical source convention: positive injection and
    negative production. Rate-controlled BHP is an augmented unknown owned by
    :class:`WellSystem`; this state-only observer rejects that request instead
    of reconstructing BHP with the removed FIELD ``ALPHA`` formula.

    All outputs are autograd-differentiable through the state.
    """

    def __init__(
        self,
        model: FlowModel,
    ) -> None:
        super().__init__()
        self.model = model
        self.differentiability = DifferentiabilitySpec(
            level=DifferentiabilityLevel.FULL_AUTOGRAD,
            trainable_inputs=_data_keys_for_model(model),
            output_keys=(
                "bhp_pa",
                "oil_surface_m3_s",
                "water_surface_m3_s",
                "gas_standard_m3_s",
                "reservoir_m3_s",
            ),
        )

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        wells: WellGroup = ctx.require_wells()
        if wells.n_cells is None:
            wells.n_cells = self.model.n_cells
        flat = _pack_state(self.model, state)
        pressure_pa, mobility_fn, density_fn, _, _ = _typed_well_adapter(self.model)
        p_cells = pressure_pa(flat)
        mobs = mobility_fn(flat)
        densities = density_fn(flat)
        if any(not isinstance(well.control, BHPControl) for well in wells.wells):
            raise FlowCapabilityError(
                "rate-controlled BHP requires the augmented WellSystem state",
                object_name=type(self).__name__,
                field="wells.control",
                expected="BHPControl or an explicit WellSystem BHP observation",
                actual="RateControl",
            )
        reports = wells.compute_rate_reports(p_cells, mobs, densities)
        signs = p_cells.new_tensor(
            [1.0 if well.well_type == "INJ" else -1.0 for well in wells.wells]
        )
        bhp = p_cells.new_tensor(
            [cast(BHPControl, well.control).pressure_pa for well in wells.wells]
        )

        def signed(field_name: str) -> torch.Tensor:
            return torch.stack([getattr(report, field_name) for report in reports]) * signs

        return ForwardOutput(
            data={
                "bhp_pa": bhp,
                "oil_surface_m3_s": signed("oil_surface_m3_s"),
                "water_surface_m3_s": signed("water_surface_m3_s"),
                "gas_standard_m3_s": signed("gas_standard_m3_s"),
                "reservoir_m3_s": signed("reservoir_m3_s"),
            },
            metadata={"well_names": [w.name for w in wells.wells]},
        )


# ---------------------------------------------------------------------------
# TransientFlowOperator: full time march
# ---------------------------------------------------------------------------


@dataclass
class _TimeMarchStats:
    """Diagnostics from a transient simulation."""

    n_steps: int = 0
    n_newton_iters_total: int = 0
    n_dt_cuts: int = 0
    final_time: float = 0.0
    dt_history: list[float] = field(default_factory=list)


class TransientFlowOperator(ForwardOperator):  # type: ignore[misc]  # core is skipped at this strict-check boundary.
    """
    Full transient time march with adaptive ``dt``.

    Inputs (ModelState): same as :class:`FlowEvolutionOperator`.

    ForwardContext:
        - ``ctx.time.t_end``: total simulation time [s].
        - ``ctx.time.scheduler``: :class:`TimeStepScheduler` or
          :class:`AdaptiveTimeStepper`. Required.
        - optional ``ctx.flow.wells``: :class:`WellGroup`.
        - optional ``ctx.flow.well_observer``,
          :class:`WellObservationOperator` to capture per-step well
          rates / BHP.
        - optional ``ctx.flow.return_states = True`` to keep all
          intermediate states (memory-heavy).

    Outputs (ForwardOutput.data):
        - ``final_pressure`` (always present)
        - ``final_sw`` / ``final_sg`` (only when applicable)
        - if ``well_observer`` is provided: canonical SI ``bhp_pa_series`` and
          typed oil/water/gas/reservoir rate series, each ``(n_steps, n_wells)``.
        - ``time_axis``: ``(n_steps,)`` end-of-step times in seconds.

    Gradient scope, honestly: the march differentiates w.r.t. the STATE-key
    trainables (pressure / saturations). Constitutive-module parameters
    (permeability, porosity, PVT coefficients) are NOT reached through this
    operator's adjoint, invert them through
    :class:`ParametricFlowOperator`, which lifts them into ``ModelState``
    and installs the implicit-function-theorem link.
    """

    def __init__(
        self,
        evolution: FlowEvolutionOperator,
    ) -> None:
        super().__init__()
        self.evolution = evolution
        # output_keys declares everything this operator can emit into ForwardOutput.data:
        # the final-state channels plus the per-step well series (produced only when a
        # ``well_observer`` is supplied at call time, but declared so InverseProblem
        # accepts them as history-matching observations). ``time_axis`` is an axis, not
        # an observation, so it lives in metadata rather than data.
        self.differentiability = DifferentiabilitySpec(
            level=evolution.differentiability.level,
            trainable_inputs=_data_keys_for_model(evolution.model),
            output_keys=tuple(f"final_{k}" for k in _data_keys_for_model(evolution.model))
            + (
                "bhp_pa_series",
                "oil_surface_m3_s_series",
                "water_surface_m3_s_series",
                "gas_standard_m3_s_series",
                "reservoir_m3_s_series",
            ),
        )

    def resource_request(
        self,
        *,
        accepted_step_bound: int,
        wells: int = 0,
    ) -> FlowResourceRequest:
        """Return the complete transient request before state allocation."""
        return self.evolution.resource_request(
            accepted_step_bound=accepted_step_bound,
            wells=wells,
        )

    def estimate_resources(
        self,
        request: FlowResourceRequest,
    ) -> FlowResourceEstimate:
        """Return component-wise bytes/work for a declared transient bound."""
        return estimate_flow_resources(request)

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        t_end: float = ctx.require_t_end()
        scheduler = ctx.require_scheduler()
        wells: WellGroup | None = ctx.flow.wells if ctx.flow else None
        well_obs: WellObservationOperator | None = ctx.flow.well_observer if ctx.flow else None

        model = self.evolution.model
        if wells is not None and wells.n_cells is None:
            wells.n_cells = model.n_cells
        wells = _snapshot_well_group(wells)
        control_snapshot = _control_snapshot(wells)

        state_keys = _data_keys_for_model(model)

        report_times = self.evolution.config.history.report_times_s
        outside_window = tuple(report for report in report_times if report > t_end + 1e-12)
        if outside_window:
            raise FlowContractError(
                "report_times_s must lie inside the requested transient window",
                object_name=type(self).__name__,
                field="report_times_s",
                expected=f"0 <= report <= {t_end}",
                actual=outside_window,
            )
        if isinstance(scheduler, TimeStepScheduler):
            fixed_schedule = list(
                _fixed_dt_schedule(
                    scheduler.steps(),
                    t_end=t_end,
                    report_times_s=report_times,
                )
            )
            accepted_step_bound = len(fixed_schedule)
        elif isinstance(scheduler, AdaptiveTimeStepper):
            accepted_step_bound = math.ceil(t_end / scheduler.dt_min) + len(report_times)
        else:
            raise FlowContractError(
                "scheduler must be TimeStepScheduler or AdaptiveTimeStepper",
                object_name=type(self).__name__,
                field="scheduler",
                expected="TimeStepScheduler or AdaptiveTimeStepper",
                actual=type(scheduler).__name__,
            )
        request = self.resource_request(
            accepted_step_bound=accepted_step_bound,
            wells=0 if wells is None else len(wells),
        )
        estimate = enforce_flow_resource_budget(
            request, self.evolution.config.resource_budget_bytes
        )

        # Initial state is packed only after the global history/Jacobian budget
        # has passed. The writer preserves graph ownership selected by config.
        flat = _pack_state(model, state)
        cur_state = state
        history_writer = FlowHistoryWriter(
            self.evolution.config.history,
            accepted_step_bound=accepted_step_bound,
        )
        history_writer.record_initial(time_s=0.0, state=state.tensors)
        accounting_payload: dict[str, int] = {
            "accepted_steps": 0,
            "rejected_steps": 0,
            "residual_evaluations": 0,
            "jacobian_assemblies": 0,
            "linear_solves": 0,
            "recomputed_steps": 0,
            "retained_state_bytes": 0,
        }

        t = 0.0
        stats = _TimeMarchStats()
        observation_keys = (
            "bhp_pa",
            "oil_surface_m3_s",
            "water_surface_m3_s",
            "gas_standard_m3_s",
            "reservoir_m3_s",
        )

        def build_output(
            *,
            require_complete_reports: bool = True,
        ) -> ForwardOutput:
            """Assemble the march result from whatever has been accumulated so far.

            Called for the normal return AND (on a dt-min stall) to attach the accepted
            states to the raised error as ``.partial_result``, so an ill-posed run that
            stalls after N accepted steps does not lose them (mirrors the bayes samplers)."""
            data: dict[str, torch.Tensor] = {}
            for k in state_keys:
                data[f"final_{k}"] = cur_state.tensors[k]
            history: FlowHistory = history_writer.finalize(
                require_complete_reports=require_complete_reports
            )
            accounting_payload.update(history.accounting.to_dict())
            if well_obs is not None:
                for key in observation_keys:
                    retained_values = [
                        retained[f"__observation__{key}"]
                        for retained in history.states
                        if f"__observation__{key}" in retained
                    ]
                    if retained_values:
                        data[f"{key}_series"] = torch.stack(retained_values, dim=0)
            fields: dict[str, torch.Tensor] = {}
            if self.evolution.config.history.mode != "final" and history.states:
                for k in state_keys:
                    fields[f"{k}_series"] = torch.stack(
                        [retained[k] for retained in history.states], dim=0
                    )
            return ForwardOutput(
                data=data,
                fields=fields,
                metadata={
                    "n_steps": stats.n_steps,
                    "n_newton_iters_total": stats.n_newton_iters_total,
                    "n_dt_cuts": stats.n_dt_cuts,
                    "final_time": stats.final_time,
                    "dt_history": list(history.accepted_dt_s),
                    "history_mode": self.evolution.config.history.mode,
                    "retained_step_indices": history.retained_step_indices,
                    "history_times_s": history.times_s,
                    "execution_accounting": accounting_payload,
                    "checkpoint_steps": tuple(
                        checkpoint.accepted_step for checkpoint in history.checkpoints
                    ),
                    "checkpoint_control_schedule": tuple(
                        tuple(_plain_control(control) for control in checkpoint.control_schedule)
                        for checkpoint in history.checkpoints
                    ),
                    "resource_estimate": estimate.to_dict(),
                    # Retained time axis is bounded by the declared policy.
                    "time_axis": (
                        torch.tensor(
                            history.times_s,
                            dtype=flat.dtype,
                            device=flat.device,
                        )
                        if history.times_s
                        else None
                    ),
                },
            )

        # Resolve scheduler into a (next_dt, on_step_result) interface.
        if isinstance(scheduler, TimeStepScheduler):
            schedule_iter = iter(fixed_schedule)
            pending_dt: float | None = None

            def next_dt(remaining: float) -> float | None:
                nonlocal pending_dt
                if pending_dt is None:
                    try:
                        pending_dt = float(next(schedule_iter))
                    except StopIteration:
                        return None
                raw = pending_dt
                dt = min(raw, remaining)
                next_report = next(
                    (value for value in report_times if value > t + 1e-12),
                    None,
                )
                if next_report is not None:
                    dt = min(dt, next_report - t)
                pending_dt = raw - dt
                if pending_dt <= 1e-12:
                    pending_dt = None
                return dt

            def report(iterations: int, converged: bool) -> None:
                return None
        elif isinstance(scheduler, AdaptiveTimeStepper):

            def next_dt(remaining: float) -> float | None:
                if remaining <= 0:
                    return None
                dt = min(float(scheduler.dt), remaining)
                next_report = next(
                    (value for value in report_times if value > t + 1e-12),
                    None,
                )
                return min(dt, next_report - t) if next_report is not None else dt

            def report(iterations: int, converged: bool) -> None:
                scheduler.update(iterations=iterations, converged=converged)

        checkpointed_backward = (
            self.evolution.config.history.mode in {"checkpoint", "recompute"}
            and self.evolution.config.autograd_mode != "detached"
        )

        def checkpointed_march(schedule: tuple[float, ...]) -> ForwardOutput:
            """Execute accepted steps in bounded non-reentrant segments."""
            nonlocal cur_state, t
            mode = self.evolution.config.history.mode
            if mode == "checkpoint":
                segment_span = self.evolution.config.history.checkpoint_interval
            else:
                segment_span = max(
                    1,
                    math.ceil(len(schedule) / self.evolution.config.history.recompute_segments),
                )

            for segment_start in range(0, len(schedule), segment_span):
                segment_dts = schedule[segment_start : segment_start + segment_span]
                start_state = cur_state
                forward_records: list[tuple[float, ModelState, Mapping[str, object]]] = []

                def run_segment(
                    *tensors: torch.Tensor,
                    _segment_dts: tuple[float, ...] = tuple(segment_dts),
                ) -> tuple[torch.Tensor, ...]:
                    local_state = ModelState(
                        tensors=dict(
                            zip(
                                state_keys,
                                tensors[: len(state_keys)],
                                strict=True,
                            )
                        )
                    )
                    replaying = torch.is_grad_enabled()
                    replay_steps = 0
                    replay_residuals = 0
                    replay_jacobians = 0
                    replay_solves = 0
                    for step_dt in _segment_dts:
                        prediction = self.evolution._forward(
                            local_state,
                            ForwardContext.of(dt=step_dt, wells=wells),
                        )
                        local_state = ModelState(tensors=dict(prediction.data))
                        iterations = int(prediction.metadata["iterations"])
                        if replaying:
                            replay_steps += 1
                            replay_residuals += iterations + 1
                            replay_jacobians += iterations
                            replay_solves += iterations
                        else:
                            forward_records.append((step_dt, local_state, prediction.metadata))
                    if replaying:
                        history_writer.record_recomputed(
                            replay_steps,
                            residual_evaluations=replay_residuals,
                            jacobian_assemblies=replay_jacobians,
                            linear_solves=replay_solves,
                        )
                        accounting_payload.update(
                            history_writer.finalize(
                                require_complete_reports=False
                            ).accounting.to_dict()
                        )
                    return tuple(local_state.tensors[key] for key in state_keys)

                inputs = tuple(start_state.tensors[key] for key in state_keys)
                outputs = _ReplaySegment.apply(
                    run_segment,
                    len(state_keys),
                    *inputs,
                    *_trainable_execution_tensors(model),
                )
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                cur_state = ModelState(tensors=dict(zip(state_keys, outputs, strict=True)))
                if len(forward_records) != len(segment_dts):
                    raise FlowContractError(
                        "checkpoint segment did not execute its declared schedule",
                        object_name=type(self).__name__,
                        field="history",
                        expected=len(segment_dts),
                        actual=len(forward_records),
                    )
                for local_index, (step_dt, local_state, metadata) in enumerate(forward_records):
                    is_segment_final = local_index == len(forward_records) - 1
                    accepted_state = cur_state if is_segment_final else local_state
                    iterations = int(
                        cast(int | str | bytes | bytearray, metadata["iterations"])
                    )
                    stats.n_steps += 1
                    stats.n_newton_iters_total += iterations
                    stats.dt_history.append(step_dt)
                    t += step_dt
                    stats.final_time = t
                    retained_state: dict[str, torch.Tensor] = dict(accepted_state.tensors)
                    if well_obs is not None and wells is not None:
                        observation = well_obs._forward(
                            accepted_state,
                            ForwardContext.of(wells=wells),
                        )
                        for key in observation_keys:
                            retained_state[f"__observation__{key}"] = observation.data[key]
                    history_writer.record_accepted(
                        time_s=t,
                        state=retained_state,
                        dt_s=step_dt,
                        control=control_snapshot,
                        residual_evaluations=iterations + 1,
                        jacobian_assemblies=iterations,
                        linear_solves=iterations,
                    )
                forward_records.clear()
            return build_output()

        if checkpointed_backward:
            if isinstance(scheduler, TimeStepScheduler):
                return checkpointed_march(tuple(fixed_schedule))

            # Adaptive step selection is a discrete forward decision. Discover
            # its accepted schedule without a graph, then replay that exact
            # schedule in bounded differentiable segments.
            probe_state = ModelState(
                tensors={key: value.detach() for key, value in cur_state.tensors.items()}
            )
            probe_time = 0.0
            adaptive_schedule: list[float] = []
            consecutive_cuts = 0
            while probe_time < t_end - 1e-12:
                remaining = t_end - probe_time
                probe_dt = min(scheduler.dt, remaining)
                next_report = next(
                    (value for value in report_times if value > probe_time + 1e-12),
                    None,
                )
                if next_report is not None:
                    probe_dt = min(probe_dt, next_report - probe_time)
                try:
                    with torch.no_grad():
                        probe_prediction = self.evolution._forward(
                            probe_state,
                            ForwardContext.of(dt=probe_dt, wells=wells),
                        )
                except GeoBrainError as exc:
                    stats.n_dt_cuts += 1
                    consecutive_cuts += 1
                    normalized = normalize_convergence_diagnostics(
                        getattr(exc, "diagnostics", None),
                        fallback_max_iterations=self.evolution.newton.max_iter,
                    )
                    history_writer.record_rejected(
                        normalized,
                        residual_evaluations=normalized.iterations + 1,
                        jacobian_assemblies=normalized.iterations,
                        linear_solves=normalized.iterations,
                    )
                    scheduler.update(
                        iterations=scheduler.target_iter + 1,
                        converged=False,
                    )
                    if scheduler.dt <= scheduler.dt_min or consecutive_cuts > 25:
                        raise FlowConvergenceError(
                            "checkpoint schedule discovery exhausted retries",
                            object_name=type(self).__name__,
                            field="dt",
                            expected="convergence above dt_min",
                            actual=scheduler.dt,
                            diagnostics=normalized,
                        ) from exc
                    continue
                iterations = int(probe_prediction.metadata["iterations"])
                history_writer.record_recomputed(
                    1,
                    residual_evaluations=iterations + 1,
                    jacobian_assemblies=iterations,
                    linear_solves=iterations,
                )
                adaptive_schedule.append(probe_dt)
                probe_time += probe_dt
                consecutive_cuts = 0
                probe_state = ModelState(
                    tensors={key: value.detach() for key, value in probe_prediction.data.items()}
                )
                scheduler.update(iterations=iterations, converged=True)
            history_writer.accepted_step_bound = len(adaptive_schedule)
            return checkpointed_march(tuple(adaptive_schedule))

        while t < t_end - 1e-12:
            remaining = t_end - t
            dt = next_dt(remaining)
            if dt is None:
                break
            step_ctx = ForwardContext.of(dt=dt, wells=wells)
            try:
                pred = self.evolution._forward(cur_state, step_ctx)
            except GeoBrainError as exc:
                # Newton failed: only meaningful to retry under adaptive scheduling.
                if isinstance(scheduler, AdaptiveTimeStepper):
                    stats.n_dt_cuts += 1
                    raw_diagnostics = getattr(exc, "diagnostics", None)
                    normalized = normalize_convergence_diagnostics(
                        raw_diagnostics,
                        fallback_max_iterations=self.evolution.newton.max_iter,
                    )
                    history_writer.record_rejected(
                        normalized,
                        residual_evaluations=normalized.iterations + 1,
                        jacobian_assemblies=normalized.iterations,
                        linear_solves=normalized.iterations,
                    )
                    report(iterations=scheduler.target_iter + 1, converged=False)
                    if scheduler.dt <= scheduler.dt_min:
                        err = GeoBrainError(
                            f"TransientFlowOperator stalled at t={t:.3f}d, "
                            f"dt={scheduler.dt:.3e} (min={scheduler.dt_min:.3e})",
                            object_name="TransientFlowOperator",
                            field="dt",
                            expected="≥ dt_min after cut",
                            actual=scheduler.dt,
                        )
                        err.partial_result = build_output(require_complete_reports=False)
                        raise err from exc
                    continue
                raise

            stats.n_steps += 1
            stats.n_newton_iters_total += int(pred.metadata["iterations"])
            stats.dt_history.append(dt)
            t += dt
            stats.final_time = t
            # Wrap returned data back into ModelState for the next iteration.
            cur_state = ModelState(tensors=dict(pred.data))
            retained_state: dict[str, torch.Tensor] = dict(cur_state.tensors)
            if well_obs is not None and wells is not None:
                obs_pred = well_obs._forward(
                    cur_state,
                    ForwardContext.of(wells=wells),
                )
                for key in observation_keys:
                    retained_state[f"__observation__{key}"] = obs_pred.data[key]
            history_writer.record_accepted(
                time_s=t,
                state=retained_state,
                dt_s=dt,
                control=control_snapshot,
                residual_evaluations=int(pred.metadata["iterations"]) + 1,
                jacobian_assemblies=int(pred.metadata["iterations"]),
                linear_solves=int(pred.metadata["iterations"]),
            )
            report(iterations=int(pred.metadata["iterations"]), converged=True)

        return build_output()


# ---------------------------------------------------------------------------
# ParametricFlowOperator: model properties as trainable inputs
# ---------------------------------------------------------------------------


class ParametricFlowOperator(ForwardOperator):  # type: ignore[misc]  # core is skipped at this strict-check boundary.
    """Transient flow march whose trainable inputs are MODEL PROPERTIES
    (permeability / porosity / ...), not the dynamic state.

    The flow family is unusual among the physics families: the transient
    operators' ``ModelState`` carries the DYNAMIC state (initial pressure /
    saturation), while the actual inversion parameters (``rock.permeability_m2``,
    ``rock.porosity``, ...) are constructor-bound inside the flow model, so
    a bare :class:`TransientFlowOperator` can never receive gradients on them
    through its input surface, and cannot join
    :class:`~geobrain.inverse.InverseProblem` /
    :class:`~geobrain.inverse.joint.JointProblem` wire-by-name for parameter
    inversion. This adapter (the in-repo 5-spot pattern, e.g.
    ``examples/08_flow/11_perm_inversion_5spot.py``, made a library citizen)
    closes that gap: each forward call reads the named property tensors from
    the ``ModelState``, REBUILDS the flow model around the live tensors via
    the caller's ``model_builder``, and runs the standard
    :class:`TransientFlowOperator` (+ optional
    :class:`WellObservationOperator`) march. Gradients then flow to the
    properties through the implicit-function-theorem adjoint installed by
    :class:`FlowEvolutionOperator` (pinned by the coupled
    flow–rock gradient suite).

    The march configuration (initial state, schedule, wells, execution
    policy) is deliberately CONSTRUCTOR-BOUND, and the ambient per-call
    :class:`~geobrain.core.context.ForwardContext` is ignored: in a
    ``JointProblem`` the shared ctx typically belongs to another physics
    (e.g. a seismic mesh), and flow's non-trainable inputs are problem
    constants, not per-call knobs. The execution policy is bound through
    ``config=``: the same :class:`FlowExecutionConfig` the bare
    :class:`FlowEvolutionOperator` takes, so an inversion driven through
    this adapter can select the sparse solver and the history retention it
    needs instead of being stuck on the defaults.

    ``JointProblem`` specifics:

    - ``param_names`` become the operator's ``trainable_inputs``: name your
      ``EarthModel`` field/link the same (e.g. a ``"perm"`` link off a
      trainable porosity) and the term wires by name.
    - ``model_builder`` receives the tensors EXACTLY as resolved (e.g. mesh-
      shaped ``(nz, nx)``); reshape to the flow model's flat ``(n_cells,)``
      inside the builder, or adapt with ``JointForward(field_to_mesh=...)``.
    - With ``observe_wells=True`` the march emits multiple channels: select
      one with ``JointForward(op, output="bhp_series")``. With
      ``observe_wells=False`` a single-phase model emits exactly
      ``("final_pressure",)`` and the bare-operator path applies.
    - The flow adjoint's sparse path is CPU-only by design. A caller holding
      GPU-resident model fields must make the transfer explicit with a
      differentiable ``JointForward(field_to_mesh=...)`` adapter; context
      overrides never move or cast tensors.

    Args:
        model_builder: ``(**{name: Tensor}) -> FlowModel``, rebuilds the
            flow model around the live property tensors; called once per
            forward with exactly ``param_names`` as keyword arguments.
        param_names: Property names lifted into the ``ModelState`` (the
            operator's ``trainable_inputs``), e.g. ``("perm",)``.
        initial_state: ``{state_variable: Tensor}``, the (non-trainable)
            initial dynamic state; its keys must match the built model's
            state variables (checked per call, since the model does not
            exist before).
        t_end: Total simulation time [s].
        scheduler: :class:`~geobrain.physics.flow.timestep.TimeStepScheduler`
            or :class:`~geobrain.physics.flow.timestep.AdaptiveTimeStepper`.
            Prefer the stateless ``TimeStepScheduler`` inside inversion
            loops, an ``AdaptiveTimeStepper`` mutates its ``dt`` across
            calls, so successive loss evaluations would see different
            schedules.
        wells: Optional :class:`~geobrain.physics.flow.wells.WellGroup`
            folded into the residual.
        observe_wells: When ``True`` (requires ``wells``), a
            :class:`WellObservationOperator` captures the per-step
            ``bhp_series`` / ``q_*_series`` channels.
        config: :class:`FlowExecutionConfig` for the inner march, solver,
            autograd mode, history retention and resource budget. Omit it
            and the march runs the platform default, EXCEPT that
            ``observe_wells=True`` additionally retains every step
            (``history=FlowHistoryConfig(mode="all")``), because the
            ``*_series`` channels are otherwise a series of length one.
            A config passed here is used exactly as given and never
            adjusted, so a caller who wants only the final well state can
            still ask for ``mode="final"``. The declared
            :attr:`differentiability` level follows ``autograd_mode``, the
            same way :class:`FlowEvolutionOperator` does it.
    """

    def __init__(
        self,
        model_builder: Callable[..., FlowModel],
        *,
        param_names: tuple[str, ...],
        initial_state: Mapping[str, torch.Tensor],
        t_end: float,
        scheduler: TimeStepScheduler | AdaptiveTimeStepper,
        wells: WellGroup | None = None,
        observe_wells: bool = False,
        config: FlowExecutionConfig | None = None,
    ) -> None:
        super().__init__()
        if not callable(model_builder):
            raise GeoBrainError(
                "ParametricFlowOperator model_builder must be callable",
                object_name="ParametricFlowOperator",
                field="model_builder",
                expected="callable(**params) -> FlowModel",
                actual=type(model_builder),
            )
        param_names = tuple(param_names)
        if not param_names or not all(isinstance(n, str) and n for n in param_names):
            raise GeoBrainError(
                "ParametricFlowOperator param_names must be a non-empty tuple of non-empty strings",
                object_name="ParametricFlowOperator",
                field="param_names",
                expected="e.g. ('perm',)",
                actual=param_names,
            )
        if not isinstance(initial_state, Mapping) or not initial_state:
            raise GeoBrainError(
                "ParametricFlowOperator initial_state must be a non-empty "
                "{state_variable: Tensor} mapping",
                object_name="ParametricFlowOperator",
                field="initial_state",
                expected="non-empty Mapping[str, Tensor]",
                actual=initial_state,
            )
        for k, v in initial_state.items():
            if not isinstance(v, torch.Tensor):
                raise GeoBrainError(
                    "ParametricFlowOperator initial_state values must be torch.Tensor",
                    object_name="ParametricFlowOperator",
                    field=f"initial_state[{k!r}]",
                    expected=torch.Tensor,
                    actual=type(v),
                )
        overlap = set(param_names) & set(initial_state)
        if overlap:
            raise GeoBrainError(
                "ParametricFlowOperator param_names must not overlap "
                "initial_state keys (a name is either a trainable property "
                "or a dynamic state variable, never both)",
                object_name="ParametricFlowOperator",
                field="param_names",
                expected="disjoint from initial_state keys",
                actual=sorted(overlap),
            )
        if observe_wells and wells is None:
            raise GeoBrainError(
                "ParametricFlowOperator observe_wells=True requires wells=",
                object_name="ParametricFlowOperator",
                field="observe_wells",
                expected="a WellGroup when observe_wells=True",
                actual=None,
            )

        self.model_builder = model_builder
        self.param_names = param_names
        self.initial_state = dict(initial_state)
        self.t_end = float(t_end)
        self.scheduler = scheduler
        self.wells = wells
        self.observe_wells = observe_wells
        if config is None:
            # A well observer with the default 'final' history returns a
            # one-step "series", which is never what a caller asking for
            # per-step rates wants. Only the DEFAULT is adjusted; an
            # explicit config is honoured exactly.
            config = (FlowExecutionConfig(history=FlowHistoryConfig(mode="all"))
                      if observe_wells else FlowExecutionConfig())
        self.config = config

        output_keys = tuple(f"final_{k}" for k in self.initial_state)
        if observe_wells:
            output_keys += (
                "bhp_pa_series",
                "oil_surface_m3_s_series",
                "water_surface_m3_s_series",
                "gas_standard_m3_s_series",
                "reservoir_m3_s_series",
            )
        self.differentiability = DifferentiabilitySpec(
            level={
                "full": DifferentiabilityLevel.FULL_AUTOGRAD,
                "implicit": DifferentiabilityLevel.IMPLICIT_VJP,
                "detached": DifferentiabilityLevel.FORWARD_ONLY,
            }[self.config.autograd_mode],
            trainable_inputs=param_names,
            output_keys=output_keys,
        )

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        params = {name: state.tensors[name] for name in self.param_names}
        model = self.model_builder(**params)
        state_keys = _data_keys_for_model(model)
        if set(state_keys) != set(self.initial_state):
            raise GeoBrainError(
                "ParametricFlowOperator initial_state keys must match the "
                "built model's state variables",
                object_name="ParametricFlowOperator",
                field="initial_state",
                expected=sorted(state_keys),
                actual=sorted(self.initial_state),
            )
        march = TransientFlowOperator(FlowEvolutionOperator(model, config=self.config))
        observer = WellObservationOperator(model) if self.observe_wells else None
        inner_ctx = ForwardContext.of(
            t_end=self.t_end,
            scheduler=self.scheduler,
            wells=self.wells,
            well_observer=observer,
        )
        return march(ModelState(tensors=dict(self.initial_state)), inner_ctx)


__all__ = [
    "FlowEvolutionOperator",
    "ParametricFlowOperator",
    "TransientFlowOperator",
    "WellObservationOperator",
]
