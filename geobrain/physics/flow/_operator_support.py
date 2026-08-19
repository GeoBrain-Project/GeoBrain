"""Flow operator support: replay autograd segment, the model-to-wells
currency adapters (surface-volume m³/s ↔ phase-mass kg/s), state
pack/unpack, well-control snapshots and schedule helpers.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from __future__ import annotations
import math
from typing import Callable, Literal, Mapping, Protocol, cast, runtime_checkable
import torch
from ...core import (
    ModelState,
)
from .config import FlowExecutionConfig
from .errors import FlowCapabilityError, FlowContractError, FlowConvergenceError
from .resources import (
    FLOW_RESOURCE_SCHEMA_VERSION,
    FlowResourceRequest,
)
from .solvers import (
    JacobianSparsitySpec,
)
from .models.single_phase import SinglePhaseModel
from .models._base import (
    FlowModel,
    pack_state as _model_pack_state,
    state_variables as _model_state_variables,
    unpack_state as _model_unpack_state,
)
from .solvers import NewtonResult
from .wells import BHPControl, FlowSourceTerms, RateControl, WellGroup


class _ReplaySegment(torch.autograd.Function):  # type: ignore[misc]  # torch is skipped at this strict-check boundary.
    """Bounded segment checkpoint compatible with both backward APIs.

    ``torch.utils.checkpoint(..., use_reentrant=True)`` is incompatible with
    :func:`torch.autograd.grad`, while its non-reentrant form can replay during
    the Newton solver's nested Jacobian construction.  This narrow custom
    checkpoint stores only segment inputs and reconstructs the segment exactly
    once when its VJP is requested.
    """

    @staticmethod
    def forward(ctx, run_segment, state_count, *inputs):  # type: ignore[no-untyped-def]
        ctx.run_segment = run_segment
        ctx.state_count = state_count
        ctx.save_for_backward(*inputs)
        return run_segment(*inputs)

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[no-untyped-def]
        saved = ctx.saved_tensors
        needs_grad = ctx.needs_input_grad[2:]
        replay_inputs = tuple(
            tensor.detach().requires_grad_(needed) if index < ctx.state_count else tensor
            for index, (tensor, needed) in enumerate(zip(saved, needs_grad, strict=True))
        )
        targets = tuple(
            tensor for tensor, needed in zip(replay_inputs, needs_grad, strict=True) if needed
        )
        with torch.enable_grad():
            replay_outputs = ctx.run_segment(*replay_inputs)
            if not isinstance(replay_outputs, tuple):
                replay_outputs = (replay_outputs,)
            target_grads = torch.autograd.grad(
                replay_outputs,
                targets,
                grad_outputs,
                allow_unused=True,
            )
        gradients = iter(target_grads)
        input_grads = tuple(next(gradients) if needed else None for needed in needs_grad)
        return None, None, *input_grads


_PressureAdapter = Callable[[torch.Tensor], torch.Tensor]


_PhaseAdapter = Callable[[torch.Tensor], Mapping[str, torch.Tensor]]


_ResidualAdapter = Callable[
    [torch.Tensor, torch.Tensor, float, FlowSourceTerms],
    torch.Tensor,
]


_ModelAdapter = tuple[
    _PressureAdapter,
    _PhaseAdapter,
    _PhaseAdapter,
    _ResidualAdapter,
    int,
]


class _LinearSolver(Protocol):
    """Structural solver contract consumed by :class:`NewtonSolver`."""

    def solve(self, J: torch.Tensor, r: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class _SparseJacobianModel(Protocol):
    """Optional sparse-Jacobian capability exposed by TPFA models."""

    _sparsity_spec: JacobianSparsitySpec | None


def _model_sparsity_spec(model: FlowModel) -> JacobianSparsitySpec | None:
    """Return an explicitly declared sparse pattern, if the model has one."""

    if isinstance(model, _SparseJacobianModel):
        return model._sparsity_spec
    return None


def _pack_state(model: FlowModel, ms: ModelState) -> torch.Tensor:
    """Pack a :class:`ModelState` into the per-variable flat vector consumed by
    ``model.residual``.

    Dispatches structurally through the :class:`~geobrain.physics.flow.models._base.FlowModel`
    contract (state_variables order + ``initial_state``), so it works for every
    model, TPFA, MPFA, thermal, compositional, not just the three TPFA models
    the old isinstance ladder recognised."""
    return cast(torch.Tensor, _model_pack_state(model, ms))


def _unpack_state(model: FlowModel, state: torch.Tensor) -> dict[str, torch.Tensor]:
    """Translate a flat state into operator-facing :class:`ModelState` keys
    (``pressure``/``sw``/``sg``/``temperature``/``composition``) via the model's
    ``state_split`` and the shared internal-key translation table."""
    return _model_unpack_state(model, state)


def _typed_well_adapter(model: FlowModel) -> _ModelAdapter:
    """Return the one model-to-typed-wells currency adapter used by all operators."""

    from .wells.implicit import _WellModelLike, _auto_adapter, _single_phase_adapter

    adapter: _ModelAdapter
    if isinstance(model, SinglePhaseModel):
        adapter = _single_phase_adapter(model)
    else:
        adapter = _auto_adapter(cast(_WellModelLike, model))
    return adapter


def _residual_with_wells(
    model: FlowModel,
    state: torch.Tensor,
    state_old: torch.Tensor,
    dt: float,
    wells: WellGroup | None,
) -> torch.Tensor:
    """Evaluate the model residual with well source rates folded in."""
    if wells is None or len(wells) == 0:
        if isinstance(model, SinglePhaseModel):
            return model.residual(state, state_old, dt)
        return model.residual(state, state_old, dt)

    pressure_pa, mobility_fn, density_fn, residual_fn, _ = _typed_well_adapter(model)
    mobilities = mobility_fn(state)
    densities = density_fn(state)
    sources = wells.compute_source_terms(
        pressure_pa(state),
        mobilities,
        densities,
    )
    return residual_fn(state, state_old, dt, sources)


def _jacobian_with_wells(
    model: FlowModel,
    state: torch.Tensor,
    state_old: torch.Tensor,
    dt: float,
    wells: WellGroup | None,
    *,
    exact: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    """
    Newton Jacobian via the model's own dense-or-sparse path, with
    well sources included (autograd / colored-FD both pick them up).

    ``exact=True`` requests an exact Jacobian for the implicit-FT adjoint
    cleanup: the dense path is already exact (autograd); the sparse
    path switches from colored central-FD to colored reverse-mode autograd.
    The forward Newton iterations call this with ``exact=False`` (cheap FD),
    so only the once-per-solve adjoint Jacobian pays for exactness."""

    def f(x: torch.Tensor) -> torch.Tensor:
        return _residual_with_wells(model, x, state_old, dt, wells)

    sparsity_spec = _model_sparsity_spec(model)
    if sparsity_spec is None:
        return torch.autograd.functional.jacobian(
            f,
            state,
            create_graph=create_graph,
            # Sparse COO source construction has no vmap rule; row-wise AD is
            # exact and preserves the sparse public source representation.
            vectorize=wells is None,
        )
    # Sparse path
    from .solvers import compute_sparse_jacobian

    return compute_sparse_jacobian(
        f,
        state,
        sparsity_spec,
        mode="ad" if exact else "fd",
    )


def _data_keys_for_model(model: FlowModel) -> tuple[str, ...]:
    """The model's ordered :class:`ModelState` variable names: its operator
    input/output channels. Structural (works for any :class:`FlowModel`):
    ``("pressure",)`` single-phase, ``("pressure", "sw", "sg")`` black-oil,
    ``("pressure", "temperature")`` thermal, ``("pressure", "composition")``
    compositional, etc."""
    return _model_state_variables(model)


def _require_newton_state(result: NewtonResult) -> torch.Tensor:
    """Return a solver state, rejecting an invalid result at the boundary."""

    state = result.state
    if state is None:
        raise FlowConvergenceError(
            "Newton solver returned no state",
            object_name="FlowEvolutionOperator",
            field="newton.state",
            expected="a state tensor",
            actual=None,
        )
    return state


def _resource_request_for_model(
    model: FlowModel,
    config: FlowExecutionConfig,
    *,
    accepted_step_bound: int,
    wells: int = 0,
) -> FlowResourceRequest:
    """Describe model topology without constructing a state or Jacobian."""
    grid = getattr(model, "grid", None)
    cells = int(model.n_cells)
    blocks = len(model.schema.residual_blocks)
    dofs = int(model.state_size())
    faces = int(getattr(grid, "n_faces", 0)) if grid is not None else 0
    spec = getattr(model, "_sparsity_spec", None)
    jacobian_layout: Literal["dense", "coo", "csr"]
    if spec is not None:
        jacobian_nnz = int(spec.rows.size)
        jacobian_layout = "coo"
    else:
        # A dense model is an explicit small-problem capability. Its resource
        # request must say N² instead of pretending a sparse pattern exists.
        jacobian_nnz = dofs * dofs
        jacobian_layout = "dense"
    stencil_nnz = max(cells, blocks * blocks * (cells + 2 * faces))
    dtype = getattr(grid, "dtype", torch.float64)
    device = getattr(grid, "device", torch.device("cpu"))
    dtype_name = {
        torch.float32: "float32",
        torch.float64: "float64",
    }.get(dtype)
    if dtype_name is None:
        raise FlowCapabilityError(
            "Flow resource accounting supports float32 and float64",
            object_name=type(model).__name__,
            field="dtype",
            expected=("float32", "float64"),
            actual=str(dtype),
        )
    return FlowResourceRequest(
        schema_version=FLOW_RESOURCE_SCHEMA_VERSION,
        cells=cells,
        faces=faces,
        primary_dofs=dofs,
        residual_blocks=blocks,
        stencil_nnz=stencil_nnz,
        jacobian_nnz=jacobian_nnz,
        wells=wells,
        accepted_step_bound=accepted_step_bound,
        history=config.history,
        dtype=dtype_name,
        device=torch.device(device).type,
        autograd_mode=config.autograd_mode,
        linear_solver=config.linear_solver,
        jacobian_layout=jacobian_layout,
        nonlinear_iteration_bound=config.nonlinear.max_iterations,
        line_search_iteration_bound=config.nonlinear.line_search_max_iterations,
    )


def _snapshot_well_group(wells: WellGroup | None) -> WellGroup | None:
    """Freeze the immutable well records used by forward and replay."""
    if wells is None:
        return None
    return WellGroup(
        wells=list(tuple(wells.wells)),
        n_cells=wells.n_cells,
        device=wells.device,
        dtype=wells.dtype,
    )


def _control_snapshot(wells: WellGroup | None) -> Mapping[str, object]:
    """Return a deterministic, JSON-safe per-step well-control snapshot."""

    def control_record(control: BHPControl | RateControl | None) -> Mapping[str, object] | None:
        if control is None:
            return None
        if isinstance(control, BHPControl):
            return {
                "type": "bhp",
                "pressure_pa": control.pressure_pa,
            }
        if isinstance(control, RateControl):
            return {
                "type": "rate",
                "kind": control.kind.value,
                "target_m3_s": control.target_m3_s,
            }
        raise FlowContractError(
            "unsupported well control in execution schedule",
            object_name="TransientFlowOperator",
            field="well.control",
            expected="BHPControl | RateControl",
            actual=type(control).__name__,
        )

    def sorted_scalars(values: Mapping[str, float] | None) -> Mapping[str, float] | None:
        if values is None:
            return None
        return {key: float(value) for key, value in sorted(values.items())}

    if wells is None:
        return {"wells": ()}
    rows: list[Mapping[str, object]] = []
    for well in wells.wells:
        standard_conditions = well.standard_conditions
        rows.append(
            {
                "name": well.name,
                "well_type": well.well_type,
                "control": control_record(well.control),
                "perforations": tuple(
                    {
                        "cell_idx": perforation.cell_idx,
                        "well_index_m3": perforation.well_index_m3,
                        "depth_offset_m": perforation.depth_offset_m,
                    }
                    for perforation in well.perforations
                ),
                "injection_phase": well.injection_phase,
                "injection_composition": sorted_scalars(well.injection_composition),
                "injection_temperature_k": well.injection_temperature_k,
                "standard_conditions": (
                    None
                    if standard_conditions is None
                    else {
                        "pressure_pa": standard_conditions.pressure_pa,
                        "temperature_k": standard_conditions.temperature_k,
                    }
                ),
                "standard_densities_kg_m3": sorted_scalars(
                    well.standard_densities_kg_m3
                ),
                "bhp_limit_pa": well.bhp_limit_pa,
                "rate_limit": control_record(well.rate_limit),
                "datum_depth_m": well.datum_depth_m,
            }
        )
    return {"wells": tuple(rows)}


def _plain_control(value: object) -> object:
    """Thaw an immutable control snapshot into strict-JSON containers."""
    if isinstance(value, Mapping):
        return {key: _plain_control(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain_control(item) for item in value)
    return value


def _trainable_execution_tensors(model: object) -> tuple[torch.Tensor, ...]:
    """Collect live differentiable model tensors for segment replay."""
    tensors: list[torch.Tensor] = []
    seen_objects: set[int] = set()
    seen_tensors: set[int] = set()

    def visit(value: object) -> None:
        identity = id(value)
        if identity in seen_objects:
            return
        seen_objects.add(identity)
        if isinstance(value, torch.Tensor):
            if value.requires_grad and identity not in seen_tensors:
                tensors.append(value)
                seen_tensors.add(identity)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
            return
        if isinstance(value, torch.nn.Module):
            for parameter in value.parameters(recurse=False):
                visit(parameter)
            for buffer in value.buffers(recurse=False):
                visit(buffer)
            for child in value.children():
                visit(child)
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for name, item in attributes.items():
                if name not in {"_parameters", "_buffers", "_modules"}:
                    visit(item)
        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if isinstance(name, str) and hasattr(value, name):
                visit(getattr(value, name))

    visit(model)
    return tuple(tensors)


def _fixed_dt_schedule(
    raw_schedule: list[float],
    *,
    t_end: float,
    report_times_s: tuple[float, ...],
) -> tuple[float, ...]:
    """Split a fixed schedule exactly at report stations and ``t_end``."""
    accepted: list[float] = []
    time_s = 0.0
    for raw_value in raw_schedule:
        raw = float(raw_value)
        if not math.isfinite(raw) or raw <= 0.0:
            raise FlowContractError(
                "fixed timesteps must be finite and positive",
                object_name="TransientFlowOperator",
                field="scheduler.dt_list",
                expected="finite seconds > 0",
                actual=raw_value,
            )
        remaining_raw = raw
        while remaining_raw > 1e-12 and time_s < t_end - 1e-12:
            dt = min(remaining_raw, t_end - time_s)
            next_report = next(
                (report for report in report_times_s if report > time_s + 1e-12),
                None,
            )
            if next_report is not None:
                dt = min(dt, next_report - time_s)
            accepted.append(dt)
            time_s += dt
            remaining_raw -= dt
        if time_s >= t_end - 1e-12:
            break
    if not math.isclose(time_s, t_end, rel_tol=1e-12, abs_tol=1e-12):
        raise FlowContractError(
            "fixed timestep schedule does not reach t_end",
            object_name="TransientFlowOperator",
            field="scheduler.dt_list",
            expected=f"sum covering {t_end} seconds",
            actual=time_s,
        )
    return tuple(accepted)

