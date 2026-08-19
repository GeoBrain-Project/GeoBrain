"""Deterministic pre-allocation resource estimates for Flow.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

from .config import FlowHistoryConfig
from .errors import FlowContractError, FlowResourceError

FLOW_RESOURCE_SCHEMA_VERSION = "geobrain.flow.resource/1.0"

_DTYPE_BYTES = {"float32": 4, "float64": 8}
_DEVICES = frozenset({"cpu", "cuda", "mps"})
_AUTOGRAD_MODES = frozenset({"full", "implicit", "detached"})
_LINEAR_SOLVERS = frozenset({"dense_direct", "sparse_direct", "gmres", "bicgstab"})
_SPARSE_LAYOUTS = frozenset({"coo", "csr"})
_JACOBIAN_LAYOUTS = frozenset({"dense", *_SPARSE_LAYOUTS})


def _integer(value: object, *, field: str, positive: bool = False) -> int:
    lower = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        raise FlowContractError(
            f"{field} must be an integer >= {lower}",
            object_name="FlowResourceRequest",
            field=field,
            expected=f"integer >= {lower}",
            actual=value,
        )
    return value


@dataclass(frozen=True, slots=True)
class FlowResourceRequest:
    """Closed, allocation-free description of one requested Flow run.

    ``jacobian_nnz`` describes the reservoir-only operator. A non-zero
    ``bordered_jacobian_nnz`` is the total stored nnz across all four
    :class:`SparseBorderedJacobian` blocks (including the reservoir block).
    Sparse estimates account for the peak where those blocks and the assembled
    matrix coexist. Dense estimates always use the full augmented matrix shape;
    structural nnz cannot reduce an allocated dense payload.

    Attributes:
        schema_version: request schema tag.
        cells / faces / primary_dofs / residual_blocks: problem sizes.
        stencil_nnz / jacobian_nnz / bordered_jacobian_nnz: sparsity sizes.
        wells / perforations: well-system sizes.
        accepted_step_bound: bound on accepted time steps.
        history: retention mode requested.
        dtype / device: execution placement.
        autograd_mode / linear_solver: execution policy axes.
        stencil_layout / jacobian_layout: sparse layout tags.
        nonlinear_iteration_bound / line_search_iteration_bound: Newton
            work bounds.
    """

    schema_version: str
    cells: int
    faces: int
    primary_dofs: int
    residual_blocks: int
    stencil_nnz: int
    jacobian_nnz: int
    wells: int
    accepted_step_bound: int
    history: FlowHistoryConfig
    dtype: str
    device: str
    autograd_mode: Literal["full", "implicit", "detached"]
    linear_solver: Literal["dense_direct", "sparse_direct", "gmres", "bicgstab"]
    stencil_layout: Literal["coo", "csr"] = "coo"
    jacobian_layout: Literal["dense", "coo", "csr"] | None = None
    bordered_jacobian_nnz: int = 0
    perforations: int = 0
    nonlinear_iteration_bound: int = 12
    line_search_iteration_bound: int = 8

    def __post_init__(self) -> None:
        if self.schema_version != FLOW_RESOURCE_SCHEMA_VERSION:
            raise FlowContractError(
                "unsupported Flow resource schema",
                object_name=type(self).__name__,
                field="schema_version",
                expected=FLOW_RESOURCE_SCHEMA_VERSION,
                actual=self.schema_version,
            )
        for field_name in (
            "cells",
            "primary_dofs",
            "residual_blocks",
            "nonlinear_iteration_bound",
            "line_search_iteration_bound",
        ):
            object.__setattr__(
                self,
                field_name,
                _integer(getattr(self, field_name), field=field_name, positive=True),
            )
        for field_name in (
            "faces",
            "stencil_nnz",
            "jacobian_nnz",
            "wells",
            "accepted_step_bound",
            "bordered_jacobian_nnz",
            "perforations",
        ):
            object.__setattr__(
                self,
                field_name,
                _integer(getattr(self, field_name), field=field_name),
            )
        if not isinstance(self.history, FlowHistoryConfig):
            raise FlowContractError(
                "history must be a FlowHistoryConfig",
                object_name=type(self).__name__,
                field="history",
                expected=FlowHistoryConfig,
                actual=type(self.history),
            )
        if self.dtype not in _DTYPE_BYTES:
            raise FlowContractError(
                "unsupported Flow resource dtype",
                object_name=type(self).__name__,
                field="dtype",
                expected=sorted(_DTYPE_BYTES),
                actual=self.dtype,
            )
        if self.device not in _DEVICES:
            raise FlowContractError(
                "unsupported Flow resource device",
                object_name=type(self).__name__,
                field="device",
                expected=sorted(_DEVICES),
                actual=self.device,
            )
        if self.autograd_mode not in _AUTOGRAD_MODES:
            raise FlowContractError(
                "unsupported Flow autograd mode",
                object_name=type(self).__name__,
                field="autograd_mode",
                expected=sorted(_AUTOGRAD_MODES),
                actual=self.autograd_mode,
            )
        if self.linear_solver not in _LINEAR_SOLVERS:
            raise FlowContractError(
                "unsupported Flow linear solver",
                object_name=type(self).__name__,
                field="linear_solver",
                expected=sorted(_LINEAR_SOLVERS),
                actual=self.linear_solver,
            )
        if self.stencil_layout not in _SPARSE_LAYOUTS:
            raise FlowContractError(
                "unsupported Flow stencil storage layout",
                object_name=type(self).__name__,
                field="stencil_layout",
                expected=sorted(_SPARSE_LAYOUTS),
                actual=self.stencil_layout,
            )
        inferred_jacobian_layout = "dense" if self.linear_solver == "dense_direct" else "coo"
        if self.jacobian_layout is None:
            object.__setattr__(self, "jacobian_layout", inferred_jacobian_layout)
        elif self.jacobian_layout not in _JACOBIAN_LAYOUTS:
            raise FlowContractError(
                "unsupported Flow Jacobian storage layout",
                object_name=type(self).__name__,
                field="jacobian_layout",
                expected=sorted(_JACOBIAN_LAYOUTS),
                actual=self.jacobian_layout,
            )
        if self.linear_solver == "dense_direct" and self.jacobian_layout != "dense":
            raise FlowContractError(
                "dense direct Flow resources require dense Jacobian storage",
                object_name=type(self).__name__,
                field="jacobian_layout",
                expected="dense",
                actual=self.jacobian_layout,
            )
        if self.linear_solver == "sparse_direct" and self.jacobian_layout == "dense":
            raise FlowContractError(
                "sparse direct Flow resources require sparse Jacobian storage",
                object_name=type(self).__name__,
                field="jacobian_layout",
                expected=sorted(_SPARSE_LAYOUTS),
                actual=self.jacobian_layout,
            )
        if self.device != "cpu" and self.linear_solver == "sparse_direct":
            raise FlowContractError(
                "sparse direct Flow resources are CPU-only",
                object_name=type(self).__name__,
                field="device",
                expected="cpu for sparse_direct",
                actual=self.device,
            )
        if self.bordered_jacobian_nnz and self.bordered_jacobian_nnz < self.jacobian_nnz:
            raise FlowContractError(
                "bordered Jacobian nnz must include the reservoir block",
                object_name=type(self).__name__,
                field="bordered_jacobian_nnz",
                expected=f">= jacobian_nnz ({self.jacobian_nnz})",
                actual=self.bordered_jacobian_nnz,
            )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic strict-JSON-ready request."""
        return {
            "schema_version": self.schema_version,
            "cells": self.cells,
            "faces": self.faces,
            "primary_dofs": self.primary_dofs,
            "residual_blocks": self.residual_blocks,
            "stencil_nnz": self.stencil_nnz,
            "jacobian_nnz": self.jacobian_nnz,
            "wells": self.wells,
            "accepted_step_bound": self.accepted_step_bound,
            "history": {
                "mode": self.history.mode,
                "report_times_s": list(self.history.report_times_s),
                "checkpoint_interval": self.history.checkpoint_interval,
                "recompute_segments": self.history.recompute_segments,
            },
            "dtype": self.dtype,
            "device": self.device,
            "autograd_mode": self.autograd_mode,
            "linear_solver": self.linear_solver,
            "stencil_layout": self.stencil_layout,
            "jacobian_layout": self.jacobian_layout,
            "bordered_jacobian_nnz": self.bordered_jacobian_nnz,
            "perforations": self.perforations,
            "nonlinear_iteration_bound": self.nonlinear_iteration_bound,
            "line_search_iteration_bound": self.line_search_iteration_bound,
        }


@dataclass(frozen=True, slots=True)
class FlowResourceEstimate:
    """Component-wise conservative memory and work estimate.

    Attributes:
        schema_version: estimate schema tag.
        model_property_bytes / topology_stencil_bytes / jacobian_bytes /
            linear_workspace_bytes / live_state_bytes /
            retained_history_bytes / checkpoint_bytes / output_bytes /
            autograd_bytes: per-category peak-memory terms.
        total_bytes: sum of the categories above.
        residual_evaluations / linear_solves / recomputed_steps: predicted
            work counters.
    """

    schema_version: str
    model_property_bytes: int
    topology_stencil_bytes: int
    jacobian_bytes: int
    linear_workspace_bytes: int
    live_state_bytes: int
    retained_history_bytes: int
    checkpoint_bytes: int
    output_bytes: int
    autograd_bytes: int
    total_bytes: int
    residual_evaluations: int
    linear_solves: int
    recomputed_steps: int

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic strict-JSON-ready estimate."""
        return asdict(self)


def _checkpoint_count(request: FlowResourceRequest) -> int:
    steps = request.accepted_step_bound
    if steps == 0:
        return 1
    if request.history.mode == "checkpoint":
        interval = int(request.history.checkpoint_interval)
        return 2 + (steps - 1) // interval
    if request.history.mode == "recompute":
        return min(steps + 1, int(request.history.recompute_segments) + 1)
    return 0


def _sparse_storage_bytes(
    *,
    rows: int,
    nnz: int,
    scalar_bytes: int,
    layout: Literal["coo", "csr"],
) -> int:
    """Exact PyTorch int64-index/value payload for one sparse matrix."""
    index_bytes = 8
    if layout == "coo":
        return nnz * (scalar_bytes + 2 * index_bytes)
    return nnz * (scalar_bytes + index_bytes) + (rows + 1) * index_bytes


def _bordered_sparse_peak_bytes(
    request: FlowResourceRequest,
    *,
    scalar_bytes: int,
    layout: Literal["coo", "csr"],
) -> int:
    """Peak payload while four bordered blocks and their assembly coexist."""
    nnz = request.bordered_jacobian_nnz
    assembled_rows = request.primary_dofs + request.wells
    assembled_coo = _sparse_storage_bytes(
        rows=assembled_rows,
        nnz=nnz,
        scalar_bytes=scalar_bytes,
        layout="coo",
    )
    if layout == "coo":
        blocks = nnz * (scalar_bytes + 2 * 8)
        return blocks + assembled_coo

    # SparseBorderedJacobian.assemble() converts every retained CSR block to
    # COO and returns COO. At peak the original CSR blocks, all converted COO
    # blocks, and the assembled COO matrix coexist.
    block_row_pointers = (2 * request.primary_dofs + 2 * request.wells + 4) * 8
    csr_blocks = nnz * (scalar_bytes + 8) + block_row_pointers
    converted_coo_blocks = nnz * (scalar_bytes + 2 * 8)
    return csr_blocks + converted_coo_blocks + assembled_coo


def estimate_flow_resources(request: FlowResourceRequest) -> FlowResourceEstimate:
    """Estimate all large components without allocating tensors or stencils."""
    if not isinstance(request, FlowResourceRequest):
        raise FlowContractError(
            "estimate_flow_resources requires FlowResourceRequest",
            object_name="estimate_flow_resources",
            field="request",
            expected=FlowResourceRequest,
            actual=type(request),
        )
    scalar = _DTYPE_BYTES[request.dtype]
    index = 8
    state_bytes = request.primary_dofs * scalar
    linear_dofs = request.primary_dofs + request.wells
    linear_state_bytes = linear_dofs * scalar
    steps = request.accepted_step_bound

    model_property_bytes = request.cells * (request.residual_blocks + 3) * scalar
    topology_stencil_bytes = (
        request.faces * 2 * index
        + _sparse_storage_bytes(
            rows=request.faces,
            nnz=request.stencil_nnz,
            scalar_bytes=scalar,
            layout=request.stencil_layout,
        )
        + request.perforations * (index + 2 * scalar)
    )
    total_jacobian_nnz = (
        request.bordered_jacobian_nnz
        if request.bordered_jacobian_nnz
        else request.jacobian_nnz
    )
    jacobian_layout = request.jacobian_layout
    assert jacobian_layout is not None  # resolved by FlowResourceRequest
    if jacobian_layout == "dense":
        # Dense storage pays for every entry in the augmented reservoir/well
        # matrix; structural nnz cannot reduce the allocated payload.
        augmented_dofs = request.primary_dofs + request.wells
        jacobian_bytes = augmented_dofs * augmented_dofs * scalar
    elif request.bordered_jacobian_nnz:
        jacobian_bytes = _bordered_sparse_peak_bytes(
            request,
            scalar_bytes=scalar,
            layout=jacobian_layout,
        )
    else:
        jacobian_bytes = _sparse_storage_bytes(
            rows=request.primary_dofs + request.wells,
            nnz=total_jacobian_nnz,
            scalar_bytes=scalar,
            layout=jacobian_layout,
        )
    if request.linear_solver == "gmres":
        # GMRES(50) retains the Arnoldi basis plus the Hessenberg matrix and
        # reduced-QR temporaries. Include extra full-size work vectors for
        # residuals, preconditioning, updates, and matvec intermediates.
        # The current core implementation allocates V[n, 51] and H[51, 50]
        # before Arnoldi can discover a smaller invariant subspace, even when
        # n < 50. The resource contract therefore uses the configured default
        # restart literally rather than shrinking it to the system dimension.
        restart = 50
        vector_scalars = (restart + 9) * linear_dofs
        dense_qr_scalars = 4 * (restart + 1) * restart
        linear_workspace_bytes = (vector_scalars + dense_qr_scalars) * scalar
    elif request.linear_solver == "bicgstab":
        linear_workspace_bytes = 8 * linear_state_bytes
    elif request.linear_solver == "sparse_direct":
        # Conservative sparse-LU factor/fill budget; exact factors are backend-
        # and ordering-dependent, so the declared estimate uses a fixed ceiling.
        linear_workspace_bytes = 6 * jacobian_bytes + 3 * linear_state_bytes
    else:
        linear_workspace_bytes = 2 * linear_dofs * linear_state_bytes
    live_state_bytes = 4 * linear_state_bytes

    mode = request.history.mode
    if mode == "all":
        retained_count = steps + 1
    elif mode == "report":
        retained_count = min(steps + 1, len(request.history.report_times_s) + 2)
    elif mode == "checkpoint":
        retained_count = _checkpoint_count(request)
    elif mode == "recompute":
        retained_count = _checkpoint_count(request)
    else:
        retained_count = 1
    # A well observer records five canonical SI scalars per well and retained
    # step. The request cannot distinguish observed from unobserved runs, so
    # budgeting assumes the larger observed case.
    observation_bytes = 5 * request.wells * scalar
    retained_record_bytes = state_bytes + observation_bytes
    retained_history_bytes = retained_count * retained_record_bytes

    checkpoints = _checkpoint_count(request)
    # The final output state is shared with the last checkpoint; all preceding
    # checkpoint states are additional live allocations. This convention keeps
    # checkpoint/recompute comparisons independent of how many states are also
    # exposed as report history.
    checkpoint_state_count = max(0, checkpoints - 1)
    # Eight bytes per accepted dt plus a compact two-word entry per well/control.
    checkpoint_bytes = (
        checkpoint_state_count * state_bytes + steps * 8 + steps * request.wells * 2 * scalar
        if checkpoints
        else 0
    )
    # The final state is always present. Observations and the retained time
    # axis are stacked into output allocations; non-final policies also expose
    # a stacked state series.
    output_bytes = state_bytes + retained_count * (observation_bytes + scalar)
    if mode != "final":
        output_bytes += retained_count * state_bytes

    if mode == "checkpoint":
        graph_step_bound = min(steps, request.history.checkpoint_interval)
    elif mode == "recompute":
        graph_step_bound = (
            0 if steps == 0 else math.ceil(steps / request.history.recompute_segments)
        )
    else:
        graph_step_bound = steps
    if request.autograd_mode == "full":
        autograd_bytes = graph_step_bound * (jacobian_bytes + 3 * linear_state_bytes)
    elif request.autograd_mode == "implicit":
        autograd_bytes = graph_step_bound * (jacobian_bytes + 2 * linear_state_bytes)
    else:
        autograd_bytes = 0

    recomputed_steps = (
        2 * steps
        if mode in {"checkpoint", "recompute"} and request.autograd_mode != "detached"
        else 0
    )
    executed_step_bound = steps + recomputed_steps
    linear_solves = executed_step_bound * request.nonlinear_iteration_bound
    residual_evaluations = executed_step_bound * (
        1
        + request.nonlinear_iteration_bound
        * (1 + request.line_search_iteration_bound)
    )
    components = (
        model_property_bytes,
        topology_stencil_bytes,
        jacobian_bytes,
        linear_workspace_bytes,
        live_state_bytes,
        retained_history_bytes,
        checkpoint_bytes,
        output_bytes,
        autograd_bytes,
    )
    return FlowResourceEstimate(
        schema_version=FLOW_RESOURCE_SCHEMA_VERSION,
        model_property_bytes=model_property_bytes,
        topology_stencil_bytes=topology_stencil_bytes,
        jacobian_bytes=jacobian_bytes,
        linear_workspace_bytes=linear_workspace_bytes,
        live_state_bytes=live_state_bytes,
        retained_history_bytes=retained_history_bytes,
        checkpoint_bytes=checkpoint_bytes,
        output_bytes=output_bytes,
        autograd_bytes=autograd_bytes,
        total_bytes=sum(components),
        residual_evaluations=residual_evaluations,
        linear_solves=linear_solves,
        recomputed_steps=recomputed_steps,
    )


def enforce_flow_resource_budget(
    request: FlowResourceRequest,
    resource_budget_bytes: int | None,
) -> FlowResourceEstimate:
    """Estimate then reject an insufficient budget before caller allocation."""
    estimate = estimate_flow_resources(request)
    if resource_budget_bytes is None:
        return estimate
    if (
        isinstance(resource_budget_bytes, bool)
        or not isinstance(resource_budget_bytes, int)
        or resource_budget_bytes <= 0
    ):
        raise FlowContractError(
            "resource budget must be a positive integer",
            object_name="enforce_flow_resource_budget",
            field="resource_budget_bytes",
            expected="positive integer byte count",
            actual=resource_budget_bytes,
        )
    if estimate.total_bytes > resource_budget_bytes:
        raise FlowResourceError(
            "Flow resource estimate exceeds the explicit budget",
            object_name="enforce_flow_resource_budget",
            field="resource_budget_bytes",
            expected=f">= {estimate.total_bytes}",
            actual=estimate.total_bytes,
            diagnostics={
                "budget_bytes": resource_budget_bytes,
                "estimate": estimate.to_dict(),
                "request": request.to_dict(),
            },
        )
    return estimate


__all__ = [
    "FLOW_RESOURCE_SCHEMA_VERSION",
    "FlowResourceEstimate",
    "FlowResourceRequest",
    "enforce_flow_resource_budget",
    "estimate_flow_resources",
]
