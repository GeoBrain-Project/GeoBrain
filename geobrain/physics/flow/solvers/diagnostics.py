"""Machine-readable convergence diagnostics shared by all Flow solve stages.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


_SCHEMA_VERSION = "geobrain.flow.convergence/1.0"


def _json_number(value: float) -> float | str:
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    return number


@dataclass(frozen=True, slots=True)
class FlowConvergenceDiagnostics:
    """Closed diagnostic record for flash, nonlinear, and linear solves.

    Attributes:
        schema_version: record schema tag.
        stage: which solve stage the record describes.
        converged / reason: outcome and its cause.
        iterations / max_iterations: Newton accounting.
        initial_residual_norm / residual_norm / residual_history: norms.
        cnv / mb: per-block convergence and mass-balance measures.
        failed_cells: cells violating tolerances.
        time_s / step_index: simulated time and step position.
    """

    schema_version: str
    stage: Literal["flash", "nonlinear", "linear"]
    converged: bool
    reason: Literal[
        "tolerance",
        "max_iterations",
        "line_search",
        "breakdown",
        "nonfinite",
        "invalid_state",
    ]
    iterations: int
    max_iterations: int
    initial_residual_norm: float
    residual_norm: float
    residual_history: tuple[float, ...]
    cnv: tuple[float, ...] = ()
    mb: tuple[float, ...] = ()
    failed_cells: tuple[int, ...] = ()
    time_s: float | None = None
    step_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic strict-JSON-safe record."""
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "converged": self.converged,
            "reason": self.reason,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "initial_residual_norm": _json_number(self.initial_residual_norm),
            "residual_norm": _json_number(self.residual_norm),
            "residual_history": [_json_number(value) for value in self.residual_history],
            "cnv": [_json_number(value) for value in self.cnv],
            "mb": [_json_number(value) for value in self.mb],
            "failed_cells": list(self.failed_cells),
            "time_s": None if self.time_s is None else _json_number(self.time_s),
            "step_index": self.step_index,
        }


def convergence_diagnostics(
    *,
    stage: Literal["flash", "nonlinear", "linear"],
    converged: bool,
    reason: Literal[
        "tolerance",
        "max_iterations",
        "line_search",
        "breakdown",
        "nonfinite",
        "invalid_state",
    ],
    iterations: int,
    max_iterations: int,
    initial_residual_norm: float,
    residual_norm: float,
    residual_history: tuple[float, ...] | list[float],
    cnv: tuple[float, ...] = (),
    mb: tuple[float, ...] = (),
    failed_cells: tuple[int, ...] = (),
    time_s: float | None = None,
    step_index: int | None = None,
) -> FlowConvergenceDiagnostics:
    """Build the canonical v1.0 diagnostic record."""
    return FlowConvergenceDiagnostics(
        schema_version=_SCHEMA_VERSION,
        stage=stage,
        converged=converged,
        reason=reason,
        iterations=int(iterations),
        max_iterations=int(max_iterations),
        initial_residual_norm=float(initial_residual_norm),
        residual_norm=float(residual_norm),
        residual_history=tuple(float(value) for value in residual_history),
        cnv=tuple(float(value) for value in cnv),
        mb=tuple(float(value) for value in mb),
        failed_cells=tuple(int(value) for value in failed_cells),
        time_s=None if time_s is None else float(time_s),
        step_index=None if step_index is None else int(step_index),
    )


def normalize_convergence_diagnostics(
    diagnostics: object,
    *,
    fallback_stage: Literal["flash", "nonlinear", "linear"] = "nonlinear",
    fallback_max_iterations: int = 0,
) -> FlowConvergenceDiagnostics:
    """Convert an older stage-specific record to the canonical Flow schema.

    GeoBrain 0.2 flash records predate the shared solver diagnostic type and
    expose per-cell iteration counts.  Retry controllers must not assume those
    records support :func:`dataclasses.replace`; normalize them at the boundary
    while retaining the scientific fields represented by the common schema.
    """
    if isinstance(diagnostics, FlowConvergenceDiagnostics):
        return diagnostics

    stage = getattr(diagnostics, "stage", fallback_stage)
    if stage not in {"flash", "nonlinear", "linear"}:
        stage = fallback_stage

    raw_iterations = getattr(diagnostics, "iterations", 0)
    if isinstance(raw_iterations, (tuple, list)):
        iterations = max((int(value) for value in raw_iterations), default=0)
    else:
        try:
            iterations = int(raw_iterations)
        except (TypeError, ValueError, OverflowError):
            iterations = 0
    try:
        max_iterations = int(
            getattr(diagnostics, "max_iterations", fallback_max_iterations)
        )
    except (TypeError, ValueError, OverflowError):
        max_iterations = int(fallback_max_iterations)
    raw_failed_cells = getattr(diagnostics, "failed_cells", ())
    try:
        failed_cells = tuple(int(value) for value in raw_failed_cells)
    except (TypeError, ValueError, OverflowError):
        failed_cells = ()

    raw_reason = getattr(diagnostics, "reason", "max_iterations")
    reasons = {
        "tolerance",
        "max_iterations",
        "line_search",
        "breakdown",
        "nonfinite",
        "invalid_state",
    }
    reason = raw_reason if raw_reason in reasons else "max_iterations"
    return convergence_diagnostics(
        stage=stage,
        converged=False,
        reason=reason,  # type: ignore[arg-type]
        iterations=max(0, iterations),
        max_iterations=max(0, max_iterations),
        initial_residual_norm=float("inf"),
        residual_norm=float("inf"),
        residual_history=(),
        failed_cells=failed_cells,
    )


__all__ = [
    "FlowConvergenceDiagnostics",
    "convergence_diagnostics",
    "normalize_convergence_diagnostics",
]
