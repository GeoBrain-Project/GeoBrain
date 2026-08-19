"""Private assembly of latent and transformed inversion results.

Execution loops supply truthful latent snapshots and histories. This module
first validates them as a complete :class:`InversionResult`, then optionally
maps independent final and best snapshots into physical parameter space.
Transform failures retain that complete latent result instead of attempting to
reuse the finite-only execution-partial contract.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from types import MappingProxyType
from typing import Any, Callable, Mapping, NoReturn, Sequence

import torch

from geobrain.core import GeoBrainError
from geobrain.core.validation import validate_param_mapping

from .execution import StopReason
from .results import InversionResult

_ResultTransform = Callable[
    [Mapping[str, torch.Tensor]],
    Mapping[str, torch.Tensor],
]


def _transform_params(
    transform: _ResultTransform,
    params: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    """Transform an independent owned snapshot and validate physical fields."""
    snapshot = MappingProxyType(
        {
            name: tensor.detach().clone()
            for name, tensor in params.items()
        }
    )
    with torch.no_grad():
        transformed = transform(snapshot)
    if not isinstance(transformed, MappingABC):
        raise GeoBrainError(
            "Inverter result_transform must return a mapping",
            object_name="Inverter.run",
            field="result_transform",
            expected="non-empty mapping of physical tensor fields",
            actual=type(transformed),
        )
    validate_param_mapping(
        transformed,
        "Inverter.run.result_transform",
        require_grad=False,
    )
    return transformed


def _raise_transform_error(
    error: Exception,
    *,
    latent_result: InversionResult,
) -> NoReturn:
    """Attach the truthful latent terminal result and preserve error policy."""
    if isinstance(error, GeoBrainError):
        setattr(error, "partial_result", latent_result)
        raise error
    wrapped = GeoBrainError(
        "Inverter.run failed during result_transform",
        object_name="Inverter.run",
        field="result_transform",
        expected="successful physical result transform",
        actual=type(error),
    )
    setattr(wrapped, "partial_result", latent_result)
    raise wrapped from error


def _physical_result(
    latent: InversionResult,
    transform: _ResultTransform,
) -> InversionResult:
    """Map corresponding latent final/best snapshots into a physical result."""
    try:
        final_params = _transform_params(transform, latent.params)
        best_params = (
            None
            if latent.best_params is None
            else _transform_params(transform, latent.best_params)
        )
        metadata: dict[str, Any] = {
            **latent.metadata,
            "parameter_space": "physical",
            "optimization_parameter_space": "latent",
            "latent_params": latent.params,
        }
        if latent.best_params is not None:
            metadata["best_latent_params"] = latent.best_params
        return InversionResult(
            params=final_params,
            requested_iters=latent.requested_iters,
            completed_iters=latent.completed_iters,
            stop_reason=latent.stop_reason,
            loss_history=latent.loss_history,
            metadata=metadata,
            best_params=best_params,
            data_loss_history=latent.data_loss_history,
            reg_loss_history=latent.reg_loss_history,
            best_loss=latent.best_loss,
            best_iter=latent.best_iter,
            wall_clock_sec=latent.wall_clock_sec,
            converged=latent.converged,
            term_losses=latent.term_losses,
        )
    except Exception as error:
        _raise_transform_error(error, latent_result=latent)


def _assemble_inversion_result(
    *,
    latent_params: Mapping[str, torch.Tensor],
    requested_iters: int,
    completed_iters: int,
    stop_reason: StopReason,
    loss_history: torch.Tensor,
    metadata: Mapping[str, Any],
    best_latent_params: Mapping[str, torch.Tensor] | None,
    data_loss_history: Sequence[float],
    reg_loss_history: Sequence[float],
    best_loss: float,
    best_iter: int | None,
    wall_clock_sec: float,
    term_losses: Mapping[str, Sequence[float]] | None,
    result_transform: _ResultTransform | None,
) -> InversionResult:
    """Build a complete latent result before any optional physical transform."""
    latent_metadata = dict(metadata)
    if result_transform is not None:
        latent_metadata["parameter_space"] = "latent"
    latent = InversionResult(
        params=latent_params,
        requested_iters=requested_iters,
        completed_iters=completed_iters,
        stop_reason=stop_reason,
        loss_history=loss_history,
        metadata=latent_metadata,
        best_params=best_latent_params,
        data_loss_history=torch.tensor(
            data_loss_history,
            dtype=torch.float64,
        ),
        reg_loss_history=torch.tensor(
            reg_loss_history,
            dtype=torch.float64,
        ),
        best_loss=best_loss,
        best_iter=best_iter,
        wall_clock_sec=wall_clock_sec,
        converged=stop_reason is StopReason.CALLBACK,
        term_losses=term_losses,
    )
    if result_transform is None:
        return latent
    return _physical_result(latent, result_transform)
