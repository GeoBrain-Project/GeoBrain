"""Strict result assembly for the internal Wave propagation engine.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import cast

from geobrain.core import ForwardOutput
import torch

from ..errors import WaveContractError, WaveNumericsError
from .contracts import PropagationResult


def _strict_json_value(value: object, active: set[int]) -> None:
    """Validate recursively against finite JSON primitives without conversion."""
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if math.isfinite(value):
            return
        raise TypeError("non-finite float")
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise TypeError("cyclic mapping")
        active.add(identity)
        try:
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError("JSON object keys must be strings")
                _strict_json_value(item, active)
        finally:
            active.remove(identity)
        return
    if type(value) in (list, tuple):
        identity = id(value)
        if identity in active:
            raise TypeError("cyclic sequence")
        active.add(identity)
        try:
            for item in cast(Sequence[object], value):
                _strict_json_value(item, active)
        finally:
            active.remove(identity)
        return
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    """Reject a non-finite real or complex live tensor."""
    with torch.no_grad():
        finite = bool(torch.isfinite(tensor).all())
    if not finite:
        raise WaveNumericsError(
            "non-finite Wave propagation result",
            object_name="PropagationResult",
            field=name,
            expected="finite real and imaginary tensor components",
            actual="non-finite tensor",
        )


def assemble_forward_output(
    result: PropagationResult,
    *,
    axis_names: Sequence[str],
    units: Mapping[str, object],
    component_order: Sequence[str],
    survey_fingerprint: str,
    backend: str,
    strategy: str,
    maturity: str,
    quality_status: str,
    differentiability: str,
    points_per_wavelength: float | None = None,
    cfl_ratio: float | None = None,
    equation: str | None = None,
    sampling: Mapping[str, object] | None = None,
) -> ForwardOutput:
    """Validate and assemble an unchanged core ForwardOutput."""
    if not result.complete:
        raise WaveNumericsError(
            "incomplete Wave propagation result",
            object_name="PropagationResult",
            field="complete",
            expected=True,
            actual=False,
        )
    _require_finite("traces", result.traces)
    for name, tensor in result.fields.items():
        _require_finite(name, tensor)
    metadata: dict[str, object] = {
        "axis_names": tuple(axis_names),
        "units": dict(units),
        "component_order": tuple(component_order),
        "survey_fingerprint": survey_fingerprint,
        "backend": backend,
        "strategy": strategy,
        "maturity": maturity,
        "quality_status": quality_status,
        "differentiability": differentiability,
    }
    if equation is not None:
        metadata["equation"] = equation
    if sampling is not None:
        metadata["sampling"] = dict(sampling)
    if points_per_wavelength is not None:
        metadata["points_per_wavelength"] = points_per_wavelength
    if cfl_ratio is not None:
        metadata["cfl_ratio"] = cfl_ratio
    try:
        _strict_json_value(metadata, set())
        _strict_json_value(result.diagnostics, set())
    except TypeError as exc:
        raise WaveContractError(
            "Wave result metadata must contain finite JSON primitives only",
            object_name="assemble_forward_output",
            field="metadata",
            expected="nested finite JSON primitives without Tensor or NumPy values",
            actual=str(exc),
        ) from exc
    return ForwardOutput(
        data={"seismic": result.traces},
        fields=result.fields,
        metadata=metadata,
    )


__all__ = ["assemble_forward_output"]
