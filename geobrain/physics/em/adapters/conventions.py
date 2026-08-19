"""Explicit tensor-only Fourier and phase-boundary adapters for the EM family.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from geobrain.core.errors import ErrorCode

from ..errors import EMContractError
from .units import _scale_with_representability


def _require_scalable_tensor(
    value: object,
    *,
    adapter: str,
    complex_only: bool = False,
) -> torch.Tensor:
    """Require a live floating/complex tensor without casting or moving it."""
    if not isinstance(value, torch.Tensor):
        raise EMContractError(
            "EM adapter input must be a torch.Tensor",
            object_name=adapter,
            field="value",
            expected="torch.Tensor",
            actual=type(value).__qualname__,
            details={
                "adapter": adapter,
                "received_type": type(value).__qualname__,
                "remediation": "pass a floating or native-complex torch.Tensor",
            },
        )
    valid_dtype = value.is_complex() if complex_only else (
        value.is_floating_point() or value.is_complex()
    )
    if not valid_dtype:
        requirement = "native-complex tensor" if complex_only else "floating or native-complex tensor"
        raise EMContractError(
            "EM adapter dtype is unsupported",
            object_name=adapter,
            field="value.dtype",
            expected=requirement,
            actual=str(value.dtype),
            code=ErrorCode.DTYPE_UNSUPPORTED,
            details={
                "adapter": adapter,
                "dtype": str(value.dtype),
                "remediation": f"pass a {requirement}",
            },
        )
    return value


def to_minus_iwt_complex(value: torch.Tensor) -> torch.Tensor:
    """Convert a native-complex ``+iωt`` value to explicit ``-iωt`` form."""
    tensor = _require_scalable_tensor(
        value,
        adapter="to_minus_iwt_complex",
        complex_only=True,
    )
    return tensor.conj()


def from_legacy_mt2d_native(
    value: torch.Tensor,
    mode: Literal["te", "tm"],
) -> torch.Tensor:
    """Map historical MT2D native TE/TM impedances into canonical ``+iωt``."""
    tensor = _require_scalable_tensor(
        value,
        adapter="from_legacy_mt2d_native",
        complex_only=True,
    )
    if type(mode) is not str or mode not in ("te", "tm"):
        raise EMContractError(
            "legacy MT2D mode must be 'te' or 'tm'",
            object_name="from_legacy_mt2d_native",
            field="mode",
            expected=("te", "tm"),
            actual=mode,
            details={
                "received_type": type(mode).__qualname__,
                "remediation": "select mode='te' or mode='tm' explicitly",
            },
        )
    conjugated = tensor.conj()
    return -conjugated if mode == "te" else conjugated


def phase_radians_to_degrees(value: torch.Tensor) -> torch.Tensor:
    """Scale an explicit phase tensor from radians to display degrees."""
    tensor = _require_scalable_tensor(value, adapter="phase_radians_to_degrees")
    return _scale_with_representability(
        tensor,
        factor=180.0 / math.pi,
        adapter="phase_radians_to_degrees",
    )


def phase_degrees_to_radians(value: torch.Tensor) -> torch.Tensor:
    """Scale an explicit display phase tensor from degrees to radians."""
    tensor = _require_scalable_tensor(value, adapter="phase_degrees_to_radians")
    return _scale_with_representability(
        tensor,
        factor=math.pi / 180.0,
        adapter="phase_degrees_to_radians",
    )


__all__ = [
    "from_legacy_mt2d_native",
    "phase_degrees_to_radians",
    "phase_radians_to_degrees",
    "to_minus_iwt_complex",
]
