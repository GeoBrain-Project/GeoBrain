"""
Abstract base + helpers for AVO reflectivity models.

These classes compute **pure reflection-coefficient math** as 1-D ``nn.Module``
callables, built on the ``geobrain.core.registry`` infrastructure. They are the
math core that the ``ForwardOperator`` AVO classes in ``wave.reflectivity.operators``
(``AkiRichards`` / ``Shuey`` / ``Zoeppritz`` / ``ConvolutionalAVO``) wrap in the
``ModelState`` / ``ForwardContext`` contract for use in the forward graph:

- Use **these models** when you have explicit interface pairs
  ``(vp1, vs1, rho1, vp2, vs2, rho2)`` and want ``R(θ)``.
- Use the **operators** (``wave.reflectivity.operators``) when you have a 1-D depth
  profile and need a differentiable operator in a chain.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import TypeVar, cast

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ....core import GeoBrainError

EPS = 1e-10
_TensorFunction = TypeVar("_TensorFunction", bound=Callable[..., Tensor])


def require_compatible_tensors(
    named_tensors: tuple[tuple[str, Tensor], ...],
    *,
    owner: str,
    allowed_dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
) -> tuple[torch.dtype, torch.device]:
    """Require live tensors to share dtype/device before arithmetic."""
    if not named_tensors:
        return torch.float32, torch.device("cpu")
    reference_name, reference = named_tensors[0]
    if reference.dtype not in allowed_dtypes:
        raise GeoBrainError(
            f"{owner} tensors use an unsupported dtype",
            object_name=owner,
            field=reference_name,
            expected=tuple(str(dtype) for dtype in allowed_dtypes),
            actual=str(reference.dtype),
        )
    for name, tensor in named_tensors[1:]:
        if tensor.dtype != reference.dtype:
            raise GeoBrainError(
                f"{owner} live Tensor dtype values must match",
                object_name=owner,
                field=name,
                expected=str(reference.dtype),
                actual=str(tensor.dtype),
            )
        if tensor.device != reference.device:
            raise GeoBrainError(
                f"{owner} live Tensor device values must match",
                object_name=owner,
                field=name,
                expected=str(reference.device),
                actual=str(tensor.device),
            )
    return reference.dtype, reference.device


def normalize_reflectivity_inputs(
    named_values: tuple[tuple[str, object], ...], *, owner: str
) -> tuple[Tensor, ...]:
    """Normalize Python/NumPy values around an unchanged live Tensor schema."""
    numpy_values = tuple(
        (name, value)
        for name, value in named_values
        if isinstance(value, np.ndarray)
    )
    for name, value in numpy_values:
        if np.issubdtype(value.dtype, np.complexfloating):
            raise GeoBrainError(
                f"{owner} complex NumPy inputs are unsupported",
                object_name=owner,
                field=name,
                expected="real-valued NumPy input",
                actual=str(value.dtype),
            )
    live = tuple(
        (name, value) for name, value in named_values if isinstance(value, Tensor)
    )
    if live:
        dtype, device = require_compatible_tensors(live, owner=owner)
    else:
        dtype = (
            torch.float64
            if any(value.dtype == np.float64 for _, value in numpy_values)
            else torch.float32
        )
        device = torch.device("cpu")
    return tuple(as_tensor(value, dtype=dtype, device=device) for _, value in named_values)


def as_tensor(
    x: object,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """
    Convert input to a tensor, optionally casting dtype/device.

    The ``dtype`` / ``device`` kwargs are honoured for every input path
    (tensor, numpy, scalar), important because AVO callers often mix
    numpy arrays from log data with torch tensors from earlier
    differentiable ops, and silent dtype/device drift would propagate
    through the chain.
    """
    if isinstance(x, torch.Tensor):
        tensor = cast(Tensor, x)
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    if isinstance(x, np.ndarray):
        out = torch.from_numpy(np.ascontiguousarray(x))
        if dtype is not None:
            out = out.to(dtype)
        elif out.dtype not in (
            torch.float32, torch.float64, torch.complex64, torch.complex128,
        ):
            out = out.to(torch.float32)
        if device is not None:
            out = out.to(device)
        return out
    return torch.tensor(x, dtype=dtype or torch.float32, device=device)


def deg2rad(x: Tensor | float) -> Tensor | float:
    return x * (np.pi / 180.0)


def vectorize_angles(func: _TensorFunction) -> _TensorFunction:
    """
    Decorator that tensor-ifies inputs, validates ``θ ∈ [0, 90]°``, converts to
    radians, and broadcasts over the property axes.
    """

    @wraps(func)
    def wrapper(
        self: object,
        vp1: object,
        vs1: object,
        rho1: object,
        vp2: object,
        vs2: object,
        rho2: object,
        theta: object,
        **kwargs: object,
    ) -> Tensor:
        vp1_t, vs1_t, rho1_t, vp2_t, vs2_t, rho2_t, theta_t = normalize_reflectivity_inputs(
            (
                ("vp1", vp1),
                ("vs1", vs1),
                ("rho1", rho1),
                ("vp2", vp2),
                ("vs2", vs2),
                ("rho2", rho2),
                ("theta", theta),
            ),
            owner=type(self).__name__,
        )
        if theta_t.ndim == 0:
            theta_t = theta_t.unsqueeze(0)

        if (theta_t > 90.0).any():
            raise GeoBrainError(
                "Incidence angle must be <= 90 degrees",
                field="theta", expected="<= 90 degrees",
                actual=float(theta_t.max()),
            )
        if (theta_t < 0).any():
            raise GeoBrainError(
                "Incidence angle must be >= 0 degrees",
                field="theta", expected=">= 0 degrees",
                actual=float(theta_t.min()),
            )

        theta_rad = theta_t * (np.pi / 180.0)

        n_dims = vp1_t.ndim
        reshape = [theta_t.numel()] + [1] * n_dims
        theta_rad = theta_rad.reshape(*reshape)

        return func(
            self,
            vp1_t,
            vs1_t,
            rho1_t,
            vp2_t,
            vs2_t,
            rho2_t,
            theta_rad,
            **kwargs,
        )

    return cast(_TensorFunction, wrapper)


class AVOModel(nn.Module):  # type: ignore[misc]  # isolated strict import boundary
    """
    Abstract base for P-P AVO reflectivity models.

    Subclasses implement :meth:`forward` returning a tensor with leading
    dimension ``n_angles`` (the rest broadcast from the property inputs).
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "base"
        self._valid_angles = (0, 90)

    @abstractmethod
    def forward(
        self,
        vp1: Tensor, vs1: Tensor, rho1: Tensor,
        vp2: Tensor, vs2: Tensor, rho2: Tensor,
        theta: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(valid_angles={self._valid_angles})"

    @property
    def name(self) -> str:
        return self._name


def normal_incidence_rc(
    vp1: object, rho1: object, vp2: object, rho2: object
) -> Tensor:
    """Normal-incidence P-P reflection coefficient ``(Z2 − Z1) / (Z2 + Z1)``."""
    vp1_t, rho1_t, vp2_t, rho2_t = normalize_reflectivity_inputs(
        (("vp1", vp1), ("rho1", rho1), ("vp2", vp2), ("rho2", rho2)),
        owner="normal_incidence_rc",
    )
    Z1 = vp1_t * rho1_t
    Z2 = vp2_t * rho2_t
    return (Z2 - Z1) / (Z2 + Z1 + EPS)


__all__ = [
    "AVOModel",
    "EPS",
    "as_tensor",
    "deg2rad",
    "normal_incidence_rc",
    "require_compatible_tensors",
    "vectorize_angles",
]
