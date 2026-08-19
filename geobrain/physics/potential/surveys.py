"""Immutable SI survey contracts for Potential operators.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from geobrain.core import ErrorCode

from .errors import PotentialCapabilityError, PotentialContractError


def _validated_positions(
    positions_m: object,
    *,
    width: int,
    object_name: str,
) -> torch.Tensor:
    if not isinstance(positions_m, torch.Tensor):
        raise PotentialContractError(
            "Survey coordinates must be a torch.Tensor.",
            object_name=object_name,
            field="positions_m",
            expected=f"non-empty (n, {width}) CPU float64 Tensor",
            actual=positions_m,
            hint=f"Provide packed metre coordinates with exactly {width} columns.",
        )
    if positions_m.layout != torch.strided:
        raise PotentialCapabilityError(
            "Canonical survey coordinates require a strided tensor layout.",
            object_name=object_name,
            field="positions_m",
            expected="torch.strided",
            actual=positions_m.layout,
            code=ErrorCode.CAPABILITY_UNAVAILABLE,
            hint="Materialize coordinates in a strided tensor before constructing the survey.",
        )
    if positions_m.ndim != 2 or positions_m.shape[0] == 0 or positions_m.shape[1] != width:
        raise PotentialContractError(
            "Survey coordinates have the wrong shape.",
            object_name=object_name,
            field="positions_m",
            expected=["n>0", width],
            actual=list(positions_m.shape),
            code=ErrorCode.SHAPE_MISMATCH,
            hint=f"Provide a non-empty tensor shaped (n, {width}).",
        )
    if positions_m.dtype != torch.float64:
        raise PotentialContractError(
            "Canonical survey coordinates must use float64.",
            object_name=object_name,
            field="positions_m",
            expected="torch.float64",
            actual=positions_m.dtype,
            code=ErrorCode.DTYPE_UNSUPPORTED,
            hint="Construct canonical survey coordinates with dtype=torch.float64.",
        )
    if positions_m.device.type != "cpu":
        raise PotentialContractError(
            "Canonical survey coordinates must reside on CPU.",
            object_name=object_name,
            field="positions_m",
            expected="cpu",
            actual=positions_m.device,
            code=ErrorCode.DEVICE_UNAVAILABLE,
            hint="Construct the canonical survey on CPU; execution compiles owned geometry explicitly.",
        )
    if positions_m.requires_grad:
        raise PotentialContractError(
            "Canonical survey coordinates cannot own gradient history.",
            object_name=object_name,
            field="positions_m",
            expected="requires_grad=False",
            actual=True,
            hint="Pass a non-gradient canonical coordinate tensor; no silent detach is performed.",
        )
    if not bool(torch.isfinite(positions_m).all()):
        raise PotentialContractError(
            "Survey coordinates must be finite.",
            object_name=object_name,
            field="positions_m",
            expected="all finite metre values",
            actual="contains non-finite values",
            hint="Replace NaN or infinite coordinates before constructing the survey.",
        )
    return positions_m.detach().clone(memory_format=torch.contiguous_format)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PotentialSurvey2D:
    """Owned canonical 2-D positions with columns ``(x, z)`` in metres.

    The ``z`` column is ELEVATION (positive UP), sign-opposite to the
    platform depth-down mesh axis. The operator converts internally
    (``station_depth = -positions[:, 1]``), so a station 5 m above the
    surface has ``z = +5.0``; feeding depth-down values silently mirrors
    the acquisition geometry.
    """

    _positions_m: torch.Tensor = field(repr=False)

    def __init__(self, positions_m: torch.Tensor) -> None:
        object.__setattr__(
            self,
            "_positions_m",
            _validated_positions(positions_m, width=2, object_name=type(self).__name__),
        )

    @property
    def positions_m(self) -> torch.Tensor:
        """Return a detached clone of the owned canonical coordinates."""
        return self._positions_m.detach().clone(memory_format=torch.contiguous_format)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PotentialSurvey3D:
    """Owned canonical 3-D positions with columns ``(x, y, z)`` in metres.

    Columns are world order with ``z`` ELEVATION (positive UP),
    sign-opposite to the platform depth-down mesh axis. The engine bridge
    (``prism_mesh_cell_bounds_m``) converts MESH geometry into this frame,
    never the survey into mesh order; feeding depth-down ``z`` silently
    mirrors the acquisition geometry.
    """

    _positions_m: torch.Tensor = field(repr=False)

    def __init__(self, positions_m: torch.Tensor) -> None:
        object.__setattr__(
            self,
            "_positions_m",
            _validated_positions(positions_m, width=3, object_name=type(self).__name__),
        )

    @property
    def positions_m(self) -> torch.Tensor:
        """Return a detached clone of the owned canonical coordinates."""
        return self._positions_m.detach().clone(memory_format=torch.contiguous_format)


__all__ = ["PotentialSurvey2D", "PotentialSurvey3D"]
