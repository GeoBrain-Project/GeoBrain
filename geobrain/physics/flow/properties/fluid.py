"""
Fluid system containers (single-phase, oil-water, black-oil 3-phase).

Each system is an ``nn.Module`` that bundles :class:`PVT` + :class:`RelPerm`
descriptors and exposes ``phase_density(p)``, ``phase_viscosity(p)``,
``phase_fvf(p)`` etc. as torch-tensor → torch-tensor functions
(autograd-friendly).

The :class:`PhaseState` is a typed bag of ``torch.Tensor`` fields used
by Newton / time-stepping code to thread the per-cell phase variables
through a forward solve.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..errors import FlowContractError
from .pvt import PVT
from .relperm import RelPerm


@dataclass
class PhaseState:
    """
    Snapshot of per-cell phase variables at one timestep.

    Tensors are CPU/GPU-agnostic; the caller brings whatever device they
    live on. Physical quantities use canonical SI.

    Attributes:
        pressure: ``(n_cells,)`` oil-phase pressure [Pa].
        sw:       ``(n_cells,)`` water saturation [-]. Optional.
        sg:       ``(n_cells,)`` gas saturation [-]. Optional.
        rs:       ``(n_cells,)`` solution ratio [standard m³/stock-tank m³]. Optional
            (live-oil PVT only).
    """

    pressure: torch.Tensor
    sw: torch.Tensor | None = None
    sg: torch.Tensor | None = None
    rs: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.pressure, torch.Tensor)
            or not self.pressure.is_floating_point()
            or self.pressure.ndim != 1
        ):
            raise FlowContractError(
                "PhaseState pressure must be a floating cell vector",
                object_name="PhaseState",
                field="pressure",
                expected="floating [cell] tensor in Pa",
                actual=(
                    type(self.pressure).__name__,
                    tuple(getattr(self.pressure, "shape", ())),
                ),
            )
        if not bool(torch.isfinite(self.pressure).all()) or bool((self.pressure <= 0).any()):
            raise FlowContractError(
                "PhaseState pressure must be positive and finite",
                object_name="PhaseState",
                field="pressure",
                expected="> 0 Pa",
                actual="contains a non-positive or non-finite value",
            )
        for field in ("sw", "sg", "rs"):
            value = getattr(self, field)
            if value is None:
                continue
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.shape != self.pressure.shape
                or value.dtype != self.pressure.dtype
                or value.device != self.pressure.device
            ):
                raise FlowContractError(
                    f"PhaseState {field} metadata must match pressure",
                    object_name="PhaseState",
                    field=f"{field}.shape/dtype/device",
                    expected=(
                        tuple(self.pressure.shape),
                        str(self.pressure.dtype),
                        str(self.pressure.device),
                    ),
                    actual=(
                        type(value).__name__,
                        tuple(getattr(value, "shape", ())),
                        str(getattr(value, "dtype", None)),
                        str(getattr(value, "device", None)),
                    ),
                )
            invalid = (value < 0) | (value > 1) if field in ("sw", "sg") else value < 0
            if not bool(torch.isfinite(value).all()) or bool(invalid.any()):
                raise FlowContractError(
                    f"PhaseState {field} is outside its physical domain",
                    object_name="PhaseState",
                    field=field,
                    expected="finite [0, 1]" if field in ("sw", "sg") else "finite >= 0",
                    actual="contains a non-finite or out-of-domain value",
                )

    @property
    def n_cells(self) -> int:
        return int(self.pressure.shape[0])

    def clone(self) -> "PhaseState":
        return PhaseState(
            pressure=self.pressure.clone(),
            sw=None if self.sw is None else self.sw.clone(),
            sg=None if self.sg is None else self.sg.clone(),
            rs=None if self.rs is None else self.rs.clone(),
        )

    def to(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> "PhaseState":
        def move(value: torch.Tensor) -> torch.Tensor:
            if dtype is None:
                return value.to(device=device)
            return value.to(device=device, dtype=dtype)

        return PhaseState(
            pressure=move(self.pressure),
            sw=None if self.sw is None else move(self.sw),
            sg=None if self.sg is None else move(self.sg),
            rs=None if self.rs is None else move(self.rs),
        )


# --- Single-phase fluid (groundwater / dead-oil Darcy) --------------------


class SinglePhaseFluid(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """Single-phase Darcy fluid wrapping one :class:`PVT` model."""

    def __init__(self, pvt: PVT) -> None:
        super().__init__()
        self.pvt = pvt

    def density(self, pressure: torch.Tensor) -> torch.Tensor:
        return self.pvt.density(pressure)

    def viscosity(self, pressure: torch.Tensor) -> torch.Tensor:
        return self.pvt.viscosity(pressure)

    def fvf(self, pressure: torch.Tensor) -> torch.Tensor:
        return self.pvt.fvf(pressure)


# --- Two-phase oil-water --------------------------------------------------


class OilWaterFluid(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """
    Oil + water two-phase fluid system.

    Bundles PVT for both phases plus the relative-permeability model.
    Optional capillary-pressure module follows the
    :class:`~geobrain.physics.flow.properties.BrooksCoreyPc` interface.
    """

    def __init__(
        self,
        pvt_o: PVT,
        pvt_w: PVT,
        relperm: RelPerm,
        capillary: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.pvt_o = pvt_o
        self.pvt_w = pvt_w
        self.relperm = relperm
        self.capillary = capillary

    def density_oil(self, p: torch.Tensor) -> torch.Tensor:
        return self.pvt_o.density(p)

    def density_water(self, p: torch.Tensor) -> torch.Tensor:
        return self.pvt_w.density(p)

    def viscosity_oil(self, p: torch.Tensor) -> torch.Tensor:
        return self.pvt_o.viscosity(p)

    def viscosity_water(self, p: torch.Tensor) -> torch.Tensor:
        return self.pvt_w.viscosity(p)

    def fvf_oil(self, p: torch.Tensor) -> torch.Tensor:
        return self.pvt_o.fvf(p)

    def fvf_water(self, p: torch.Tensor) -> torch.Tensor:
        return self.pvt_w.fvf(p)

    def kr_oil(self, sw: torch.Tensor) -> torch.Tensor:
        return self.relperm.kr_oil(sw)

    def kr_water(self, sw: torch.Tensor) -> torch.Tensor:
        return self.relperm.kr_water(sw)


# --- Three-phase dead-oil black-oil ---------------------------------------


class BlackOilFluid(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """
    Three-phase black-oil fluid container.

    It supports dead-oil PVT (``R_s=0``), saturated live oil, and the explicit
    variable-switching live-oil model. The concrete model schema determines
    which primary variables are stored; derived phase properties remain here.
    """

    def __init__(
        self,
        pvt_o: PVT,
        pvt_w: PVT,
        pvt_g: PVT,
        relperm: RelPerm,
        capillary_ow: nn.Module | None = None,
        capillary_og: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.pvt_o = pvt_o
        self.pvt_w = pvt_w
        self.pvt_g = pvt_g
        self.relperm = relperm
        self.capillary_ow = capillary_ow
        self.capillary_og = capillary_og


__all__ = [
    "BlackOilFluid",
    "OilWaterFluid",
    "PhaseState",
    "SinglePhaseFluid",
]
